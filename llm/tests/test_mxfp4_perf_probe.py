"""CPU-only contracts for the optional NPU performance diagnostics."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "llm/diagnostics/mxfp4_perf_probe.py"
CONFIG = {
    "model_type": "qwen3_5_moe",
    "hidden_size": 2048,
    "moe_intermediate_size": 512,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
}


def load_probe():
    spec = importlib.util.spec_from_file_location("perf_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPerfProbe(unittest.TestCase):
    def test_timing_warms_up_excludes_setup_and_alternates_order(self):
        probe = load_probe()
        state = {"ticks": 0, "calls": []}

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                self.tick = state["ticks"]

            def elapsed_time(self, other):
                return other.tick - self.tick

        def run(name, cost):
            state["calls"].append(name)
            state["ticks"] += cost

        npu = SimpleNamespace(Event=Event, synchronize=lambda: None)
        results = probe.measure_cases(
            {"a": lambda: run("a", 2), "b": lambda: run("b", 5)},
            npu,
            warmup=2,
            iterations=3,
            repeats=2,
            mode="eager",
        )
        self.assertEqual(results["a"]["stream_ms"], [2, 2])
        self.assertEqual(results["b"]["stream_ms"], [5, 5])
        self.assertEqual(
            state["calls"], ["a", "a", "b", "b"] + ["a"] * 3 + ["b"] * 6 + ["a"] * 3
        )
        self.assertEqual(len(results["a"]["wall_ms"]), 2)

    def test_graph_unroll_amortizes_launch_and_reports_raw_times(self):
        probe = load_probe()

        def check_unroll(unroll):
            with self.subTest(unroll=unroll):
                state = {"ticks": 0, "capture": None, "calls": [], "replays": []}
                streams, graphs = [], []

                class Stream:
                    def __init__(self):
                        streams.append(self)

                    def wait_stream(self, other):
                        pass

                current = [Stream()]

                @contextmanager
                def stream_context(stream):
                    previous, current[0] = current[0], stream
                    try:
                        yield
                    finally:
                        current[0] = previous

                class Graph:
                    def __init__(self):
                        self.cost = 0
                        self.nodes = 0
                        graphs.append(self)

                    def replay(self):
                        state["replays"].append(self)
                        # Deterministic model of a fixed 160 us launch cost.
                        state["ticks"] += 0.160 + self.cost

                @contextmanager
                def graph_context(graph, stream):
                    with stream_context(stream):
                        state["capture"] = graph
                        yield
                        state["capture"] = None

                class Event:
                    def __init__(self, **kwargs):
                        pass

                    def record(self):
                        self.tick = state["ticks"]
                        self.stream = current[0]

                    def elapsed_time(self, other):
                        self_check.assertIs(self.stream, other.stream)
                        return other.tick - self.tick

                def run(name, cost):
                    state["calls"].append(name)
                    if state["capture"] is not None:
                        state["capture"].cost += cost
                        state["capture"].nodes += 1
                    else:
                        state["ticks"] += cost
                    return name

                self_check = self
                npu = SimpleNamespace(
                    Stream=Stream,
                    current_stream=lambda: current[0],
                    stream=stream_context,
                    NPUGraph=Graph,
                    graph=graph_context,
                    Event=Event,
                    synchronize=lambda: None,
                )
                results = probe.measure_cases(
                    {"a": lambda: run("a", 0.004), "b": lambda: run("b", 0.010)},
                    npu,
                    warmup=2,
                    iterations=3,
                    repeats=2,
                    mode="graph",
                    graph_unroll=unroll,
                )
                for name, cost in (("a", 0.004), ("b", 0.010)):
                    values = results[name]
                    self.assertEqual(values["calls_per_replay"], unroll)
                    self.assertEqual(values["replays_per_round"], 3)
                    for elapsed, raw in zip(
                        values["stream_ms"], values["replay_stream_ms"]
                    ):
                        self.assertAlmostEqual(elapsed, cost + 0.160 / unroll)
                        self.assertAlmostEqual(raw, elapsed * unroll)
                    for wall, raw in zip(values["wall_ms"], values["replay_wall_ms"]):
                        self.assertAlmostEqual(raw, wall * unroll)
                    self.assertEqual(len(values["host_submit_ms_per_replay"]), 2)
                    self.assertEqual(state["calls"].count(name), 2 + unroll)
                self.assertEqual([g.nodes for g in graphs], [unroll, unroll])
                # Reuse a capture stream, rather than one new stream per case.
                self.assertEqual(len(streams), 2)
                a, b = graphs
                self.assertEqual(
                    state["replays"], [a] * 2 + [b] * 2 + [a] * 3 + [b] * 6 + [a] * 3
                )

        for unroll in (1, 16, 64):
            check_unroll(unroll)

    def test_graph_replay_check_rejects_stale_output(self):
        import torch

        probe = load_probe()
        zeros = torch.zeros
        for stale in (False, True):
            with self.subTest(stale=stale):

                def capture(fn, npu, warmup, stale=stale, **kwargs):
                    output = fn()

                    def replay():
                        if not stale:
                            output.copy_(fn())
                        return output

                    return replay

                with (
                    patch.object(probe, "capture", side_effect=capture),
                    patch.object(
                        torch,
                        "zeros",
                        side_effect=lambda *a, **kw: zeros(
                            *a, **{**kw, "device": "cpu"}
                        ),
                    ),
                ):
                    npu = SimpleNamespace(synchronize=lambda: None)
                    if stale:
                        with self.assertRaisesRegex(RuntimeError, "stale|incorrect"):
                            probe.check_graph_replay(torch, npu)
                    else:
                        self.assertEqual(
                            probe.check_graph_replay(torch, npu)[
                                "changed_input_replays"
                            ],
                            2,
                        )

    def test_graph_unroll_cli_is_explicit_and_positive(self):
        probe = load_probe()
        args = probe.parser().parse_args(["--model-path", ".", "--mode", "graph"])
        self.assertEqual(args.graph_unroll, [1, 16, 64])
        args = probe.parser().parse_args(
            ["--model-path", ".", "--graph-unroll", "8", "32"]
        )
        self.assertEqual(args.graph_unroll, [8, 32])
        with self.assertRaises(SystemExit):
            probe.parser().parse_args(["--model-path", ".", "--graph-unroll", "0"])

    def test_graph_sweep_emits_control_and_unroll_without_requantizing_weights(self):
        import torch

        probe = load_probe()
        records = []
        args = SimpleNamespace(
            sections=["gdn"],
            tokens=[32],
            mode="graph",
            graph_unroll=[1, 16, 16, 64],
            warmup=2,
            iterations=3,
            repeats=2,
        )
        plan = {"hidden_size": 64, "gdn_qkvz_n": 128, "gdn_ba_n": 8}
        npu = SimpleNamespace(empty_cache=lambda: None)
        zeros = torch.zeros
        seen = []

        def measure(cases, npu, **kwargs):
            unroll = kwargs["graph_unroll"]
            seen.append(unroll)
            self.assertEqual(list(cases), ["graph_control_add", "example"])
            self.assertTrue(torch.equal(cases["graph_control_add"](), torch.ones(4)))
            return {
                name: {
                    "stream_ms": [0.16 / unroll] * 2,
                    "wall_ms": [0.17 / unroll] * 2,
                    "calls_per_replay": unroll,
                    "replays_per_round": 3,
                }
                for name in cases
            }

        with (
            patch.object(torch, "npu", npu, create=True),
            patch.object(
                torch,
                "zeros",
                side_effect=lambda *a, **kw: zeros(*a, **{**kw, "device": "cpu"}),
            ),
            patch.object(probe, "make_layers", return_value=None) as make_layers,
            patch.object(
                probe,
                "gdn_cases",
                return_value=({"example": lambda: None}, {"test": True}),
            ),
            patch.object(probe, "measure_cases", side_effect=measure),
        ):
            probe.run_sections(SimpleNamespace(torch=torch), args, plan, records.append)
        self.assertEqual(make_layers.call_count, 2)
        self.assertEqual(seen, [1, 16, 64])
        self.assertEqual(records[0]["evaluation_mode"], "eager_before_timing")
        self.assertEqual(
            [r["calls_per_replay"] for r in records[1:]], [1, 1, 16, 16, 64, 64]
        )
        self.assertEqual(records[-1]["case"], "example")
        self.assertAlmostEqual(records[-1]["median_stream_ms"], 0.16 / 64)

    def test_gdn_ablation_counts_only_activation_quant_in_timed_cases(self):
        import torch

        probe = load_probe()
        counts = {"weight": 0, "activation": 0}

        class Dense:
            def _quantize_weights(self, layer):
                counts["weight"] += 1

            def prepare_input(self, x):
                counts["activation"] += 1
                return SimpleNamespace(value=x)

            def apply(self, layer, x):
                if isinstance(x, torch.Tensor):
                    x = self.prepare_input(x)
                return torch.nn.functional.linear(x.value, layer.weight)

        randn = torch.randn
        with patch.object(
            torch,
            "randn",
            side_effect=lambda *args, **kwargs: randn(
                *args, **{**kwargs, "device": "cpu"}
            ),
        ):
            rt = SimpleNamespace(torch=torch, dense=Dense())
            layers = [
                probe.make_layers(rt, {"weight": (n, 64)}, torch.bfloat16)
                for n in (128, 8)
            ]
            cases, checks = probe.gdn_cases(
                rt, {"hidden_size": 64}, layers, 4, torch.bfloat16
            )
        self.assertEqual(counts["weight"], 2)
        self.assertEqual(checks["shared_vs_separate"][0]["relative_l2"], 0)
        for name, expected in (
            ("pair_w4_shared", 1),
            ("pair_w4_separate", 2),
            ("pair_bf16", 0),
            ("qkvz_w4_matmul_only", 0),
            ("qkvz_w4_total", 1),
            ("pair_w4_qkvz_bf16_ba", 1),
        ):
            with self.subTest(case=name):
                counts["activation"] = 0
                cases[name]()
                self.assertEqual(counts["activation"], expected)
                self.assertEqual(counts["weight"], 2)

    def test_moe_cases_keep_group_counts_and_quant_only_activations(self):
        import torch

        probe = load_probe()
        counts = {"weights": 0, "activation": 0}

        def quant(x):
            counts["activation"] += 1
            return SimpleNamespace(value=x)

        class Method:
            def __init__(self, quantized):
                self.quantized = quantized

            def process_weights_after_loading(self, layer, prefix):
                if self.quantized:
                    counts["weights"] += 1

            def apply(self, layer, x, groups, scale, dtype, prefix, group_type):
                self_check.assertEqual(group_type, 1)
                self_check.assertEqual(groups.sum().item(), x.shape[0])
                chunks = x.split(groups.tolist())
                return torch.cat(
                    [
                        torch.nn.functional.linear(chunk, weight)
                        for chunk, weight in zip(
                            chunks, getattr(layer, prefix + "_weight")
                        )
                    ]
                )

            def apply_fused_gmm1_swiglu(self, layer, x, groups, scale, group_type):
                if scale is None:
                    quant(x)
                value = self.apply(layer, x, groups, scale, x.dtype, "w13", group_type)
                a, b = value.chunk(2, dim=-1)
                return torch.nn.functional.silu(a) * b, torch.ones(x.shape[0], 1, 2)

        class Router:
            def __init__(self, quant_mode):
                self.mode = quant_mode

            def _init_routing(self, x, ids, e, k):
                flat = ids.flatten().long()
                order = flat.argsort(stable=True)
                groups = torch.bincount(flat, minlength=e)
                scale = None if self.mode == -1 else torch.ones(flat.numel(), 1, 2)
                return x.repeat_interleave(k, dim=0)[order], order, groups, scale

        def finalize(x, weights, order, ids):
            output = torch.empty_like(x)
            output[order] = x
            return (output.reshape(*ids.shape, -1) * weights.unsqueeze(-1)).sum(1)

        self_check = self
        randn, full, tensor_to = torch.randn, torch.full, torch.Tensor.to

        def to_cpu(tensor, *args, **kwargs):
            if kwargs.get("device") == "npu":
                kwargs["device"] = "cpu"
            return tensor_to(tensor, *args, **kwargs)

        rt = SimpleNamespace(
            torch=torch,
            dense=SimpleNamespace(prepare_input=quant),
            bf=Method(False),
            w4=Method(True),
            routing=Router,
            finalize=SimpleNamespace(_finalize_routing=finalize),
        )
        plan = {"hidden_size": 64, "num_experts": 4, "top_k": 2}
        with (
            patch.object(
                torch,
                "randn",
                side_effect=lambda *a, **kw: randn(*a, **{**kw, "device": "cpu"}),
            ),
            patch.object(
                torch,
                "full",
                side_effect=lambda *a, **kw: full(*a, **{**kw, "device": "cpu"}),
            ),
            patch.object(torch.Tensor, "to", to_cpu),
            patch.object(
                torch.ops.npu,
                "npu_swiglu",
                create=True,
                side_effect=lambda x: (
                    torch.nn.functional.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1]
                ),
            ),
        ):
            layers = probe.make_layers(
                rt,
                {"w13_weight": (4, 128, 64), "w2_weight": (4, 64, 64)},
                torch.bfloat16,
            )
            for routing in ("uniform", "skewed"):
                cases, checks = probe.moe_cases(
                    rt, plan, layers, 4, torch.bfloat16, routing
                )
                self.assertEqual(sum(checks["expert_token_counts"]), 8)
                self.assertEqual(checks["fused_vs_unfused_route"]["relative_l2"], 0)
                for name, expected in (
                    ("moe_rank_bf16", 0),
                    ("moe_rank_w4_unfused_route", 1),
                    ("moe_rank_w4_fused_route", 0),
                    ("gmm1_w4_fused", 0),
                    ("gmm2_w4", 0),
                ):
                    counts["activation"] = 0
                    output = cases[name]()
                    self.assertEqual(
                        output[0].shape[-1]
                        if isinstance(output, tuple)
                        else output.shape[-1],
                        64,
                    )
                    self.assertEqual(counts["activation"], expected)
                    self.assertEqual(counts["weights"], 2)

    def test_invalid_tp_and_k_alignment_are_rejected(self):
        probe = load_probe()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            for field, value, tp in (
                ("hidden_size", 2049, 2),
                ("num_experts_per_tok", 300, 2),
                ("linear_num_key_heads", 15, 2),
                ("hidden_size", True, 2),
                ("moe_intermediate_size", 512, 3),
                ("hidden_act", "gelu", 2),
            ):
                with self.subTest(field=field, value=value):
                    path.write_text(json.dumps({**CONFIG, field: value}))
                    with self.assertRaises(ValueError):
                        probe.make_plan(directory, tp)

    def test_runtime_failure_preserves_partial_report_without_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(CONFIG))
            (root / "torch.py").write_text(
                'raise RuntimeError("injected runtime failure")'
            )
            output = root / "results"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--model-path",
                    str(root),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(root)},
            )
            self.assertEqual(result.returncode, 2)
            records = [
                json.loads(line)
                for line in (output / "results.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["kind"] for r in records], ["plan", "error"])
            self.assertIn(
                "injected runtime failure", (output / "error.log").read_text()
            )

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(CONFIG))
            output = root / "results"
            output.mkdir()
            (output / "sentinel").write_text("keep")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--model-path",
                    str(root),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual((output / "sentinel").read_text(), "keep")
            self.assertEqual(len(list(output.iterdir())), 1)

    def test_graph_capture_failure_does_not_silently_fall_back(self):
        probe = load_probe()
        npu = SimpleNamespace(
            Stream=lambda: (_ for _ in ()).throw(RuntimeError("capture unsupported"))
        )
        with self.assertRaisesRegex(RuntimeError, "capture unsupported"):
            probe.measure_cases(
                {"case": lambda: None},
                npu,
                warmup=1,
                iterations=1,
                repeats=1,
                mode="graph",
            )

    def test_dry_run_reads_nested_config_without_npu_or_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"text_config": CONFIG}))
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--model-path",
                    str(root),
                    "--tp",
                    "2",
                    "--dry-run",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["hidden_size"], 2048)
            self.assertEqual(plan["gdn_qkvz_n"], 6144)
            self.assertEqual(plan["gdn_ba_n"], 32)
            self.assertEqual(plan["moe_local_intermediate"], 256)
            self.assertEqual(plan["num_experts"], 256)
            self.assertEqual(plan["top_k"], 8)
            self.assertEqual(list(root.iterdir()), [root / "config.json"])


if __name__ == "__main__":
    unittest.main()
