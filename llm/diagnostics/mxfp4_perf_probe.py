#!/usr/bin/env python3
"""Single-rank, synthetic-input W4A8 microbenchmarks; NOT serving performance.

Read only config.json to derive Qwen GDN/MoE TP shapes. Use the installed SGLang
NPU methods, without launching a model or changing source, branches or PYTHONPATH.
Weight conversion is outside timing. Run on an idle NPU in the serving environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def make_plan(model_path, tp):
    path = Path(model_path) / "config.json"
    raw = path.read_bytes()
    config = json.loads(raw)
    config = config.get("text_config", config)

    def integer(name):
        value = config[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid positive integer {name}: {value!r}")
        return value

    h = integer("hidden_size")
    intermediate = integer("moe_intermediate_size")
    experts, top_k = integer("num_experts"), integer("num_experts_per_tok")
    kh, vh = integer("linear_num_key_heads"), integer("linear_num_value_heads")
    kd, vd = integer("linear_key_head_dim"), integer("linear_value_head_dim")
    if tp <= 0 or any(size % tp for size in (kh, vh, intermediate)):
        raise ValueError(
            "TP must divide key heads, value heads and MoE intermediate size"
        )
    if top_k > experts or h % 64 or (intermediate // tp) % 64:
        raise ValueError("Require top_k <= experts and full K axes divisible by 64")
    if config.get("hidden_act", "silu") not in ("silu", "swish"):
        raise ValueError(
            "This probe models ordinary SwiGLU, not custom/clamped activation"
        )
    if any(config.get(key) is not None for key in ("swiglu_limit", "swiglu_alpha")):
        raise ValueError("Clamped/custom SwiGLU is outside this probe")
    return {
        "model_path": str(Path(model_path).resolve()),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "model_type": config.get("model_type"),
        "tp": tp,
        "hidden_size": h,
        "gdn_qkvz_n": (2 * kh * kd + 2 * vh * vd) // tp,
        "gdn_ba_n": 2 * vh // tp,
        "moe_local_intermediate": intermediate // tp,
        "num_experts": experts,
        "top_k": top_k,
        "bf16_expert_weight_gib": experts * 3 * h * (intermediate // tp) * 2 / 2**30,
        "scope": "One TP rank; synthetic weights/activations; no communication or GDN core",
    }


def capture(fn, npu, warmup):
    stream = npu.Stream()
    stream.wait_stream(npu.current_stream())
    with npu.stream(stream):
        for _ in range(warmup):
            fn()
    npu.synchronize()
    graph = npu.NPUGraph()
    with npu.graph(graph, stream=stream):
        output = fn()
    npu.current_stream().wait_stream(stream)
    npu.synchronize()

    def replay():
        graph.replay()
        return output  # Retain captured output storage for the graph lifetime.

    return replay


def measure_cases(cases, npu, *, warmup, iterations, repeats, mode):
    """Stream elapsed includes launch gaps; wall elapsed also includes draining.

    Neither is summed kernel time. Real overlap/occupancy requires a profiler.
    Alternate variant order between repeats; exclude preparation, warmup and capture.
    """
    if mode == "graph":
        cases = {name: capture(fn, npu, warmup) for name, fn in cases.items()}
    for fn in cases.values():
        for _ in range(warmup):
            fn()
    npu.synchronize()
    results = {name: {"stream_ms": [], "wall_ms": []} for name in cases}
    names = list(cases)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            start, end = npu.Event(enable_timing=True), npu.Event(enable_timing=True)
            npu.synchronize()
            before = time.perf_counter()
            start.record()
            for _ in range(iterations):
                cases[name]()
            end.record()
            npu.synchronize()
            results[name]["wall_ms"].append(
                (time.perf_counter() - before) * 1000 / iterations
            )
            elapsed = start.elapsed_time(end) / iterations
            if elapsed <= 0:
                raise RuntimeError(
                    f"Nonpositive event time for {name}; increase iterations"
                )
            results[name]["stream_ms"].append(elapsed)
    return results


def source_identity(path):
    path = Path(path).resolve()
    result = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for key, command in (
        ("git_head", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--porcelain", "--untracked-files=no"]),
    ):
        process = subprocess.run(
            ["git", "-C", str(path.parent), *command],
            capture_output=True,
            text=True,
            check=False,
        )
        result[key] = process.stdout.strip() if process.returncode == 0 else None
    return result


def load_runtime(device):
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError(
            "NPU unavailable; --dry-run needs neither torch_npu nor SGLang"
        )
    torch.npu.set_device(device)
    # Initialize the registry first, avoiding the NPU helper cold-import cycle.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting
    from sglang.srt.hardware_backend.npu.moe.init_routing import NPUMoEInitRouting_v2
    from sglang.srt.hardware_backend.npu.quantization import (
        linear_method_npu,
        moe_methods,
    )

    dense = linear_method_npu.NPUMXFP4W4A8LinearMethod()
    if not hasattr(dense, "prepare_input"):
        raise RuntimeError(
            "Installed SGLang lacks shared-input support; select the updated checkout yourself"
        )
    return SimpleNamespace(
        torch=torch,
        torch_npu=torch_npu,
        dense=dense,
        w4=moe_methods.NPUW4A8MXFP4MoEMethod(),
        bf=moe_methods.NPUUnquantMoEMethod(),
        routing=NPUMoEInitRouting_v2,
        finalize=NPUFinalizeRouting(),
        sources={
            "linear": source_identity(linear_method_npu.__file__),
            "moe": source_identity(moe_methods.__file__),
        },
    )


def make_layers(rt, shapes, dtype):
    """Convert once using production weight policy/layout, never inside a case."""
    torch = rt.torch
    bf, w4 = torch.nn.Module(), torch.nn.Module()
    for name, shape in shapes.items():
        weight = torch.randn(shape, dtype=dtype, device="npu") / shape[-1] ** 0.5
        bf.register_parameter(
            name, torch.nn.Parameter(weight.clone(), requires_grad=False)
        )
        w4.register_parameter(name, torch.nn.Parameter(weight, requires_grad=False))
    if list(shapes) == ["weight"]:
        rt.dense._quantize_weights(w4)
    else:
        for prefix in ("w13", "w2"):
            rt.bf.process_weights_after_loading(bf, prefix)
            rt.w4.process_weights_after_loading(w4, prefix)
    return bf, w4


def difference(torch, actual, reference):
    actual, reference = actual.float(), reference.float()
    if (
        actual.shape != reference.shape
        or not torch.isfinite(actual).all().item()
        or not torch.isfinite(reference).all().item()
    ):
        raise RuntimeError("Invalid/non-finite diagnostic output")
    delta = actual - reference
    return {
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-12)).item(),
        "max_abs": delta.abs().max().item(),
    }


def gdn_cases(rt, plan, layers, tokens, dtype):
    torch = rt.torch
    (bf_q, w4_q), (bf_ba, w4_ba) = layers
    x = torch.randn(tokens, plan["hidden_size"], dtype=dtype, device="npu")
    prepared = rt.dense.prepare_input(x)
    bf_q_fn = lambda: torch.nn.functional.linear(x, bf_q.weight)
    bf_ba_fn = lambda: torch.nn.functional.linear(x, bf_ba.weight)
    separate = lambda: (rt.dense.apply(w4_q, x), rt.dense.apply(w4_ba, x))

    def shared():
        activation = rt.dense.prepare_input(x)
        return rt.dense.apply(w4_q, activation), rt.dense.apply(w4_ba, activation)

    shared_out, separate_out = shared(), separate()
    checks = {
        "shared_vs_separate": [
            difference(torch, a, b) for a, b in zip(shared_out, separate_out)
        ],
        "w4_vs_bf16": [
            difference(torch, shared_out[0], bf_q_fn()),
            difference(torch, shared_out[1], bf_ba_fn()),
        ],
    }
    return {
        "activation_quant": lambda: rt.dense.prepare_input(x),
        "qkvz_bf16": bf_q_fn,
        "qkvz_w4_matmul_only": lambda: rt.dense.apply(w4_q, prepared),
        "qkvz_w4_total": lambda: rt.dense.apply(w4_q, x),
        "ba_bf16": bf_ba_fn,
        "ba_w4_matmul_only": lambda: rt.dense.apply(w4_ba, prepared),
        "ba_w4_total": lambda: rt.dense.apply(w4_ba, x),
        "pair_bf16": lambda: (bf_q_fn(), bf_ba_fn()),
        "pair_w4_separate": separate,
        "pair_w4_shared": shared,
        "pair_w4_qkvz_bf16_ba": lambda: (rt.dense.apply(w4_q, x), bf_ba_fn()),
    }, checks


def moe_cases(rt, plan, layers, tokens, dtype, distribution):
    torch = rt.torch
    bf, w4 = layers
    e, k, h = plan["num_experts"], plan["top_k"], plan["hidden_size"]
    x = torch.randn(tokens, h, dtype=dtype, device="npu")
    active = e if distribution == "uniform" else max(k, e // 8)
    ids = (
        torch.rand(tokens, active, device="cpu")
        .topk(k, dim=-1)
        .indices.to(device="npu", dtype=torch.int32)
    )
    weights = torch.full((tokens, k), 1 / k, dtype=dtype, device="npu")
    routers = [rt.routing(quant_mode=mode) for mode in (-1, 3)]

    def route(index):
        return routers[index]._init_routing(x, ids, e, k)

    def bf_first(routed):
        value = rt.bf.apply(bf, routed[0], routed[2], None, dtype, "w13", 1)
        return torch.ops.npu.npu_swiglu(value)

    def w4_first(routed):
        return rt.w4.apply_fused_gmm1_swiglu(w4, routed[0], routed[2], routed[3], 1)

    def bf_second(value, counts):
        return rt.bf.apply(bf, value, counts, None, dtype, "w2", 1)

    def w4_second(value, counts):
        return rt.w4.apply(w4, value[0], counts, value[1], dtype, "w2", 1)

    def chain(quantized, fused):
        routed = route(int(fused))
        output = (
            w4_second(w4_first(routed), routed[2])
            if quantized
            else bf_second(bf_first(routed), routed[2])
        )
        return rt.finalize._finalize_routing(output, weights, routed[1], ids)

    b_route, q_route = route(0), route(1)
    b_first, q_first = bf_first(b_route), w4_first(q_route)
    counts = b_route[2].cpu().tolist()
    q_output = chain(True, True)
    checks = {
        "fused_vs_unfused_route": difference(torch, q_output, chain(True, False)),
        "w4_vs_bf16": difference(torch, q_output, chain(False, False)),
        "expert_token_counts": counts,
        "mean_rows_per_expert": tokens * k / e,
    }
    return {
        "route_bf16": lambda: route(0),
        "route_mxfp8": lambda: route(1),
        "quant_routed_activations": lambda: rt.dense.prepare_input(b_route[0]),
        "gmm1_bf16_swiglu": lambda: bf_first(b_route),
        "gmm1_w4_fused": lambda: w4_first(q_route),
        "gmm2_bf16": lambda: bf_second(b_first, b_route[2]),
        "gmm2_w4": lambda: w4_second(q_first, q_route[2]),
        "moe_rank_bf16": lambda: chain(False, False),
        "moe_rank_w4_unfused_route": lambda: chain(True, False),
        "moe_rank_w4_fused_route": lambda: chain(True, True),
    }, checks


def run_probe(args, plan, emit):
    rt = load_runtime(args.device)
    torch = rt.torch
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    dtype = torch.bfloat16
    emit(
        {
            "kind": "runtime",
            "torch": torch.__version__,
            "torch_npu": rt.torch_npu.__version__,
            "device": args.device,
            "device_name": torch.npu.get_device_name(args.device),
            "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
            "sources": rt.sources,
            "cann_home": os.getenv("ASCEND_HOME_PATH"),
            "dtype": str(dtype),
        }
    )
    with torch.inference_mode():
        run_sections(rt, args, plan, emit)


def run_sections(rt, args, plan, emit):
    torch = rt.torch
    dtype = torch.bfloat16
    for section in args.sections:
        h = plan["hidden_size"]
        if section == "gdn":
            layers = [
                make_layers(rt, {"weight": (plan[key], h)}, dtype)
                for key in ("gdn_qkvz_n", "gdn_ba_n")
            ]
        else:
            if plan["bf16_expert_weight_gib"] > args.max_weight_gib:
                raise ValueError(
                    "Expert BF16 weights exceed --max-weight-gib; do not silently shrink experts"
                )
            e, n = plan["num_experts"], plan["moe_local_intermediate"]
            layers = make_layers(
                rt, {"w13_weight": (e, 2 * n, h), "w2_weight": (e, h, n)}, dtype
            )
        for tokens in args.tokens:
            for distribution in args.routing if section == "moe" else ["none"]:
                cases, checks = (
                    gdn_cases(rt, plan, layers, tokens, dtype)
                    if section == "gdn"
                    else moe_cases(rt, plan, layers, tokens, dtype, distribution)
                )
                emit(
                    {
                        "kind": "sanity",
                        "section": section,
                        "tokens": tokens,
                        "routing": distribution,
                        "checks": checks,
                        "note": "Synthetic checks, not model accuracy or bitwise-equivalence acceptance",
                    }
                )
                results = measure_cases(
                    cases,
                    torch.npu,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    mode=args.mode,
                )
                for name, values in results.items():
                    median = statistics.median(values["stream_ms"])
                    emit(
                        {
                            "kind": "timing",
                            "section": section,
                            "tokens": tokens,
                            "routing": distribution,
                            "mode": args.mode,
                            "case": name,
                            "median_stream_ms": median,
                            "median_wall_ms": statistics.median(values["wall_ms"]),
                            **values,
                        }
                    )
                    print(
                        f"{section:3} M={tokens:4} {distribution:7} {name:28} {median:10.4f} ms",
                        flush=True,
                    )
                del cases
        del layers
        torch.npu.empty_cache()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", required=True, type=Path)
    result.add_argument(
        "--tp", type=positive, default=2, help="Shape partition only, not NPU count"
    )
    result.add_argument(
        "--dry-run", action="store_true", help="Print shapes without torch/NPU imports"
    )
    result.add_argument(
        "--device",
        type=int,
        default=0,
        help="Logical NPU index after visibility masking",
    )
    result.add_argument(
        "--tokens", nargs="+", type=positive, default=[1, 8, 32, 128, 512]
    )
    result.add_argument(
        "--sections", nargs="+", choices=["gdn", "moe"], default=["gdn", "moe"]
    )
    result.add_argument(
        "--routing",
        nargs="+",
        choices=["uniform", "skewed"],
        default=["uniform", "skewed"],
    )
    result.add_argument("--mode", choices=["eager", "graph"], default="eager")
    result.add_argument("--warmup", type=positive, default=10)
    result.add_argument("--iterations", type=positive, default=50)
    result.add_argument("--repeats", type=positive, default=3)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--max-weight-gib",
        type=float,
        default=4,
        help="BF16 expert weight cap; actual peak is higher",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        help="Must not exist; default ./artifacts/mxfp4-perf/<unique-run>",
    )
    return result


def main():
    args = parser().parse_args()
    try:
        plan = make_plan(args.model_path, args.tp)
        if args.device < 0 or not args.max_weight_gib > 0:
            raise ValueError("device must be nonnegative and max-weight-gib positive")
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        output = (
            args.output_dir
            or Path("artifacts/mxfp4-perf")
            / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        ).resolve()
        output.mkdir(parents=True, exist_ok=False)
    except (ValueError, KeyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for variable, suffix in {
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "cache",
        "HF_HOME": "cache/hf",
        "HF_HUB_CACHE": "cache/hf/hub",
        "TRITON_CACHE_DIR": "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "PYTHONPYCACHEPREFIX": "cache/pycache",
        "ASCEND_PROCESS_LOG_PATH": "ascend-logs",
    }.items():
        path = output / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    print(
        f"Results: {output / 'results.jsonl'}\nTiming is synthetic single-rank, not end-to-end throughput.",
        flush=True,
    )
    with (output / "results.jsonl").open("x") as report:

        def emit(record):
            report.write(json.dumps(record, default=str) + "\n")
            report.flush()

        emit({"kind": "plan", "plan": plan, "options": vars(args)})
        try:
            # Runtime imports and all FP4 weight conversion occur outside measured cases.
            run_probe(args, plan, emit)
        except Exception as error:  # noqa: BLE001 -- retain partial measurements and fail, never mask an NPU error
            emit({"kind": "error", "error": str(error)})
            (output / "error.log").write_text(traceback.format_exc())
            print(
                f"error: {error}; partial results preserved in {output}",
                file=sys.stderr,
            )
            return 2
        emit(
            {
                "kind": "complete",
                "status": "measurements_collected_not_performance_acceptance",
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
