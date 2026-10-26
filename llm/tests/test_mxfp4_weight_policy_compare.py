"""CPU reference/checkpoint/mocked-call tests, not NPU validation.

python -m unittest discover -s llm/tests -p 'test_mxfp4_weight_policy_compare.py' -v
Requires torch and safetensors. Temporary fixtures stay under the current cwd.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
import mxfp4_weight_policy_compare as probe


class PolicyCompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_spread_and_row_ranges(self):
        self.assertEqual(probe.spread(range(9), 3), [0, 4, 8])
        self.assertEqual(probe.spread(range(8), 2), [0, 7])
        for size in range(1, 160):
            for limit in (0, 1, 2, 3, 7, 128, 256):
                ranges = probe.row_ranges(size, limit)
                ids = [i for start, stop in ranges for i in range(start, stop)]
                expected = min(size, limit) if limit else size
                self.assertEqual(len(ids), expected)
                self.assertEqual(len(set(ids)), expected)
                self.assertTrue(all(0 <= i < size for i in ids))
        ranges = probe.row_ranges(4096, 128)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 4096)

    def test_automatic_layer_expert_selection(self):
        names = {
            f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight": "s"
            for layer in range(9)
            for expert in range(8)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        names["model.visual.layers.0.mlp.up_proj.weight"] = "s"
        names["model.language_model.layers.0.mlp.gate.weight"] = "s"
        names["model.language_model.layers.0.self_attn.q_proj.weight"] = "s"
        names["model.language_model.layers.0.mlp.shared_expert.up_proj.weight"] = "s"
        selected = probe.select_names(names, None, 3, 2)
        self.assertEqual(len(selected), 3 * 2 * 3 + 2)
        self.assertEqual({int(probe.LAYER.search(n)[1]) for n in selected}, {0, 4, 8})
        self.assertEqual(
            {
                int(probe.EXPERT.search(n)[1])
                for n in selected
                if probe.EXPERT.search(n)
            },
            {0, 7},
        )
        self.assertFalse(any("visual" in n for n in selected))
        with self.assertRaisesRegex(ValueError, "not found"):
            probe.select_names(names, ["missing"], 3, 2)
        key = next(iter(names))
        self.assertEqual(probe.select_names(names, [key, key], 3, 2), [key])

    def test_real_safetensors_slices_2d_and_3d(self):
        weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(256 * 64)
            .reshape(256, 64)
            .to(torch.float16),
            "model.layers.0.mlp.experts.gate_up_proj": torch.arange(4 * 256 * 64)
            .reshape(4, 256, 64)
            .to(torch.bfloat16),
        }
        save_file(weights, self.root / "model.safetensors")
        mapping = probe.checkpoint_index(self.root)
        self.assertEqual(set(mapping), set(weights))
        self.assertEqual(set(probe.select_names(mapping, None, 3, 2)), set(weights))
        with safe_open(str(self.root / "model.safetensors"), framework="pt") as handle:
            for name, original in weights.items():
                samples = list(probe.read_samples(handle, name, 128, 2))
                self.assertEqual(len(samples), 2 if original.ndim == 3 else 1)
                for sample, meta in samples:
                    ids = [
                        i
                        for start, stop in meta["row_ranges"]
                        for i in range(start, stop)
                    ]
                    expected = (
                        original
                        if original.ndim == 2
                        else original[meta["expert"] : meta["expert"] + 1]
                    )
                    torch.testing.assert_close(
                        sample, expected.index_select(-2, torch.tensor(ids))
                    )
                    self.assertEqual(sample.ndim, original.ndim)
                    self.assertEqual(sample.shape[-2:], (128, 64))
                full = next(probe.read_samples(handle, name, 0, 1))[0]
                self.assertEqual(full.shape[-2], 256)

    def test_sharded_index_and_invalid_input(self):
        mapping = {"model.layers.0.mlp.down_proj.weight": "part-1.safetensors"}
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
        self.assertEqual(probe.checkpoint_index(self.root), mapping)
        (self.root / "quant_model_description.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "original BF16"):
            probe.checkpoint_index(self.root)
        for weight in (
            torch.zeros(8, 64, dtype=torch.uint8),
            torch.zeros(8, 33, dtype=torch.bfloat16),
            torch.zeros(64, dtype=torch.bfloat16),
        ):
            save_file({"w": weight}, self.root / "invalid.safetensors")
            with (
                safe_open(
                    str(self.root / "invalid.safetensors"), framework="pt"
                ) as handle,
                self.assertRaises(ValueError),
            ):
                list(probe.read_samples(handle, "w", 128, 2))

    def test_known_codes_and_weight_metrics(self):
        # Every E2M1 code, negative zero included; different adjacent block scales.
        packed = torch.tensor(
            [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
        ).repeat(2, 4)
        scale = torch.tensor([[126, 128], [127, 129]], dtype=torch.uint8)
        reference = probe.dequantize_weight(packed, scale)
        expected = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        torch.testing.assert_close(reference[0, :16], expected * 0.5)
        torch.testing.assert_close(reference[0, 32:48], expected * 2)
        result = probe.weight_metrics(
            reference.to(torch.bfloat16).unsqueeze(0), packed, scale
        )
        self.assertEqual(result["nmse"], 0)
        self.assertEqual(result["zero_fraction"], 2 / 16)
        self.assertEqual(result["out_of_range_fraction"], 0)
        # +/-6 code occupancy alone is not clipping; source=7 actually exceeds it.
        clipped = probe.weight_metrics(
            torch.full((1, 64), 7.0),
            torch.full((1, 32), 0x77, dtype=torch.uint8),
            torch.full((1, 2), 127, dtype=torch.uint8),
        )
        self.assertEqual(clipped["mse"], 1)
        self.assertEqual(clipped["out_of_range_fraction"], 1)
        self.assertAlmostEqual(clipped["nmse"], 1 / 49)
        zero = probe.weight_metrics(
            torch.zeros(1, 64),
            torch.zeros(1, 32, dtype=torch.uint8),
            torch.full((1, 2), 127, dtype=torch.uint8),
        )
        self.assertIsNone(zero["nmse"])
        with self.assertRaisesRegex(ValueError, "NaN"):
            probe.weight_metrics(reference, packed, torch.full_like(scale, 255))

    def test_code_difference_counts_nibbles_not_bytes(self):
        packed = torch.zeros(1, 32, dtype=torch.uint8)
        scale = torch.full((1, 2), 127, dtype=torch.uint8)
        other_packed, other_scale = packed.clone(), scale.clone()
        other_packed[0, 0] = 0x10
        other_scale[0, 0] += 1
        result = probe.code_differences((packed, scale), (other_packed, other_scale))
        self.assertEqual(result["fp4_code_changed_fraction"], 1 / 64)
        self.assertEqual(result["scale_changed_fraction"], 0.5)

    def test_three_calls_preserve_inputs_and_exact_kwargs(self):
        weight = torch.ones(1, 8, 64, dtype=torch.bfloat16)
        packed = torch.full((1, 8, 32), 0x22, dtype=torch.uint8)  # E2M1 1.0
        scale = torch.full((1, 8, 1, 2), 127, dtype=torch.uint8)
        with (
            mock.patch.object(
                torch, "npu", SimpleNamespace(synchronize=mock.Mock()), create=True
            ),
            mock.patch.object(
                torch.ops.npu,
                "npu_dynamic_mx_quant",
                return_value=(packed, scale),
                create=True,
            ) as op,
        ):
            result = probe.compare_sample(weight, "cpu", 296)
            self.assertEqual(op.call_count, 3)
            for call, policy in zip(op.call_args_list, probe.POLICIES):
                self.assertIs(call.args[0], weight)
                self.assertEqual(
                    call.kwargs, {"dst_type": 296, **probe.POLICIES[policy]}
                )
                self.assertEqual(result["policies"][policy]["mse"], 0)
            self.assertEqual(
                result["differences"]["C_vs_A"]["fp4_code_changed_fraction"], 0
            )
            op.reset_mock()
            op.side_effect = [
                (packed, scale),
                (packed, scale),
                TypeError("unsupported scale_alg"),
            ]
            with self.assertRaisesRegex(TypeError, "scale_alg"):
                probe.compare_sample(weight, "cpu", 296)
            self.assertEqual(op.call_count, 3)  # C must fail, not silently fall back.

    def test_summary_direction_and_energy_weighting(self):
        results = [
            {
                "policies": {
                    p: {"squared_error": e, "energy": 10.0, "elements": 5}
                    for p, e in zip("ABC", errors)
                }
            }
            for errors in ((4.0, 2.0, 1.0), (1.0, 1.0, 3.0))
        ]
        result = probe.summarize(results)
        self.assertEqual(result["policies"]["C"]["pooled_nmse"], 4 / 20)
        self.assertEqual(
            result["comparisons"]["C_vs_A"],
            {"lower_error": 1, "equal_error": 0, "higher_error": 1},
        )
        self.assertEqual(result["comparisons"]["B_vs_A"]["equal_error"], 1)

    def test_list_and_partial_error_report_without_npu(self):
        key = "model.layers.0.mlp.down_proj.weight"
        save_file(
            {key: torch.ones(8, 64, dtype=torch.bfloat16)},
            self.root / "model.safetensors",
        )
        report = self.root / "results.jsonl"
        argv = ["probe", "--model-path", str(self.root), "--output", str(report)]
        with (
            mock.patch.object(sys, "argv", argv + ["--list"]),
            contextlib.redirect_stdout(io.StringIO()) as captured,
        ):
            self.assertEqual(probe.main(), 0)
        self.assertIn(key, captured.getvalue())
        self.assertFalse(report.exists())
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                probe, "run", side_effect=RuntimeError("NPU unavailable")
            ),
            self.assertRaisesRegex(RuntimeError, "NPU unavailable"),
        ):
            probe.main()
        events = [json.loads(line) for line in report.read_text().splitlines()]
        self.assertEqual([e["event"] for e in events], ["config", "error"])
        with mock.patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
            probe.main()
        self.assertEqual(report.read_text().count("\n"), 2)


if __name__ == "__main__":
    unittest.main()
