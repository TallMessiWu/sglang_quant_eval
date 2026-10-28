import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "diagnostics/npu_mamba_clear_probe.py"
spec = importlib.util.spec_from_file_location("mamba_clear_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROBE)


class TestMambaClearProbe(unittest.TestCase):
    def test_clear_preserves_every_unselected_element_and_all_layers(self):
        for shape in ((3, 7, 2, 3, 4), (3, 7, 3, 11)):
            for slots in ([], [6], [6, 0, 3], [5, 2, 5], list(reversed(range(7)))):
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    # Nonuniform values expose wrong strides/layers and accidental
                    # copying between unselected slots; exact CPU comparison only.
                    original = (
                        torch.arange(torch.tensor(shape).prod()).reshape(shape) % 97 + 1
                    ).to(dtype)
                    expected = original.clone()
                    for slot in slots:
                        expected[:, slot] = 0
                    for name in PROBE.METHODS:
                        with self.subTest(
                            shape=shape, slots=slots, dtype=dtype, case=name
                        ):
                            pool = original.clone()
                            indices = torch.tensor(slots, dtype=torch.int32)
                            PROBE.make_cases(torch, pool, indices)[name]()
                            self.assertTrue(torch.equal(pool, expected))

    def test_reused_case_reads_changed_indices(self):
        indices = torch.tensor([0, 3], dtype=torch.int64)
        for name in PROBE.METHODS:
            indices.copy_(torch.tensor([0, 3]))
            pool = torch.ones(2, 7, 3, 5, dtype=torch.bfloat16)
            fn = PROBE.make_cases(torch, pool, indices)[name]
            fn()
            PROBE.check_result(torch, pool, [0, 3], chunk_rows=3)
            pool.fill_(1)
            indices.copy_(torch.tensor([6, 2]))
            fn()
            PROBE.check_result(torch, pool, [6, 2], chunk_rows=3)

    def test_full_checker_detects_unselected_corruption_and_missed_clear(self):
        pool = torch.ones(3, 7, 2, 3, dtype=torch.bfloat16)
        pool[:, 2] = 0
        PROBE.check_result(torch, pool, [2], chunk_rows=4)
        pool[2, 6, 1, 2] = 0
        with self.assertRaisesRegex(RuntimeError, "unselected"):
            PROBE.check_result(torch, pool, [2], chunk_rows=4)
        pool[2, 6, 1, 2] = 1
        pool[0, 2, 0, 1] = 1
        with self.assertRaisesRegex(RuntimeError, "zeroed"):
            PROBE.check_result(torch, pool, [2], chunk_rows=4)

    def test_noncontiguous_pool_is_not_silently_copied(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            PROBE.make_cases(
                torch, torch.ones(3, 7, 8).transpose(0, 1), torch.tensor([0])
            )

    def test_plan_limits_and_no_npu_dry_run(self):
        args = PROBE.parser().parse_args([])
        plan = PROBE.make_plan(args)
        self.assertEqual(len(plan), 4)
        temporal = next(
            p for p in plan if p["state"] == "temporal" and p["capacity"] == 1583
        )
        self.assertAlmostEqual(temporal["pool_gib"], 23.1884765625)
        self.assertGreater(temporal["minimum_free_gib"], 25)
        args.max_pool_gib = 1
        with self.assertRaisesRegex(ValueError, "exceeds"):
            PROBE.make_plan(args)
        args.max_pool_gib = 24
        args.counts = [1600]
        with self.assertRaisesRegex(ValueError, "counts"):
            PROBE.make_plan(args)
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)), 4)

    def test_summary_uses_medians_and_distinguishes_wall_from_event(self):
        args = PROBE.parser().parse_args([])
        rows = [
            {
                **item,
                "count": count,
                "timings": {
                    name: {"stream_ms": [1, 100, 3], "wall_ms": [4, 101, 5]}
                    for name in PROBE.METHODS
                },
            }
            for item in PROBE.make_plan(args)
            for count in args.counts
        ]
        text = PROBE.render(
            {
                "torch": "cpu-test",
                "torch_npu": "mock",
                "device_name": "mock",
                "source": {"git_head": "test"},
            },
            rows,
        )
        self.assertIn("3.0000/5.0000", text)
        self.assertIn("NOT model accuracy", text)
        self.assertIn("NOT kernel sums", text)
        self.assertLess(len(text), 3000)


if __name__ == "__main__":
    unittest.main()
