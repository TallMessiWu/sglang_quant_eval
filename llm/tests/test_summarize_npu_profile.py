"""CPU fixtures for native NPU exports, not hardware/performance validation."""

import csv
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "llm/diagnostics/summarize_npu_profile.py"
SPEC = importlib.util.spec_from_file_location("summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def export(directory, rows, events=None, steps=None):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "kernel_details.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if events is not None:
        (directory / "trace_view.json").write_text(json.dumps(events))
    if steps:
        with (directory / "step_trace_time.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(steps[0]))
            writer.writeheader()
            writer.writerows(steps)


def kernel(device="6", name="GroupedMatmul", duration="100"):
    return {
        "Device_id": device,
        "Name": "private_kernel_instance",
        "Type": name,
        "Start Time(us)": "0",
        "Duration(us)": duration,
        "Input Shapes": "32,2048",
    }


class TestNpuSummary(unittest.TestCase):
    def test_preserves_startup_and_resolved_scheduler_settings(self):
        self.assertEqual(
            SUMMARY.server_settings(
                {
                    "max_mamba_cache_size": None,
                    "internal_states": [
                        {"max_mamba_cache_size": 873},
                        {"max_mamba_cache_size": 1582},
                    ],
                }
            ),
            {"max_mamba_cache_size": [None, 873, 1582]},
        )

    def test_all_shapes_keeps_every_operator_without_trace_payload(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            export(root, [kernel(name="IndexFill"), kernel(name="GroupedMatmul")])
            result = SUMMARY.collect(root, operators=["*"])
            ops = result["captures"][0]["devices"]["6"]["ops"]
            self.assertEqual(set(ops), {"IndexFill", "GroupedMatmul"})
            self.assertTrue(all("shapes" in op for op in ops.values()))
            self.assertIn("GroupedMatmul:", SUMMARY.render(result))
            result, text = SUMMARY.write_summary(root, all_shapes=True)
            self.assertIn(
                "shapes", result["captures"][0]["devices"]["6"]["ops"]["GroupedMatmul"]
            )
            self.assertNotIn(
                "    shape=", text
            )  # All shapes go to JSON, not a truncated text report.

    def test_gdn_and_hccl_launch_are_not_misclassified(self):
        self.assertEqual(
            SUMMARY.category("_fused_sigmoid_gating_delta_rule_update_decode_kernel"),
            "gdn/conv",
        )
        self.assertEqual(SUMMARY.category("HcclLaunchAicpuKernel"), "comm_launch")
        self.assertEqual(SUMMARY.category("hcom_allReduce_AicpuKernel"), "comm")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            export(
                root,
                [kernel(name="HcclLaunchAicpuKernel")],
                steps=[
                    {"Device_id": "6", "Computing": "100", "Communication": "0"},
                ],
            )
            text = SUMMARY.render(SUMMARY.collect(root))
            self.assertIn("communication duration is not established", text)

    def test_selected_operator_shapes_keep_counts_and_duration_units(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            rows = [kernel(name="IndexFill", duration=v) for v in ("100", "300", "50")]
            rows[2]["Input Shapes"] = "30,1024,128,128;1"
            export(
                root / "one", rows + [kernel(name="MoeInitRoutingV3", duration="1000")]
            )
            result = SUMMARY.collect(root, operators=["IndexFill"])
            text = SUMMARY.render(result, "C01")
            self.assertIn("IndexFill: 3 0.450", text)
            self.assertIn(
                "count=2 sum_ms=0.400 avg_us=200.00 min_us=100.00 max_us=300.00", text
            )
            self.assertIn("shape=30,1024,128,128;1", text)
            self.assertNotIn("MoeInitRoutingV3:", text)
            self.assertNotIn(
                "shapes",
                result["captures"][0]["devices"]["6"]["ops"]["MoeInitRoutingV3"],
            )
            self.assertLessEqual(len(text), SUMMARY.MAX_TEXT)

    def test_native_exports_stay_separate_and_do_not_double_count_trace(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            events = [
                {"ph": "X", "name": "step[DECODE bs=32]", "dur": 2000},
                # Nested CPU and duplicate device trace events are not kernels
                # in this CSV-based report, and are never added to device time.
                {"ph": "X", "name": "aten::linear", "dur": 1900},
                {"ph": "X", "cat": "kernel", "name": "GroupedMatmul", "dur": 100},
            ]
            export(
                root / "traces/a/ASCEND_PROFILER_OUTPUT",
                [
                    kernel(),
                    kernel(duration="200"),
                    kernel("7", "DynamicMxQuant", "50"),
                ],
                events,
                [
                    {
                        "Device_id": "6",
                        "Step": "",
                        "Computing": "250",
                        "Communication": "70",
                        "Communication(Not Overlapped)": "20",
                        "Overlapped": "50",
                        "Free": "10",
                        "Stage": "280",
                    }
                ],
            )
            export(
                root / "traces/b/ASCEND_PROFILER_OUTPUT",
                [kernel(duration="400")],
                [
                    {"ph": "X", "name": "step[EXTEND bs=2 toks=1024]"},
                ],
            )
            result, text = SUMMARY.write_summary(root)
            first, second = result["captures"]
            self.assertEqual(first["phase"], "decode")
            self.assertEqual(second["phase"], "prefill")
            self.assertEqual(
                first["devices"]["6"]["ops"]["GroupedMatmul"], {"count": 2, "us": 300}
            )
            self.assertEqual(first["devices"]["7"]["rows"], 1)
            self.assertEqual(second["devices"]["6"]["ops"]["GroupedMatmul"]["us"], 400)
            self.assertIn(
                "compute=0.250 comm=0.070 comm_uncovered=0.020 overlap=0.050", text
            )
            self.assertIn("GroupedMatmul: 2 0.300 100.0% 150.00", text)
            self.assertIn("kernels=2 sum=0.300", text)
            self.assertIn("extend_tokens=1024..1024", text)
            self.assertEqual((root / "summary.txt").read_text(), text)

    def test_streaming_trace_formats_gzip_and_chunk_boundaries(self):
        events = [
            {"ph": "X", "name": "ignored", "args": {"stack": "隐私" * 50000}},
            {"ph": "X", "name": "step[DECODE bs=8 g_sk=400 forward_id=42]"},
            {"ph": "X", "name": "step[DECODE bs=32]"},
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            for compressed, wrapped in ((False, False), (False, True), (True, True)):
                with self.subTest(compressed=compressed, wrapped=wrapped):
                    data = (
                        {
                            "displayTimeUnit": "ms",
                            "traceEvents": events,
                            "metadata": {"n": 1},
                        }
                        if wrapped
                        else events
                    )
                    path = root / (
                        "trace_view.json.gz" if compressed else "trace_view.json"
                    )
                    if compressed:
                        (root / "trace_view.json").unlink()
                        with gzip.open(path, "wt") as stream:
                            json.dump(data, stream)
                    else:
                        path.write_text(json.dumps(data))
                    steps, evidence = SUMMARY.read_steps(root)
                    self.assertEqual(evidence, "step_labels")
                    self.assertEqual({s["bs"] for s in steps}, {8, 32})
                    self.assertEqual(sum(s["count"] for s in steps), 2)

    def test_unknown_phase_is_not_inferred_from_folder_or_operator_names(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            row = kernel(name="FusedInferAttentionScore")
            del row["Device_id"]
            export(
                root / "TP0-DECODE", [row], [{"ph": "X", "name": "decode_attention"}]
            )
            capture = SUMMARY.collect(root)["captures"][0]
            self.assertEqual(capture["phase"], "unknown")
            self.assertIn("unknown", capture["devices"])
            # Invalid/truncated traces do not invent phase or lose valid CSV data.
            (root / "TP0-DECODE/trace_view.json").write_text('[{"ph": "X"}')
            capture = SUMMARY.collect(root)["captures"][0]
            self.assertEqual(capture["phase_evidence"], "trace_unreadable")
            self.assertEqual(capture["devices"]["unknown"]["rows"], 1)

    def test_bounded_copyable_output_and_metadata_allowlist(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "label": "w4a8",
                        "reported_server_revision": "a" * 40,
                        "command": "secret command",
                        "server": {
                            "api_key": "secret",
                            "model_path": "/private/company/model",
                            "tp_size": 2,
                            "internal_states": [{"max_prefill_tokens": 4096}],
                        },
                    }
                )
            )
            for index in range(12):
                export(
                    root / f"traces/{index:02d}",
                    [
                        kernel(name=f"operator_{n:02d}_" + "long" * 40)
                        for n in range(20)
                    ],
                    [
                        {
                            "ph": "X",
                            "name": "step[DECODE bs=32]",
                            "args": {"stack": "secret"},
                        }
                    ],
                )
            result, text = SUMMARY.write_summary(root)
            self.assertLessEqual(len(text), SUMMARY.MAX_TEXT)
            self.assertEqual(
                result["settings"], {"tp_size": 2, "max_prefill_tokens": 4096}
            )
            for private in ("secret", "/private", "private_kernel_instance"):
                self.assertNotIn(private, text)
            self.assertIn("omitted=", text)
            self.assertNotIn("omitted=0", text)
            selected = SUMMARY.render(result, "C12")
            self.assertIn("C12 device=6", selected)
            self.assertNotIn("C01 device", selected)
            self.assertIn("remaining_types=14", selected)

    def test_unusable_export_returns_failure_with_small_diagnostic(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("No kernel_details.csv", result.stdout)
            self.assertTrue((root / "summary.txt").is_file())
            for bad in ("NaN", "-1", "unknown"):
                export(root / "bad", [kernel(duration=bad)])
                result = SUMMARY.collect(root)
                self.assertEqual(result["captures"], [])
                self.assertEqual(result["errors"], ["C01:kernel_csv:ValueError"])

    def test_broken_cann_csv_does_not_leave_partial_communication_totals(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            export(
                root / "one",
                [kernel()],
                steps=[
                    {"Device_id": "6", "Computing": "100", "Communication": "20"},
                    {"Device_id": "6", "Computing": "100", "Communication": "nan"},
                ],
            )
            result = SUMMARY.collect(root)
            self.assertNotIn("cann_us", result["captures"][0]["devices"]["6"])
            self.assertIn("communication coverage unknown", SUMMARY.render(result))
            self.assertIn("step_csv:ValueError", SUMMARY.render(result))


if __name__ == "__main__":
    unittest.main()
