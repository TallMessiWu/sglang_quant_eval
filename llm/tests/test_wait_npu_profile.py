"""Status-only waiting for automatic analysis; no NPU runtime or process control."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "diffusion/wait_npu_profile.py"
spec = importlib.util.spec_from_file_location("wait_npu_profile", SCRIPT)
waiter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(waiter)


class WaitNPUProfileTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(dir=Path.cwd())))
        self.capture = self.root / "rank0_ascend_pt"
        self.capture.mkdir()
        self.status = self.capture / "sglang_analysis.json"
        self.trace = self.capture / "ASCEND_PROFILER_OUTPUT/trace_view.json"

    def write_trace(self):
        self.trace.parent.mkdir(exist_ok=True)
        self.trace.write_text("{}")

    def test_startup_without_status_and_running_without_trace_are_not_failures(self):
        states = []

        def advance(_):
            states.append(len(states))
            if len(states) == 1:
                self.status.write_text(json.dumps({"state": "running"}))
                self.write_trace()  # A partial trace must not bypass running status.
            else:
                self.status.write_text(json.dumps({"state": "succeeded"}))

        with patch.object(waiter.time, "sleep", side_effect=advance):
            waiter.wait_for_traces(self.root, 1)
        self.assertEqual(len(states), 2)

    def test_failed_status_is_not_masked_by_existing_trace(self):
        self.write_trace()
        self.status.write_text(
            json.dumps({"state": "failed", "error": "native failure"})
        )
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            waiter.wait_for_traces(self.root, 1)

    def test_success_requires_nonempty_trace(self):
        self.status.write_text(json.dumps({"state": "succeeded"}))
        with self.assertRaisesRegex(RuntimeError, "trace_view.json"):
            waiter.wait_for_traces(self.root, 1)
        self.trace.parent.mkdir()
        self.trace.touch()
        with self.assertRaisesRegex(RuntimeError, "trace_view.json"):
            waiter.wait_for_traces(self.root, 1)

    def test_native_trace_does_not_require_a_status_file(self):
        self.write_trace()
        waiter.wait_for_traces(self.root, 1)

    def test_interrupted_wait_returns_without_touching_parser_status(self):
        self.status.write_text(json.dumps({"state": "running"}))
        with (
            patch.object(waiter.sys, "argv", [str(SCRIPT), str(self.root)]),
            patch.object(waiter.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            self.assertEqual(waiter.main(), 130)
        self.assertEqual(json.loads(self.status.read_text())["state"], "running")

    def test_wrong_capture_count_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "期望 2 份"):
            waiter.wait_for_traces(self.root, 2)


if __name__ == "__main__":
    unittest.main()
