"""Allocator-window and spawn contracts; no NPU, model load or process cleanup."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

LLM = Path(__file__).resolve().parents[1]
SCRIPT = LLM / "diagnostics/npu_load_memory.py"
LAUNCHER = LLM / "record_w4a4_load_memory.sh"
GIB = 1 << 30


class FakeNPU:
    def __init__(self):
        self.allocated = 10 * GIB
        self.reserved = 12 * GIB
        self.peak = 100 * GIB
        self.reserved_peak = 120 * GIB
        self.events = []
        self.broken = False

    def current_device(self):
        return 0

    def get_device_name(self, device):
        return "fake-npu"

    def synchronize(self):
        self.events.append("sync")

    def reset_peak_memory_stats(self, device):
        self.events.append("reset")
        self.peak, self.reserved_peak = self.allocated, self.reserved

    def memory_allocated(self, device):
        return self.allocated

    def memory_reserved(self, device):
        return self.reserved

    def max_memory_allocated(self, device):
        if self.broken:
            raise RuntimeError("stats unavailable")
        return self.peak

    def max_memory_reserved(self, device):
        return self.reserved_peak


class TestLoadMemory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location("load_memory", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.npu = FakeNPU()
        metadata = {
            "label": "test",
            "pid": os.getpid(),
            "logical_device": 0,
            "ranks": {"tp_rank": 0},
            "source_revision": None,
        }
        for patcher in (
            patch.dict(os.environ, {self.module.OUTPUT_ENV: str(self.root)}),
            patch.object(self.module, "_npu_backend", return_value=self.npu),
            patch.object(
                self.module, "_metadata", side_effect=lambda *a: dict(metadata)
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def report(self):
        files = list(self.root.glob("load-memory-*.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())

    def test_window_resets_before_construction_and_records_peak_not_delta(self):
        def load(runner):
            self.assertEqual(self.npu.events, ["sync", "reset"])
            pending = self.report()
            self.assertFalse(pending["complete"])
            self.assertEqual(pending["status"], "loading")
            self.assertEqual(self.npu.peak, 10 * GIB)
            self.npu.events.append("load")
            self.npu.peak = 64 * GIB
            self.npu.reserved_peak = 80 * GIB
            self.npu.allocated = 16 * GIB
            self.npu.reserved = 20 * GIB
            return runner

        wrapped = self.module._wrap_load(load)
        self.assertIs(self.module._wrap_load(wrapped), wrapped)
        runner = SimpleNamespace(startup_weight_load=None)
        self.assertIs(wrapped(runner), runner)
        self.assertEqual(self.npu.events, ["sync", "reset", "load", "sync"])
        report = self.report()
        self.assertTrue(report["complete"])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(
            report["summary_gib"],
            {
                "peak_allocated": 64,
                "peak_reserved": 80,
                "end_allocated": 16,
                "end_reserved": 20,
                "peak_allocated_above_begin": 54,
            },
        )
        # Work after load_model is outside this report, even if it fills KV memory.
        self.npu.peak = 110 * GIB
        self.assertEqual(self.report()["summary_gib"]["peak_allocated"], 64)

    def test_oom_is_preserved_and_report_is_incomplete(self):
        failure = RuntimeError("NPU out of memory")

        def load(runner):
            self.npu.peak = 30 * GIB
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            self.module._wrap_load(load)(SimpleNamespace())
        self.assertIs(caught.exception, failure)
        report = self.report()
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["complete"])
        self.assertEqual(report["summary_gib"]["peak_allocated"], 30)

    def test_failed_stats_do_not_mask_loading_exception(self):
        failure = RuntimeError("original loading failure")

        def load(runner):
            self.npu.broken = True
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            self.module._wrap_load(load)(SimpleNamespace())
        self.assertIs(caught.exception, failure)
        report = self.report()
        self.assertIn("original loading failure", report["error"])
        self.assertIn("stats unavailable", report["measurement_error"])
        self.assertFalse(report["complete"])

    def test_deferred_loading_is_not_reported_as_completed_weights(self):
        runner = SimpleNamespace(startup_weight_load=object())
        self.module._wrap_load(lambda self: None)(runner)
        self.assertFalse(self.report()["complete"])
        self.assertEqual(self.report()["status"], "deferred_weights")

    def test_multiple_loads_have_separate_files(self):
        wrapped = self.module._wrap_load(lambda self: None)
        wrapped(SimpleNamespace(is_draft_worker=False))
        wrapped(SimpleNamespace(is_draft_worker=True))
        self.assertEqual(len(list(self.root.glob("load-memory-*.json"))), 2)

    def test_summary_flags_unfinished_records(self):
        record = {"complete": False, "status": "loading", "ranks": {"tp_rank": 0}}
        (self.root / "load-memory-killed.json").write_text(json.dumps(record))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = self.module.summarize([str(self.root)])
        self.assertEqual(status, 1)
        self.assertIn("INCOMPLETE:loading", output.getvalue())
        self.assertIn("peak_alloc_GiB", output.getvalue())

    def test_summary_cli_reads_completed_window_without_npu(self):
        self.module._wrap_load(lambda self: None)(SimpleNamespace())
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "summarize", str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("\tok\t", result.stdout)
        self.assertIn("10.000", result.stdout)

    def test_metadata_does_not_claim_outer_repo_revision_or_export_secrets(self):
        # Use a fresh module so this test exercises real metadata collection.
        spec = importlib.util.spec_from_file_location("metadata_memory", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = SimpleNamespace(
            server_args=SimpleNamespace(api_key="secret", model_path="model")
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "torch": SimpleNamespace(__version__="fake-torch"),
                    "torch_npu": SimpleNamespace(__version__="fake-npu"),
                    "sglang.srt.runtime_context": SimpleNamespace(
                        get_parallel=lambda: SimpleNamespace(tp_rank=0, tp_size=1)
                    ),
                },
            ),
            patch.object(
                module.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "git"),
            ),
            patch.object(module.subprocess, "check_output") as revision,
        ):
            metadata = module._metadata(runner, lambda self: None, self.npu, 0)
        revision.assert_not_called()
        self.assertIsNone(metadata["source_revision"])
        self.assertNotIn("secret", json.dumps(metadata))


class TestMemoryLauncher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, relative, source):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
        return path

    def test_help_and_invalid_label_do_not_start_device_selection(self):
        for argument, expected in (("--help", 0), ("6", 2), ("../before", 2)):
            with self.subTest(argument=argument):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), argument],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertFalse((self.root / "artifacts").exists())

    def test_real_spawn_reinstalls_hook_without_forcing_early_imports(self):
        for package in (
            "sglang",
            "sglang/cli",
            "sglang/srt",
            "sglang/srt/model_executor",
        ):
            self.put(f"{package}/__init__.py", "")
        self.put("torch.py", '__version__ = "fake-torch"\n')
        self.put(
            "torch_npu.py",
            """
            __version__ = "fake-npu"
            class Backend:
                allocated = 10
                reserved = 12
                peak = 100
                reserved_peak = 120
                def current_device(self): return 0
                def get_device_name(self, device): return "fake-npu"
                def synchronize(self): pass
                def reset_peak_memory_stats(self, device):
                    self.peak, self.reserved_peak = self.allocated, self.reserved
                def memory_allocated(self, device): return self.allocated
                def memory_reserved(self, device): return self.reserved
                def max_memory_allocated(self, device): return self.peak
                def max_memory_reserved(self, device): return self.reserved_peak
            npu = Backend()
        """,
        )
        self.put("sglang/srt/utils.py", "def is_npu(): return True\n")
        self.put(
            "sglang/srt/runtime_context.py",
            """
            from types import SimpleNamespace
            def get_parallel(): return SimpleNamespace(tp_rank=0, tp_size=1)
        """,
        )
        self.put(
            "sglang/srt/model_executor/model_runner.py",
            """
            assert __loader__.get_source(__name__) is not None
            class ModelRunner:
                def load_model(self):
                    import torch_npu
                    assert torch_npu.npu.peak == 10
                    torch_npu.npu.peak = 40
                    torch_npu.npu.reserved_peak = 50
                    torch_npu.npu.allocated = 20
                    torch_npu.npu.reserved = 22
                    return 42
        """,
        )
        self.put(
            "sglang/cli/main.py",
            """
            import multiprocessing as mp
            import os
            import sys
            def worker():
                from sglang.srt.model_executor.model_runner import ModelRunner
                assert ModelRunner.load_model._npu_load_memory_wrapped
                assert ModelRunner().load_model() == 42
            def main():
                assert "sglang.srt.model_executor.model_runner" not in sys.modules
                print("parent_pid=" + str(os.getpid()), flush=True)
                process = mp.get_context("spawn").Process(target=worker)
                process.start()
                process.join(15)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    raise RuntimeError("worker timeout")
                if process.exitcode != 0:
                    raise RuntimeError("worker failed")
        """,
        )
        output = self.root / "reports"
        env = {
            **os.environ,
            "PYTHONPATH": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NPU_LOAD_MEMORY_DIR": str(output),
            "NPU_LOAD_MEMORY_LABEL": "spawn",
        }
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "serve"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        files = list(output.glob("load-memory-*.json"))
        self.assertEqual(len(files), 1)
        report = json.loads(files[0].read_text())
        self.assertTrue(report["complete"])
        self.assertEqual(report["end"]["peak_allocated_bytes"], 40)
        self.assertEqual(report["end"]["allocated_bytes"], 20)
        parent_pid = int(result.stdout.strip().split("parent_pid=")[-1])
        self.assertNotEqual(report["pid"], parent_pid)
        self.assertEqual(
            report["source_file"],
            str(self.root / "sglang/srt/model_executor/model_runner.py"),
        )

    def test_shell_uses_normal_parameters_and_preserves_failure_exit(self):
        launcher = self.put("llm/record_w4a4_load_memory.sh", LAUNCHER.read_text())
        self.put(
            "llm/common/npu-launch.sh",
            """
            npu_prepare_launch() {
                shift
                export NPU_TP_SIZE=$#
                export ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "$*")
            }
        """,
        )
        self.put("llm/diagnostics/npu_load_memory.py", "")
        python = self.put(
            "bin/python3",
            f"#!{sys.executable}\n"
            + """
import json, os, sys
from pathlib import Path
Path(os.environ["CAPTURE"]).write_text(json.dumps({
    "args": sys.argv[1:], "visible": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
    "output": os.environ["NPU_LOAD_MEMORY_DIR"],
    "label": os.environ["NPU_LOAD_MEMORY_LABEL"],
    "pythonpath": os.environ.get("PYTHONPATH"),
}))
print("fake server log")
sys.exit(int(os.environ.get("FAILURE", "0")))
""",
        )
        python.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{python.parent}:{os.environ['PATH']}",
            "CAPTURE": str(self.root / "capture.json"),
            "PYTHONPATH": "unchanged",
        }
        for code in (0, 17):
            result = subprocess.run(
                ["bash", str(launcher), "before", "6"],
                cwd=self.root,
                env={**env, "FAILURE": str(code)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, code, result.stderr)
            captured = json.loads((self.root / "capture.json").read_text())
            self.assertEqual(captured["visible"], "6")
            self.assertEqual(captured["label"], "before")
            self.assertEqual(captured["pythonpath"], "unchanged")
            args = captured["args"]
            for flag, expected in (
                ("--tp", "1"),
                ("--quantization", "mxfp4"),
                ("--device", "npu"),
            ):
                self.assertEqual(args[args.index(flag) + 1], expected)
            self.assertIn(
                "fake server log", (Path(captured["output"]) / "server.log").read_text()
            )
        self.assertEqual(len(list((self.root / "artifacts/load-memory").iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
