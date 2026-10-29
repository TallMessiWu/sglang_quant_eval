"""Profile wrapper contracts; no real serving process or NPU required."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "llm/diagnostics/npu_profile_serving.py"


class TestProfileServing(unittest.TestCase):
    def test_filter_does_not_export_credentials_or_launch_command(self):
        spec = importlib.util.spec_from_file_location("profile_serving", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        info = {
            "api_key": "secret",
            "launch_command": "secret",
            "dtype": "bfloat16",
            "exec": {"moe": {"moe_runner_backend": "ascend"}, "token": "secret"},
            "schedule": {"max_prefill_tokens": 8192, "enable_mixed_chunk": False},
            "graph": {"disable_cuda_graph_padding": False},
        }
        filtered = module.filtered_info(info)
        self.assertNotIn("secret", json.dumps(filtered))
        self.assertEqual(filtered["exec"]["moe"]["moe_runner_backend"], "ascend")
        self.assertEqual(filtered["schedule"], info["schedule"])
        self.assertEqual(filtered["graph"], info["graph"])

    def test_filter_preserves_resolved_scheduler_pools_not_just_startup(self):
        spec = importlib.util.spec_from_file_location("profile_serving", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        info = {
            "max_mamba_cache_size": None,
            "internal_states": [
                {
                    "max_mamba_cache_size": 1582,
                    "memory_usage": {"weight": 9.5, "token_capacity": 100000},
                    "env_vars": {
                        "SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT": "0",
                        "API_KEY": "secret",
                    },
                    "launch_command": "secret",
                },
                {"max_mamba_cache_size": 1580, "api_key": "secret"},
            ],
        }
        filtered = module.filtered_info(info)
        self.assertEqual(filtered["internal_states"][0]["max_mamba_cache_size"], 1582)
        self.assertEqual(filtered["internal_states"][1]["max_mamba_cache_size"], 1580)
        self.assertEqual(filtered["internal_states"][0]["memory_usage"]["weight"], 9.5)
        self.assertNotIn("secret", json.dumps(filtered))
        # An absent label is not proof of BF16; check nested resolved settings.
        with self.assertRaises(ValueError):
            module.validate_server(
                {"internal_states": [{"quantization": "mxfp_w4a8"}]}, "bf16"
            )
        with self.assertRaises(ValueError):
            module.validate_server({"version": "test"}, "bf16")
        module.validate_server({"quantization": None}, "bf16")

    def test_diagnose_plan_is_bounded_and_separates_profiled_results(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--label",
                    "w4a8",
                    "--model-path",
                    str(root),
                    "--output-dir",
                    str(root / "run"),
                    "--diagnose",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=root,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(
                [case["name"] for case in plan],
                ["prefill", "decode", "sharegpt", "profile"],
            )
            for case in plan:
                command = case["command"]
                self.assertIn("--flush-cache", command)
                self.assertEqual("--profile" in command, case["name"] == "profile")
                self.assertNotIn(
                    "--trace-stalls", command
                )  # Not in the formal PR branch.
                if case["name"] in ("prefill", "decode"):
                    self.assertIn("--tokenize-prompt", command)
                    self.assertEqual(
                        command[command.index("--random-range-ratio") + 1], "1"
                    )
            for case, count in zip(plan, (64, 32, 500, 128)):
                command = case["command"]
                self.assertEqual(
                    int(command[command.index("--num-prompts") + 1]), count
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_local_capture_and_failure_handling(self):
        state = {"quantization": "mxfp_w4a8", "stops": 0, "gets": 0, "restart": False}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                state["gets"] += 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "quantization": state["quantization"],
                            "version": "test",
                            "startup_time": state["gets"] if state["restart"] else 100,
                            "internal_states": [{"max_mamba_cache_size": 1582}],
                            "api_key": "secret",
                        }
                    ).encode()
                )

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                state["stops"] += 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"stopped")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                package = root / "sglang"
                package.mkdir()
                (package / "__init__.py").write_text("")
                (root / "config.json").write_text("{}")
                (root / "dataset.json").write_text("[]")
                (package / "bench_serving.py").write_text("""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
mode = os.environ['FAKE_PROFILE_MODE']
if '--profile' not in args:
    count = int(args[args.index('--num-prompts') + 1])
    record = {'completed': count, 'generated_texts': ['secret request'],
              'server_info': {'api_key': 'secret', 'internal_states': [{'max_mamba_cache_size': 1582}]}}
    if mode == 'partial':
        record['completed'] -= 1
    Path(args[args.index('--output-file') + 1]).write_text(json.dumps(record) + '\\n')
    raise SystemExit(0)
if mode != 'no_ack':
    print('Profiler started', flush=True)
if mode == 'fail':
    raise SystemExit(13)
if mode != 'no_trace':
    trace = Path(args[args.index('--profile-output-dir') + 1]) / 'TP0' / 'trace_view.json'
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps([{'ph': 'X', 'name': 'step[DECODE bs=32]'}]))
    if mode != 'no_kernel_export':
        (trace.parent / 'kernel_details.csv').write_text('Device_id,Name,Type,Duration(us)\\n0,gmm,GroupedMatmul,100\\n')
Path('env.json').write_text(json.dumps({k: os.environ[k] for k in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME')}))
""")
                for mode in (
                    "ok",
                    "no_ack",
                    "no_trace",
                    "fail",
                    "mismatch",
                    "no_kernel_export",
                    "diagnose",
                    "partial",
                    "restart",
                ):
                    with self.subTest(mode=mode):
                        output = root / mode
                        diagnostic = mode in ("diagnose", "partial", "restart")
                        state["restart"] = mode == "restart"
                        state["quantization"] = (
                            None if mode == "mismatch" else "mxfp_w4a8"
                        )
                        env = {
                            **os.environ,
                            "PYTHONPATH": str(root),
                            "FAKE_PROFILE_MODE": mode,
                        }
                        result = subprocess.run(
                            [
                                sys.executable,
                                str(SCRIPT),
                                "--label",
                                "w4a8",
                                "--model-path",
                                str(root),
                                "--server-url",
                                f"http://127.0.0.1:{server.server_port}",
                                "--output-dir",
                                str(output),
                                *(
                                    [
                                        "--diagnose",
                                        "--dataset-path",
                                        str(root / "dataset.json"),
                                    ]
                                    if diagnostic
                                    else []
                                ),
                            ],
                            check=False,
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=20,
                        )
                        if mode in ("ok", "diagnose"):
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertTrue((output / "traces.json").exists())
                            self.assertTrue((output / "summary.txt").is_file())
                            self.assertIn("GroupedMatmul", result.stdout)
                            self.assertIn("Copy only", result.stdout)
                            settings = json.loads((output / "env.json").read_text())
                            self.assertTrue(
                                all(
                                    Path(v).is_relative_to(output)
                                    for v in settings.values()
                                )
                            )
                        else:
                            self.assertEqual(result.returncode, 2, result.stdout)
                            self.assertTrue((output / "error.txt").exists())
                            if mode == "no_kernel_export":
                                self.assertIn("No kernel_details.csv", result.stdout)
                        self.assertNotIn(
                            "secret", (output / "metadata.json").read_text()
                        )
                        if diagnostic:
                            report = json.loads((output / "report.json").read_text())
                            self.assertNotIn("secret", json.dumps(report))
                            self.assertTrue((output / "report.txt").exists())
                            if mode == "diagnose":
                                self.assertEqual(
                                    list(report["unprofiled"]),
                                    ["prefill", "decode", "sharegpt"],
                                )
                                self.assertEqual(
                                    report["metadata"]["server"]["internal_states"][0][
                                        "max_mamba_cache_size"
                                    ],
                                    1582,
                                )
                                self.assertEqual(len(report["profile"]["captures"]), 1)
                            else:
                                self.assertIn(
                                    "INCOMPLETE", (output / "report.txt").read_text()
                                )
                            if mode == "partial":
                                self.assertFalse((output / "benchmark.log").exists())
                self.assertEqual(
                    state["stops"], 2
                )  # Failed owned captures only, never no_ack/mismatch.
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_dry_run_reuses_serving_benchmark_and_keeps_outputs_local(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--label",
                    "w4a8",
                    "--model-path",
                    str(root),
                    "--output-dir",
                    str(root / "profile"),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=root,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            command = json.loads(result.stdout)
            self.assertEqual(
                command[:3], [sys.executable, "-m", "sglang.bench_serving"]
            )
            self.assertIn("--profile-by-stage", command)
            self.assertIn("--flush-cache", command)
            self.assertEqual(command[command.index("--max-concurrency") + 1], "32")
            self.assertEqual(command[command.index("--num-prompts") + 1], "500")
            self.assertEqual(
                command[command.index("--profile-output-dir") + 1],
                str(root / "profile/traces"),
            )
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
