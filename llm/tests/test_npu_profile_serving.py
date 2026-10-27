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

    def test_local_capture_and_failure_handling(self):
        state = {"quantization": "mxfp_w4a8", "stops": 0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "quantization": state["quantization"],
                            "version": "test",
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
                (package / "bench_serving.py").write_text("""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
mode = os.environ['FAKE_PROFILE_MODE']
if mode != 'no_ack':
    print('Profiler started', flush=True)
if mode == 'fail':
    raise SystemExit(13)
if mode != 'no_trace':
    trace = Path(args[args.index('--profile-output-dir') + 1]) / 'TP0' / 'trace.json'
    trace.parent.mkdir(parents=True)
    trace.write_text('{}')
Path('env.json').write_text(json.dumps({k: os.environ[k] for k in ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME')}))
""")
                for mode in ("ok", "no_ack", "no_trace", "fail", "mismatch"):
                    with self.subTest(mode=mode):
                        output = root / mode
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
                            ],
                            check=False,
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=20,
                        )
                        if mode == "ok":
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertTrue((output / "traces.json").exists())
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
                        self.assertNotIn(
                            "secret", (output / "metadata.json").read_text()
                        )
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
