"""Shell/PTY tests for the tokenizer menu; never contact a model service."""

import fcntl
import os
import pty
import re
import select
import shutil
import subprocess
import tempfile
import termios
import time
import unittest
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]
SCRIPT = LLM_DIR / "bench_serving.sh"


class BenchServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.llm = self.root / "scripts with spaces"
        self.llm.mkdir()
        self.script = self.llm / SCRIPT.name
        shutil.copy2(SCRIPT, self.script)
        self.models = []
        for source in sorted(LLM_DIR.glob("*_bf16.sh")):
            text = source.read_text()
            self.models.append(re.search(r"--model-path\s+(\S+)", text).group(1))
            (self.llm / source.name).write_text('touch "$LAUNCH_SENTINEL"\n' + text)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.argv = self.root / "argv"
        self.cache_env = self.root / "cache-env"
        self.sentinel = self.root / "launched-server"
        python = self.bin / "python3"
        python.write_text(
            "#!/bin/bash\n"
            'printf "%s\\0" "$@" > "$TEST_ARGV"\n'
            'printf "%s\\0" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" > "$TEST_CACHE_ENV"\n'
            'previous=""\n'
            'for arg in "$@"; do\n'
            '    if [[ $previous == --output-file ]]; then printf "{}\\n" > "$arg"; fi\n'
            "    previous=$arg\n"
            "done\n"
            'echo "mock benchmark stdout"\n'
            'echo "mock benchmark stderr" >&2\n'
            'exit "${TEST_EXIT_CODE:-0}"\n'
        )
        python.chmod(0o755)
        self.env = os.environ.copy()
        for name in (
            "VLLM_PORT",
            "SERVER_HOST",
            "DATASET_PATH",
            "MAX_CONCURRENCY",
            "NUM_PROMPTS",
            "FLUSH_CACHE",
            "BENCH_ROOT",
            "NO_COLOR",
            "BASH_ENV",
            "TEST_EXIT_CODE",
        ):
            self.env.pop(name, None)
        self.env.update(
            PATH=f"{self.bin}:{self.env['PATH']}",
            TERM="xterm-256color",
            TEST_ARGV=str(self.argv),
            TEST_CACHE_ENV=str(self.cache_env),
            LAUNCH_SENTINEL=str(self.sentinel),
        )

    def args(self):
        return self.argv.read_bytes().decode().rstrip("\0").split("\0")

    def interactive(self, keys, overrides=None):
        master, slave = pty.openpty()
        original_state = termios.tcgetattr(slave)

        def controlling_terminal():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        process = subprocess.Popen(
            [str(self.script)],
            cwd=self.root,
            env=self.env | (overrides or {}),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            preexec_fn=controlling_terminal,  # noqa: PLW1509 - Single-threaded PTY test.
        )
        output = bytearray()
        sent = False
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                ready, _, _ = select.select([master], [], [], 0.05)
                if ready:
                    output.extend(os.read(master, 65536))
                if not sent and "不会切换服务端模型".encode() in output:
                    os.write(master, keys)
                    sent = True
                if process.poll() is not None and not ready:
                    break
            else:
                self.fail(f"menu timed out: {output.decode(errors='replace')}")
            self.assertTrue(sent, output.decode(errors="replace"))
            code = process.wait(timeout=1)
            self.assertEqual(termios.tcgetattr(slave), original_state)
            self.assertFalse(self.sentinel.exists(), "BF16 launcher was executed")
            text = output.decode(errors="replace")
            self.assertIn("\x1b[?25h\x1b[?1049l", text)
            return code, text
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)
            os.close(slave)

    def noninteractive(self, args=()):
        return subprocess.run(
            [str(self.script), *args],
            cwd=self.root,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            timeout=5,
            check=False,
        )

    def test_every_current_bf16_model_and_default_command(self):
        self.assertTrue(self.models)
        for index, model in enumerate(self.models):
            with self.subTest(model=model):
                code, text = self.interactive(b"\x1b[B" * index + b"\r")
                self.assertEqual(code, 0, text)
                for path in self.models:
                    self.assertIn(f"{Path(path).name}\x1b[0m \x1b[2m({path})", text)
                args = self.args()
                self.assertEqual(args[:2], ["-m", "sglang.bench_serving"])
                options = dict(zip(args[2::2], args[3::2]))
                self.assertEqual(
                    options,
                    {
                        "--backend": "sglang",
                        "--host": "127.0.0.1",
                        "--port": "6969",
                        "--dataset-name": "sharegpt",
                        "--dataset-path": "/mnt/share/dataset/ShareGPT_V3_unfiltered_cleaned_split.json",
                        "--tokenizer": model,
                        "--max-concurrency": "32",
                        "--num-prompts": "500",
                        "--output-file": options["--output-file"],
                    },
                )
                result = Path(options["--output-file"])
                self.assertTrue(
                    result.is_relative_to(self.root / "artifacts/bench-serving")
                )
                self.assertTrue(result.is_file())
                log = result.with_name("bench.log").read_text()
                self.assertIn("mock benchmark stdout", log)
                self.assertIn("mock benchmark stderr", log)
                for directory in (
                    self.cache_env.read_bytes().decode().rstrip("\0").split("\0")
                ):
                    self.assertTrue(Path(directory).is_relative_to(result.parent))

    def test_up_wraps_and_down_returns_to_first_model(self):
        for keys, expected in (
            (b"\x1b[A\r", self.models[-1]),
            (b"\x1b[A\x1b[B\r", self.models[0]),
        ):
            with self.subTest(keys=keys):
                code, text = self.interactive(keys)
                self.assertEqual(code, 0, text)
                args = self.args()
                self.assertEqual(args[args.index("--tokenizer") + 1], expected)

    def test_flush_cache_is_opt_in_without_enabling_profiler(self):
        code, text = self.interactive(b"\r", {"FLUSH_CACHE": "1"})
        self.assertEqual(code, 0, text)
        self.assertEqual(self.args().count("--flush-cache"), 1)
        self.assertNotIn("--profile", self.args())
        self.env["FLUSH_CACHE"] = "bad"
        result = self.noninteractive()
        self.assertEqual(result.returncode, 2)
        self.assertIn("FLUSH_CACHE", result.stdout)

    def test_overrides_and_no_color_preserve_argument_boundaries(self):
        dataset = str(self.root / "data with spaces.json")
        code, text = self.interactive(
            b"\r",
            {
                "VLLM_PORT": "7979",
                "SERVER_HOST": "10.0.0.2",
                "DATASET_PATH": dataset,
                "MAX_CONCURRENCY": "8",
                "NUM_PROMPTS": "30",
                "BENCH_ROOT": str(self.root / "custom results"),
                "NO_COLOR": "1",
            },
        )
        self.assertEqual(code, 0, text)
        self.assertNotIn("\x1b[1m", text)
        self.assertNotIn("\x1b[2m", text)
        options = dict(zip(self.args()[2::2], self.args()[3::2]))
        for key, value in (
            ("--port", "7979"),
            ("--host", "10.0.0.2"),
            ("--dataset-path", dataset),
            ("--max-concurrency", "8"),
            ("--num-prompts", "30"),
        ):
            self.assertEqual(options[key], value)

    def test_cancel_and_interrupt_do_not_start_benchmark(self):
        for keys, status in ((b"q", 0), (b"\x1b", 0), (b"\x03", 130)):
            with self.subTest(keys=keys):
                code, text = self.interactive(keys)
                self.assertEqual(code, status, text)
                self.assertFalse(self.argv.exists())
                self.assertFalse((self.root / "artifacts").exists())

    def test_benchmark_failure_propagates_through_tee(self):
        code, text = self.interactive(b"\r", {"TEST_EXIT_CODE": "7"})
        self.assertEqual(code, 7, text)

    def test_help_and_missing_terminal(self):
        result = self.noninteractive(("--help",))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("↑/↓", result.stdout)
        result = self.noninteractive()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("需要交互终端", result.stdout)
        self.assertFalse(self.argv.exists())

    def test_menu_tracks_new_bf16_script_without_executing_it(self):
        path = "/weights/New-Model"
        (self.llm / "zz_new_bf16.sh").write_text(
            f'touch "$LAUNCH_SENTINEL"\nsglang serve \\\n    --model-path {path} \\\n'
        )
        code, text = self.interactive(b"\x1b[A\r")
        self.assertEqual(code, 0, text)
        self.assertIn("New-Model", text)
        args = self.args()
        self.assertEqual(args[args.index("--tokenizer") + 1], path)

    def test_missing_or_variable_model_path_fails_closed(self):
        for path in ("", "$MODEL_PATH", "/weights/$MODEL_NAME"):
            with self.subTest(path=path):
                (self.llm / "zz_bad_bf16.sh").write_text(f"    --model-path {path}\n")
                result = self.noninteractive()
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("无法读取固定模型路径", result.stdout)
                self.assertFalse(self.argv.exists())


if __name__ == "__main__":
    unittest.main()
