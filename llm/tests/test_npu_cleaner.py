"""Isolated shell/PTY regression tests; no real NPU, model launch, or kill calls."""

import fcntl
import os
from pathlib import Path
import pty
import select
import subprocess
import tempfile
import termios
import time
import unittest


LLM_DIR = Path(__file__).resolve().parents[1]
CLEANER = LLM_DIR / "npu-cleaner.sh"
LAUNCHERS = sorted(LLM_DIR.glob("qwen*.sh")) + [
    LLM_DIR / "a5_fia_mixed_split_serve.sh"
]
OWN_PID = "900000001"
OTHER_PID = "900000002"
OTHER_CARD_PID = "900000003"
ROWS = f"| 0 | {OWN_PID} | python |\n| 0 | {OTHER_PID} | python |\n| 1 | {OTHER_CARD_PID} | python |\n"


class NpuCleanerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.kill_log = self.root / "kill.log"
        self.launch_log = self.root / "launch.log"
        self.query_log = self.root / "query.log"
        self.online = self.root / "online"
        self.offline = self.root / "offline"
        self.online.mkdir()
        self.offline.mkdir()
        (self.offline / "quant_model_description.json").write_text("{}")
        self.bash_env = self.root / "bash-env.sh"
        self.bash_env.write_text(
            'kill() { printf "%s\\n" "$*" >> "$TEST_KILL_LOG"; '
            'return "${TEST_KILL_STATUS:-0}"; }\n'
            'sleep() { :; }\nexport -f kill sleep\n'
        )
        self.mock(
            "npu-smi",
            'printf "query\\n" >> "$TEST_QUERY_LOG"\n'
            'printf "%s" "$TEST_SMI_ROWS"\nexit "${TEST_SMI_STATUS:-0}"\n',
        )
        self.mock("id", 'printf "%s\\n" "${TEST_UID:-1000}"\n')
        self.mock(
            "ps",
            'if [[ "${4:-}" != uid= ]]; then\n'
            '    printf "    PID USER     COMMAND\\n%s alice python own\\n%s bob python other\\n" '
            f'"{OWN_PID}" "{OTHER_PID}"\n'
            '    exit 0\nfi\n'
            'if [[ "$2" == "${TEST_MISSING_PID:-}" ]]; then exit 1; fi\n'
            'if [[ "$2" == "${TEST_CHANGED_PID:-}" ]]; then\n'
            '    if [[ -f "$TEST_STATE_DIR/$2" ]]; then echo 2000; exit 0; fi\n'
            '    touch "$TEST_STATE_DIR/$2"\nfi\n'
            'case "$2" in\n'
            f'    {OWN_PID}) echo "${{TEST_FIRST_UID:-1000}}" ;;\n'
            f'    {OTHER_PID}|{OTHER_CARD_PID}) echo 2000 ;;\n'
            '    *) exit 1 ;;\nesac\n',
        )
        self.mock(
            "sglang",
            'printf "visible:%s\\nargs:%s\\n" "${ASCEND_RT_VISIBLE_DEVICES-<unset>}" '
            '"$*" >> "$TEST_LAUNCH_LOG"\n',
        )
        self.env = os.environ.copy()
        for name in ("SUDO_UID", "MTP", "EXTRA_ARGS"):
            self.env.pop(name, None)
        self.env.update(
            PATH=f"{self.bin_dir}:{self.env['PATH']}",
            BASH_ENV=str(self.bash_env),
            TEST_KILL_LOG=str(self.kill_log),
            TEST_LAUNCH_LOG=str(self.launch_log),
            TEST_QUERY_LOG=str(self.query_log),
            TEST_STATE_DIR=str(self.root),
            TEST_SMI_ROWS=ROWS,
            TEST_UID="1000",
            MODEL_PATH=str(self.online),
            BENCH_ROOT=str(self.root / "bench"),
            ASCEND_RT_VISIBLE_DEVICES="stale-mask",
        )

    def mock(self, name, body):
        path = self.bin_dir / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def clear_logs(self):
        for path in (self.kill_log, self.launch_log, self.query_log):
            path.unlink(missing_ok=True)

    @staticmethod
    def contents(path):
        return path.read_text() if path.exists() else ""

    def command(self, script, args):
        if script.name == "a5_fia_mixed_split_serve.sh":
            args = ["on", *args]
        return ["bash", str(script), *args]

    def environment(self, script, overrides):
        env = self.env | (overrides or {})
        if script.name == "qwen3.5_dense_offline_w4a8.sh":
            env["MODEL_PATH"] = str(self.offline)
        return env

    def run_script(self, script=CLEANER, args=("0",), overrides=None):
        self.clear_logs()
        return subprocess.run(
            self.command(script, list(args)),
            env=self.environment(script, overrides),
            cwd=self.root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # No inherited controlling terminal.
            timeout=5,
        )

    def run_interactive(self, key, script=CLEANER, overrides=None):
        self.clear_logs()
        master, slave = pty.openpty()

        def controlling_terminal():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        process = subprocess.Popen(
            self.command(script, ["0"]),
            env=self.environment(script, overrides),
            cwd=self.root,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            preexec_fn=controlling_terminal,
        )
        os.close(slave)
        output = b""
        sent = False
        deadline = time.monotonic() + 5
        try:
            while time.monotonic() < deadline:
                if select.select([master], [], [], 0.05)[0]:
                    try:
                        chunk = os.read(master, 8192)
                    except OSError:
                        break  # Linux PTY reports EIO after the child closes.
                    if not chunk:
                        break
                    output += chunk
                if not sent and "Esc 取消：".encode() in output:
                    self.assertEqual(self.contents(self.kill_log), "")
                    self.assertEqual(self.contents(self.launch_log), "")
                    os.write(master, key)
                    sent = True
            self.assertTrue(sent, output.decode(errors="replace"))
            code = process.wait(timeout=1)
            return code, output.decode(errors="replace")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)

    def test_no_selected_processes_or_pid_zero(self):
        result = self.run_script(overrides={"TEST_SMI_ROWS": "| 1 | 900000003 |\n| 0 | 0 |\n| NPU | Process id |\n"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("暂无运行进程", result.stdout)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_own_processes_deduplicated_without_confirmation(self):
        row = f"| 0 | {OWN_PID} | python |\n"
        result = self.run_script(overrides={"TEST_SMI_ROWS": row * 2})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Enter", result.stdout)
        self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID}\n")

    def test_foreign_processes_without_terminal_fail_closed(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("没有可用的交互终端", result.stdout)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_enter_confirms_all_selected_processes(self):
        code, output = self.run_interactive(b"\n")
        self.assertEqual(code, 0, output)
        self.assertIn("bob python other", output)
        self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID} {OTHER_PID}\n")

    def test_escape_cancels_without_killing_own_or_foreign_processes(self):
        code, output = self.run_interactive(b"\x1b")
        self.assertEqual(code, 130, output)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_other_keys_do_not_authorize_cleanup(self):
        code, output = self.run_interactive(b"y\x1b")
        self.assertEqual(code, 130, output)
        self.assertIn("请按 Enter", output)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_force_short_and_long_flags_in_any_position(self):
        for args in (("-f", "0"), ("0", "--force")):
            with self.subTest(args=args):
                result = self.run_script(args=args)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("跳过确认", result.stdout)
                self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID} {OTHER_PID}\n")

    def test_all_cards(self):
        result = self.run_script(args=("all", "-f"))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID} {OTHER_PID} {OTHER_CARD_PID}\n")

    def test_sudo_uses_original_user_uid(self):
        result = self.run_script(overrides={"TEST_UID": "0", "SUDO_UID": "1000", "TEST_SMI_ROWS": f"| 0 | {OWN_PID} |\n"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Enter", result.stdout)

    def test_exited_process_is_skipped(self):
        result = self.run_script(overrides={"TEST_SMI_ROWS": f"| 0 | {OWN_PID} |\n", "TEST_MISSING_PID": OWN_PID})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_owner_changes_before_kill_abort(self):
        result = self.run_script(overrides={"TEST_SMI_ROWS": f"| 0 | {OWN_PID} |\n", "TEST_CHANGED_PID": OWN_PID})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("所属用户已变化", result.stdout)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_invalid_owner_aborts(self):
        result = self.run_script(overrides={"TEST_FIRST_UID": "unknown"})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_query_and_kill_failures_are_not_success(self):
        for overrides in ({"TEST_SMI_STATUS": "1"}, {"TEST_KILL_STATUS": "1"}):
            with self.subTest(overrides=overrides):
                result = self.run_script(args=("-f", "0"), overrides=overrides)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertNotIn("清理完成！", result.stdout)

    def test_invalid_arguments_never_query_or_kill(self):
        for args in ((), ("-f",), ("-1",), ("0", "oops")):
            with self.subTest(args=args):
                result = self.run_script(args=args)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertEqual(self.contents(self.query_log), "")
                self.assertEqual(self.contents(self.kill_log), "")

    def test_every_launcher_force_flags_do_not_pollute_device_mask(self):
        for script in LAUNCHERS:
            for args in (("-f", "0", "1"), ("0", "--force"), ("-f", "all")):
                with self.subTest(script=script.name, args=args):
                    result = self.run_script(script, args)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    mask = "<unset>" if "all" in args else "0,1" if "1" in args else "0"
                    launch = self.contents(self.launch_log)
                    self.assertIn(f"visible:{mask}\n", launch)
                    self.assertNotIn("--force", launch)
                    self.assertNotIn("-f ", launch)

    def test_every_launcher_no_devices_means_no_cleanup(self):
        for script in LAUNCHERS:
            for args in ((), ("-f",)):
                with self.subTest(script=script.name, args=args):
                    result = self.run_script(script, args)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn("visible:<unset>\n", self.contents(self.launch_log))
                    self.assertEqual(self.contents(self.query_log), "")
                    self.assertEqual(self.contents(self.kill_log), "")

    def test_every_launcher_foreign_noninteractive_and_kill_failure_abort(self):
        for script in LAUNCHERS:
            for overrides in ({}, {"TEST_KILL_STATUS": "1"}):
                with self.subTest(script=script.name, overrides=overrides):
                    args = ("-f", "0") if overrides else ("0",)
                    result = self.run_script(script, args, overrides)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertEqual(self.contents(self.launch_log), "")
                    if not overrides:
                        self.assertEqual(self.contents(self.kill_log), "")

    def test_every_launcher_escape_aborts(self):
        for script in LAUNCHERS:
            with self.subTest(script=script.name):
                code, output = self.run_interactive(b"\x1b", script)
                self.assertEqual(code, 130, output)
                self.assertEqual(self.contents(self.kill_log), "")
                self.assertEqual(self.contents(self.launch_log), "")

    def test_every_launcher_enter_continues(self):
        for script in LAUNCHERS:
            with self.subTest(script=script.name):
                code, output = self.run_interactive(b"\n", script)
                self.assertEqual(code, 0, output)
                self.assertIn("visible:0\n", self.contents(self.launch_log))

    def test_every_launcher_own_processes_continue_without_prompt(self):
        for script in LAUNCHERS:
            with self.subTest(script=script.name):
                result = self.run_script(script, overrides={"TEST_SMI_ROWS": f"| 0 | {OWN_PID} |\n"})
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("Enter", result.stdout)
                self.assertIn("visible:0\n", self.contents(self.launch_log))

    def test_shell_syntax(self):
        for script in [CLEANER, LLM_DIR / "common/npu-launch.sh", *LAUNCHERS]:
            with self.subTest(script=script.name):
                result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
