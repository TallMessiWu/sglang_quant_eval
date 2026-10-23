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
LORA = LLM_DIR / "ascend_moe_lora/serve.sh"
LLM_LAUNCHERS = sorted(LLM_DIR.glob("qwen*.sh")) + [
    LLM_DIR / "a5_fia_mixed_split_serve.sh"
]
DIFFUSION_DIR = LLM_DIR.parent / "diffusion"
DIFFUSION_LAUNCHERS = sorted(DIFFUSION_DIR.glob("*.sh"))
LAUNCHERS = LLM_LAUNCHERS + DIFFUSION_LAUNCHERS
HYBRID_LAUNCHERS = [
    script for script in LAUNCHERS
    if script.name.startswith("qwen3.5_") or script.name == "a5_fia_mixed_split_serve.sh"
]
OWN_PID = "900000001"
OTHER_PID = "900000002"
OTHER_CARD_PID = "900000003"
ROWS = f"| 0 | {OWN_PID} | python |\n| 0 | {OTHER_PID} | python |\n| 1 | {OTHER_CARD_PID} | python |\n"


class NpuCleanerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.kill_log = self.root / "kill.log"
        self.launch_log = self.root / "launch.log"
        self.query_log = self.root / "query.log"
        self.device_log = self.root / "devices.log"
        self.pwdx_log = self.root / "pwdx.log"
        self.online = self.root / "online"
        self.offline = self.root / "offline"
        self.online.mkdir()
        self.offline.mkdir()
        (self.offline / "quant_model_description.json").write_text("{}")
        (self.root / "sglang/python/sglang").mkdir(parents=True)
        self.bash_env = self.root / "bash-env.sh"
        self.bash_env.write_text(
            'kill() { printf "%s\\n" "$*" >> "$TEST_KILL_LOG"; '
            'return "${TEST_KILL_STATUS:-0}"; }\n'
            'sleep() { :; }\nexport -f kill sleep\n'
            'if [[ "${TEST_PWDX_MISSING:-0}" == 1 ]]; then\n'
            '    command() {\n'
            '        if [[ "${1:-}" == -v && "${2:-}" == pwdx ]]; then return 1; fi\n'
            '        builtin command "$@"\n'
            '    }\n'
            '    export -f command\n'
            'fi\n'
        )
        self.mock(
            "npu-smi",
            'printf "query\\n" >> "$TEST_QUERY_LOG"\n'
            'printf "%s" "$TEST_SMI_ROWS"\nexit "${TEST_SMI_STATUS:-0}"\n',
        )
        self.identity_log = self.root / "identity.log"
        self.mock(
            "id",
            'printf "id\\n" >> "$TEST_IDENTITY_LOG"\n'
            'printf "%s\\n" "${TEST_UID:-1000}"\n',
        )
        self.mock(
            "ps",
            'if [[ "${4:-}" == lstart= ]]; then\n'
            '    echo "Wed Oct 14 08:30:00 2026"; exit 0\nfi\n'
            'if [[ "${4:-}" != uid= ]]; then\n'
            '    if [[ "${TEST_PS_STATUS:-0}" != 0 ]]; then exit "$TEST_PS_STATUS"; fi\n'
            '    if [[ "$2" == "${TEST_PS_MISSING_PID:-}" ]]; then exit 1; fi\n'
            '    case "$2" in\n'
            f'        {OWN_PID}) cmd="${{TEST_PS_COMMAND:-python own}}" ;;\n'
            f'        {OTHER_PID}) cmd="python other" ;;\n'
            f'        {OTHER_CARD_PID}) cmd="python unselected" ;;\n'
            '        *) exit 1 ;;\n    esac\n'
            '    printf "%s\\n" "$cmd"\n'
            '    exit 0\nfi\n'
            'printf "ps uid\\n" >> "$TEST_IDENTITY_LOG"\n'
            'case "$2" in\n'
            f'    {OWN_PID}) echo "${{TEST_FIRST_UID:-1000}}" ;;\n'
            f'    {OTHER_PID}|{OTHER_CARD_PID}) echo 2000 ;;\n'
            '    *) exit 1 ;;\nesac\n',
        )
        self.mock(
            "pwdx",
            'printf "%s\\n" "$1" >> "$TEST_PWDX_LOG"\n'
            'if [[ "$1" == "${TEST_PWDX_FAILURE_PID:-}" ]]; then\n'
            '    printf "%s: %s\\n" "$1" "${TEST_PWDX_ERROR:-Permission denied}" >&2\n'
            '    exit 1\nfi\n'
            'case "$1" in\n'
            f'    {OWN_PID}) directory=/work/alice/models ;;\n'
            f'    {OTHER_PID}) directory="/work/bob/my project" ;;\n'
            f'    {OTHER_CARD_PID}) directory=/work/carol/models ;;\n'
            '    *) exit 1 ;;\nesac\n'
            'printf "%s: %s\\n" "$1" "$directory"\n',
        )
        self.mock(
            "sglang",
            'printf "ssm_dtype:%s\\n" "${SGLANG_MAMBA_SSM_DTYPE-<unset>}" '
            '>> "$TEST_LAUNCH_LOG"\n'
            'printf "visible:%s\\nargs:%s\\n" "${ASCEND_RT_VISIBLE_DEVICES-<unset>}" '
            '"$*" >> "$TEST_LAUNCH_LOG"\n',
        )
        self.mock(
            "python3",
            'if [[ "$1" == -c ]]; then\n'
            '    printf "%s\\n" "${ASCEND_RT_VISIBLE_DEVICES-<unset>}" >> "$TEST_DEVICE_LOG"\n'
            '    printf "%s\\n" "${TEST_NPU_COUNT:-8}"\n'
            '    exit "${TEST_DEVICE_STATUS:-0}"\nfi\n'
            'if [[ "$1" == -m && "$2" == sglang.launch_server ]]; then\n'
            '    exec sglang "$@"\nfi\nexit 2\n',
        )
        self.mock(
            "python",
            '[[ -f "$1" ]] || exit 2\n'
            'exec sglang "$@"\n',
        )
        self.env = os.environ.copy()
        for name in ("SUDO_UID", "MTP", "EXTRA_ARGS", "SGLANG_MAMBA_SSM_DTYPE"):
            self.env.pop(name, None)
        self.env.update(
            PATH=f"{self.bin_dir}:{self.env['PATH']}",
            BASH_ENV=str(self.bash_env),
            TEST_KILL_LOG=str(self.kill_log),
            TEST_LAUNCH_LOG=str(self.launch_log),
            TEST_QUERY_LOG=str(self.query_log),
            TEST_DEVICE_LOG=str(self.device_log),
            TEST_PWDX_LOG=str(self.pwdx_log),
            TEST_IDENTITY_LOG=str(self.identity_log),
            TEST_SMI_ROWS=ROWS,
            TEST_UID="1000",
            MODEL_PATH=str(self.online),
            BENCH_ROOT=str(self.root / "bench"),
            ASCEND_RT_VISIBLE_DEVICES="stale-mask",
            TP_SIZE="99",
            NPU_TP_SIZE="99",
            SGLANG_DIR=str(self.root / "sglang"),
            ADAPTER_A_PATH=str(self.root / "adapter-a"),
            ADAPTER_B_PATH=str(self.root / "adapter-b"),
        )

    def mock(self, name, body):
        path = self.bin_dir / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def clear_logs(self):
        for path in (self.kill_log, self.launch_log, self.query_log, self.device_log, self.identity_log, self.pwdx_log):
            path.unlink(missing_ok=True)

    @staticmethod
    def contents(path):
        return path.read_text() if path.exists() else ""

    def assert_parallel_size(self, script, launch, size):
        option = "--num-gpus" if script in DIFFUSION_LAUNCHERS else "--tp-size" if script == LORA else "--tp"
        self.assertRegex(launch, rf"{option} {size}(?: |$)")

    def process_row(self, output, pid):
        rows = [line for line in output.splitlines() if line.split() and line.split()[0] == pid]
        self.assertEqual(len(rows), 1, output)
        return rows[0]

    def command(self, script, args):
        if script.name == "a5_fia_mixed_split_serve.sh" and (not args or args[0] not in ("on", "off")):
            args = ["on", *args]
        if script == LORA and (not args or args[0] not in ("baseline", "lora")):
            args = ["baseline", "graph", *args]
        # Execute via the shebang so missing executable bits fail like on servers.
        return [str(script), *args]

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

    def run_interactive(self, key, script=CLEANER, overrides=None, args=("0",), before_key=None):
        self.clear_logs()
        master, slave = pty.openpty()

        def controlling_terminal():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        process = subprocess.Popen(
            self.command(script, list(args)),
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
                    if before_key is not None:
                        before_key(output.decode(errors="replace"))
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

    def test_offline_w4a8_existing_same_uid_pid_requires_confirmation(self):
        result = self.run_script(
            LLM_DIR / "qwen3.5_dense_offline_w4a8.sh",
            overrides={"TEST_SMI_ROWS": f"| 0 | {OWN_PID} | python |\n"},
        )
        self.assertEqual(self.contents(self.kill_log), "", result.stdout)
        self.assertEqual(self.contents(self.launch_log), "", result.stdout)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("没有可用的交互终端", result.stdout)
        self.assertEqual(self.contents(self.identity_log), "")

    def test_existing_processes_deduplicated_after_confirmation(self):
        row = f"| 0 | {OWN_PID} | python |\n"
        code, output = self.run_interactive(b"\n", overrides={"TEST_SMI_ROWS": row * 2})
        self.assertEqual(code, 0, output)
        self.assertIn("Enter 确认", output)
        self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID}\n")
        self.assertEqual(self.contents(self.identity_log), "")

    def test_existing_processes_without_terminal_fail_closed(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("没有可用的交互终端", result.stdout)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_enter_confirms_all_selected_processes(self):
        code, output = self.run_interactive(b"\n")
        self.assertEqual(code, 0, output)
        self.assertRegex(self.process_row(output, OTHER_PID), rf"^{OTHER_PID}\s+2026-10-14 08:30:00\s+python other\s+/work/bob/my project$")
        self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID} {OTHER_PID}\n")

    def test_skip_keeps_processes_and_returns_success(self):
        for key in (b"p", b"P"):
            for args in (("0",), ("all",)):
                with self.subTest(key=key, args=args):
                    code, output = self.run_interactive(
                        key, args=args, overrides={"TEST_KILL_STATUS": "1"}
                    )
                    self.assertEqual(code, 0, output)
                    self.assertIn("P 保留进程并继续", output)
                    self.assertIn("已跳过清理", output)
                    self.assertEqual(self.contents(self.kill_log), "")
                    self.assertNotIn("清理完成！", output)

    def test_escape_cancels_without_killing_any_process(self):
        code, output = self.run_interactive(b"\x1b")
        self.assertEqual(code, 130, output)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_process_table_has_working_directory_as_last_column(self):
        result = self.run_script(args=("-f", "0"))
        self.assertEqual(result.returncode, 0, result.stdout)
        headers = [line for line in result.stdout.splitlines() if line.split() == ["PID", "START_TIME", "COMMAND", "工作目录"]]
        self.assertEqual(len(headers), 1, result.stdout)
        for pid, command, directory in (
            (OWN_PID, "python own", "/work/alice/models"),
            (OTHER_PID, "python other", "/work/bob/my project"),
        ):
            rows = [line for line in result.stdout.splitlines() if line.split() and line.split()[0] == pid]
            self.assertEqual(len(rows), 1, result.stdout)
            self.assertRegex(rows[0], rf"^{pid}\s+2026-10-14 08:30:00\s+{command}\s+{directory}$")
        self.assertNotIn("USER", result.stdout)
        self.assertNotIn("进程当前工作目录", result.stdout)
        self.assertNotIn(f"{OWN_PID}: /work/alice/models", result.stdout)

    def test_table_keeps_long_commands_and_aligns_working_directories(self):
        command = 'python /work/alice/server.py --model-path "/weights/a model with spaces" --port 6969'
        result = self.run_script(args=("-f", "0"), overrides={"TEST_PS_COMMAND": command})
        self.assertEqual(result.returncode, 0, result.stdout)
        own_row = self.process_row(result.stdout, OWN_PID)
        other_row = self.process_row(result.stdout, OTHER_PID)
        self.assertIn(command, own_row)
        self.assertEqual(own_row.index("/work/alice/models"), other_row.index("/work/bob/my project"))

    def test_unavailable_process_metadata_does_not_shift_directory_rows(self):
        result = self.run_script(args=("-f", "0"), overrides={"TEST_PS_MISSING_PID": OWN_PID})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertRegex(self.process_row(result.stdout, OWN_PID), rf"^{OWN_PID}\s+2026-10-14 08:30:00\s+\?\s+/work/alice/models$")
        self.assertRegex(self.process_row(result.stdout, OTHER_PID), rf"^{OTHER_PID}\s+2026-10-14 08:30:00\s+python other\s+/work/bob/my project$")

    def test_pwdx_directories_are_shown_before_confirmation(self):
        def check_display(output):
            self.assertEqual(self.contents(self.pwdx_log), f"{OWN_PID}\n{OTHER_PID}\n")
            for pid, directory in ((OWN_PID, "/work/alice/models"), (OTHER_PID, "/work/bob/my project")):
                row = self.process_row(output, pid)
                self.assertTrue(row.endswith(directory), output)
                self.assertLess(output.index(row), output.index("Enter 确认"))
            self.assertNotIn(OTHER_CARD_PID, self.contents(self.pwdx_log))

        code, output = self.run_interactive(b"\x1b", before_key=check_display)
        self.assertEqual(code, 130, output)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_pwdx_failure_does_not_skip_confirmation(self):
        for error in ("Permission denied", "No such process"):
            with self.subTest(error=error):
                def check_display(output):
                    self.assertTrue(self.process_row(output, OWN_PID).endswith("不可获取"), output)
                    self.assertTrue(self.process_row(output, OTHER_PID).endswith("/work/bob/my project"), output)
                    self.assertEqual(self.contents(self.pwdx_log), f"{OWN_PID}\n{OTHER_PID}\n")

                code, output = self.run_interactive(
                    b"\x1b",
                    overrides={"TEST_PWDX_FAILURE_PID": OWN_PID, "TEST_PWDX_ERROR": error},
                    before_key=check_display,
                )
                self.assertEqual(code, 130, output)
                self.assertEqual(self.contents(self.kill_log), "")

    def test_missing_pwdx_is_explicit_and_confirmation_still_required(self):
        def check_display(output):
            for pid in (OWN_PID, OTHER_PID):
                self.assertTrue(self.process_row(output, pid).endswith("不可获取（未安装 pwdx）"), output)
            self.assertEqual(self.contents(self.pwdx_log), "")

        code, output = self.run_interactive(
            b"\x1b", overrides={"TEST_PWDX_MISSING": "1"}, before_key=check_display
        )
        self.assertEqual(code, 130, output)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_pwdx_queries_only_deduplicated_selected_pids(self):
        row = f"| 0 | {OWN_PID} | python |\n"
        result = self.run_script(args=("-f", "0"), overrides={"TEST_SMI_ROWS": row * 2 + f"| 1 | {OTHER_CARD_PID} |\n"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.contents(self.pwdx_log), f"{OWN_PID}\n")
        self.assertTrue(self.process_row(result.stdout, OWN_PID).endswith("/work/alice/models"), result.stdout)
        self.assertLess(result.stdout.index("/work/alice/models"), result.stdout.index("锁定进程"))

    def test_empty_target_card_does_not_query_pwdx(self):
        result = self.run_script(overrides={"TEST_SMI_ROWS": f"| 1 | {OTHER_CARD_PID} |\n"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.contents(self.pwdx_log), "")

    def test_every_launcher_shows_pwdx_before_confirmation(self):
        def check_display(output):
            for pid, directory in ((OWN_PID, "/work/alice/models"), (OTHER_PID, "/work/bob/my project")):
                row = self.process_row(output, pid)
                self.assertTrue(row.endswith(directory), output)
                self.assertLess(output.index(row), output.index("Enter 确认"))

        for script in LAUNCHERS:
            with self.subTest(script=script.name):
                code, output = self.run_interactive(b"\x1b", script, before_key=check_display)
                self.assertEqual(code, 130, output)
                self.assertEqual(self.contents(self.kill_log), "")
                self.assertEqual(self.contents(self.launch_log), "")

    def test_other_keys_do_not_authorize_cleanup(self):
        for key in (b"y", b"s", b"S"):
            with self.subTest(key=key):
                code, output = self.run_interactive(key + b"\x1b")
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

    def test_uid_and_sudo_never_exempt_existing_pids(self):
        for uid, owner_uid, sudo_uid in (("1000", "1000", "1000"), ("0", "0", "0"), ("0", "1000", "1000"), ("1000", "2000", "1000")):
            with self.subTest(uid=uid, owner_uid=owner_uid, sudo_uid=sudo_uid):
                result = self.run_script(overrides={"TEST_UID": uid, "TEST_FIRST_UID": owner_uid, "SUDO_UID": sudo_uid, "TEST_SMI_ROWS": f"| 0 | {OWN_PID} |\n"})
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("没有可用的交互终端", result.stdout)
                self.assertEqual(self.contents(self.kill_log), "")
                self.assertEqual(self.contents(self.identity_log), "")

    def test_ps_failure_cannot_skip_confirmation(self):
        result = self.run_script(overrides={"TEST_PS_STATUS": "1"})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("没有可用的交互终端", result.stdout)
        self.assertEqual(self.contents(self.kill_log), "")
        code, output = self.run_interactive(b"\x1b", overrides={"TEST_PS_STATUS": "1"})
        self.assertEqual(code, 130, output)
        self.assertEqual(self.contents(self.kill_log), "")

    def test_all_cards_require_confirmation(self):
        code, output = self.run_interactive(b"\n", args=("all",))
        self.assertEqual(code, 0, output)
        self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID} {OTHER_PID} {OTHER_CARD_PID}\n")

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
                    mask = "0,1,2,3,4,5,6,7" if "all" in args else "0,1" if "1" in args else "0"
                    launch = self.contents(self.launch_log)
                    self.assertIn(f"visible:{mask}\n", launch)
                    self.assert_parallel_size(script, launch, len(mask.split(',')))
                    self.assertNotIn("--force", launch)
                    self.assertNotIn("-f ", launch)

    def test_every_launcher_no_devices_means_no_cleanup(self):
        for script in LAUNCHERS:
            for args in ((), ("-f",)):
                with self.subTest(script=script.name, args=args):
                    result = self.run_script(script, args)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn("visible:0,1,2,3,4,5,6,7\n", self.contents(self.launch_log))
                    self.assert_parallel_size(script, self.contents(self.launch_log), 1)
                    self.assertEqual(self.contents(self.device_log), "<unset>\n")
                    self.assertEqual(self.contents(self.query_log), "")
                    self.assertEqual(self.contents(self.kill_log), "")

    def test_every_launcher_noninteractive_and_kill_failure_abort(self):
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

    def test_every_launcher_skip_continues_without_killing(self):
        for script in LAUNCHERS:
            for args in (("0",), ("all",)):
                with self.subTest(script=script.name, args=args):
                    code, output = self.run_interactive(
                        b"p", script, args=args, overrides={"TEST_KILL_STATUS": "1"}
                    )
                    self.assertEqual(code, 0, output)
                    self.assertEqual(self.contents(self.kill_log), "")
                    self.assertIn("已跳过清理", output)
                    self.assertNotIn("清理完毕", output)
                    mask = "0,1,2,3,4,5,6,7" if args == ("all",) else "0"
                    launch = self.contents(self.launch_log)
                    self.assertIn(f"visible:{mask}\n", launch)
                    self.assert_parallel_size(script, launch, len(mask.split(',')))

    def test_every_launcher_single_same_uid_pid_requires_confirmation(self):
        for script in LAUNCHERS:
            for key in (b"\x1b", b"\n"):
                with self.subTest(script=script.name, key=key):
                    code, output = self.run_interactive(key, script, overrides={"TEST_SMI_ROWS": f"| 0 | {OWN_PID} |\n"})
                    self.assertIn("Enter 确认", output)
                    self.assertEqual(self.contents(self.identity_log), "")
                    if key == b"\x1b":
                        self.assertEqual(code, 130, output)
                        self.assertEqual(self.contents(self.kill_log), "")
                        self.assertEqual(self.contents(self.launch_log), "")
                    else:
                        self.assertEqual(code, 0, output)
                        self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID}\n")
                        self.assertIn("visible:0\n", self.contents(self.launch_log))

    def test_every_launcher_empty_target_card_continues_without_prompt(self):
        for script in LAUNCHERS:
            with self.subTest(script=script.name):
                result = self.run_script(script, overrides={"TEST_SMI_ROWS": f"| 1 | {OTHER_CARD_PID} |\n"})
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("Enter", result.stdout)
                self.assertEqual(self.contents(self.kill_log), "")
                self.assertIn("visible:0\n", self.contents(self.launch_log))

    def test_launchers_default_ssm_dtype_by_model_family(self):
        for script in [*LLM_LAUNCHERS, LORA]:
            with self.subTest(script=script.name):
                result = self.run_script(script, args=())
                self.assertEqual(result.returncode, 0, result.stdout)
                expected = "bfloat16" if script in HYBRID_LAUNCHERS else "<unset>"
                self.assertIn(f"ssm_dtype:{expected}\n", self.contents(self.launch_log))
                self.assertEqual(self.contents(self.kill_log), "")

    def test_hybrid_launchers_empty_ssm_dtype_uses_bfloat16(self):
        for script in HYBRID_LAUNCHERS:
            with self.subTest(script=script.name):
                result = self.run_script(
                    script, args=(), overrides={"SGLANG_MAMBA_SSM_DTYPE": ""}
                )
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("ssm_dtype:bfloat16\n", self.contents(self.launch_log))

    def test_launchers_preserve_explicit_ssm_dtype(self):
        # Only tests env forwarding, not NPU support for these override values.
        for script in [*LLM_LAUNCHERS, LORA]:
            for dtype in ("float32", "float16", "bfloat16"):
                with self.subTest(script=script.name, dtype=dtype):
                    result = self.run_script(
                        script, args=(), overrides={"SGLANG_MAMBA_SSM_DTYPE": dtype}
                    )
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn(f"ssm_dtype:{dtype}\n", self.contents(self.launch_log))

    def test_mtp_and_fia_modes_use_bfloat16_ssm_state(self):
        cases = [
            (LLM_DIR / "qwen3.5_dense_bf16.sh", (), {"MTP": "1"}),
            (LLM_DIR / "a5_fia_mixed_split_serve.sh", ("on",), {}),
            (LLM_DIR / "a5_fia_mixed_split_serve.sh", ("off",), {}),
        ]
        for script, args, overrides in cases:
            with self.subTest(script=script.name, args=args, overrides=overrides):
                result = self.run_script(script, args, overrides)
                self.assertEqual(result.returncode, 0, result.stdout)
                launch = self.contents(self.launch_log)
                self.assertIn("ssm_dtype:bfloat16\n", launch)
                if overrides.get("MTP") == "1":
                    self.assertIn("--speculative-algorithm NEXTN", launch)

    def test_launchers_are_executable(self):
        for script in [CLEANER, *LAUNCHERS, LORA]:
            with self.subTest(script=script.name):
                self.assertTrue(os.access(script, os.X_OK), f"Not executable: {script}")

    def test_shell_syntax(self):
        for script in [CLEANER, LLM_DIR / "common/npu-launch.sh", *LAUNCHERS, LORA]:
            with self.subTest(script=script.name):
                result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_device_count_and_all_for_every_launcher(self):
        for script in [*LAUNCHERS, LORA]:
            for count in (1, 2, 8):
                with self.subTest(script=script.name, count=count):
                    result = self.run_script(
                        script, ("all",),
                        {"TEST_NPU_COUNT": str(count), "TEST_SMI_ROWS": ""},
                    )
                    self.assertEqual(result.returncode, 0, result.stdout)
                    mask = ",".join(map(str, range(count)))
                    launch = self.contents(self.launch_log)
                    self.assertIn(f"visible:{mask}\n", launch)
                    self.assert_parallel_size(script, launch, count)
                    self.assertEqual(self.contents(self.device_log), "<unset>\n")

    def test_nonzero_device_ids_determine_tp_by_count(self):
        for script in [*LAUNCHERS, LORA]:
            for devices in (("3",), ("5", "3")):
                with self.subTest(script=script.name, devices=devices):
                    result = self.run_script(script, devices, {"TEST_SMI_ROWS": ""})
                    self.assertEqual(result.returncode, 0, result.stdout)
                    launch = self.contents(self.launch_log)
                    self.assertIn(f"visible:{','.join(devices)}\n", launch)
                    self.assert_parallel_size(script, launch, len(devices))
                    self.assertEqual(self.contents(self.device_log), "")

    def test_default_tp_two_is_preserved_with_all_devices_visible(self):
        script = self.root / "default-tp-two.sh"
        script.write_text(
            '#!/bin/bash\n'
            f'source "{LLM_DIR}/common/npu-launch.sh"\n'
            'npu_prepare_launch 2 "$@" || exit $?\n'
            'sglang serve --tp "$NPU_TP_SIZE"\n'
        )
        script.chmod(0o755)
        for args, tp in (((), 2), (("-f",), 2), (("3",), 1), (("all",), 8)):
            with self.subTest(args=args):
                result = self.run_script(script, args, {"TEST_SMI_ROWS": ""})
                self.assertEqual(result.returncode, 0, result.stdout)
                mask = "3" if args == ("3",) else "0,1,2,3,4,5,6,7"
                self.assertIn(f"visible:{mask}\n", self.contents(self.launch_log))
                self.assertRegex(self.contents(self.launch_log), rf"--tp {tp}(?: |$)")
                if not args or args == ("-f",):
                    self.assertEqual(self.contents(self.query_log), "")

    def test_invalid_devices_and_failed_count_abort_before_cleanup(self):
        cases = [(("all", "0"), {}), (("all", "all"), {}), (("3", "3"), {}), (("oops",), {})]
        for args in ((), ("all",)):
            for overrides in ({"TEST_DEVICE_STATUS": "1"}, {"TEST_NPU_COUNT": "0"}, {"TEST_NPU_COUNT": "invalid"}):
                cases.append((args, overrides))
        for script in [*LAUNCHERS, LORA]:
            for args, overrides in cases:
                with self.subTest(script=script.name, args=args, overrides=overrides):
                    result = self.run_script(script, args, overrides)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(self.contents(self.query_log), "")
                    self.assertEqual(self.contents(self.kill_log), "")
                    self.assertEqual(self.contents(self.launch_log), "")

    def test_all_confirmation_can_continue_or_cancel(self):
        for script in LAUNCHERS:
            for key in (b"\n", b"\x1b"):
                with self.subTest(script=script.name, key=key):
                    code, output = self.run_interactive(key, script, args=("all",))
                    self.assertEqual(code, 0 if key == b"\n" else 130, output)
                    if key == b"\n":
                        self.assertEqual(self.contents(self.kill_log), f"-9 -- {OWN_PID} {OTHER_PID} {OTHER_CARD_PID}\n")
                        self.assertIn("visible:0,1,2,3,4,5,6,7\n", self.contents(self.launch_log))
                    else:
                        self.assertEqual(self.contents(self.kill_log), "")
                        self.assertEqual(self.contents(self.launch_log), "")

    def test_diffusion_uses_native_parallelism_not_llm_tp(self):
        self.assertEqual(len(DIFFUSION_LAUNCHERS), 10)
        for script in DIFFUSION_LAUNCHERS:
            with self.subTest(script=script.name):
                result = self.run_script(script, ("5", "3"), {"TEST_SMI_ROWS": ""})
                self.assertEqual(result.returncode, 0, result.stdout)
                launch = self.contents(self.launch_log)
                self.assertIn("visible:5,3\n", launch)
                self.assert_parallel_size(script, launch, 2)
                self.assertNotIn("--tp", launch)
                self.assertNotIn("TP=", result.stdout)

    def test_diffusion_resource_paths_are_independent_of_cwd(self):
        result = self.run_script(DIFFUSION_DIR / "ti2v_bf16.sh", args=())
        self.assertEqual(result.returncode, 0, result.stdout)
        launch = self.contents(self.launch_log)
        self.assertIn(f"args:{DIFFUSION_DIR}/run_wan22_ti2v.py ", launch)
        self.assertIn(f"--image-path {DIFFUSION_DIR}/gyro.jpg ", launch)
        self.assertIn("--output-dir ./outputs_bf16", launch)

        result = self.run_script(DIFFUSION_DIR / "i2v_offline_mxfp8.sh", args=())
        self.assertEqual(result.returncode, 0, result.stdout)
        launch = self.contents(self.launch_log)
        self.assertIn(f"--image-path {DIFFUSION_DIR}/gyro.jpg ", launch)
        self.assertIn(f"--output-path {self.root}/outputs_i2v_mxfp8 ", launch)
        self.assertTrue((self.root / "outputs_i2v_mxfp8/run.log").exists())

    def test_a5_modes_and_lora_modes_accept_all(self):
        script = LLM_DIR / "a5_fia_mixed_split_serve.sh"
        for mode in ("on", "off"):
            result = self.run_script(script, (mode, "-f", "all"))
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertRegex(self.contents(self.launch_log), r"--tp 8(?: |$)")
        for kind in ("baseline", "lora"):
            for mode in ("eager", "graph"):
                for devices, tp in (((), 1), (("all",), 8), (("3", "5"), 2)):
                    with self.subTest(kind=kind, mode=mode, devices=devices):
                        result = self.run_script(LORA, (kind, mode, *devices))
                        self.assertEqual(result.returncode, 0, result.stdout)
                        self.assertRegex(self.contents(self.launch_log), rf"--tp-size {tp}(?: |$)")
                        self.assertEqual(self.contents(self.query_log), "")
                        self.assertEqual(self.contents(self.kill_log), "")


if __name__ == "__main__":
    unittest.main()
