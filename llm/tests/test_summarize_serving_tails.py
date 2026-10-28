"""Small native benchmark fixtures; no server, NPU or performance validation."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "llm/diagnostics/summarize_serving_tails.py"
SPEC = importlib.util.spec_from_file_location("tail_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def fixture():
    return {
        "completed": 3,
        "itls": [[0.01, 1.0, 4.8, 4.9], [0.5, 1.1], [], [100.0]],
        "ttfts": [0.4, 2.0, 0.01, 100],
        "errors": ["", "", "", "private-error-body"],
        "generated_texts": ["private-generated-text"],
        "server_info": {
            "schedule": {"chunked_prefill_size": 8192, "enable_mixed_chunk": False},
            "mem_fraction_static": 0.77,
            "api_key": "private-api-key",
            "launch_command": "private-launch-command",
        },
        "mean_itl_ms": 52.0,
        "p99_itl_ms": 400.0,
    }


class ServingTailTests(unittest.TestCase):
    def test_counts_use_seconds_exclude_failed_and_count_requests_once(self):
        text = "\n".join(SUMMARY.summarize(fixture()))
        self.assertIn("itl_samples=6 max_itl_ms=4900.00", text)
        self.assertIn(
            ">500ms: itl_samples=4/6 itl_requests=2/3 ttft_requests=1/3", text
        )
        self.assertIn(
            ">1000ms: itl_samples=3/6 itl_requests=2/3 ttft_requests=1/3", text
        )
        self.assertIn(
            ">3000ms: itl_samples=2/6 itl_requests=1/3 ttft_requests=0/3", text
        )
        self.assertIn("mean=52.00", text)
        self.assertIn("p99=400.00", text)
        self.assertIn('"chunked_prefill_size":8192', text)
        self.assertNotIn("private-", text)

    def test_missing_details_are_unknown_not_zero(self):
        text = "\n".join(SUMMARY.summarize({"completed": 500}))
        self.assertIn("details=absent", text)
        self.assertNotIn("itl_samples=", text)

    def test_successful_single_token_requests_have_no_itl_maximum(self):
        record = {"completed": 1, "itls": [[]], "ttfts": [0.2], "errors": [""]}
        text = "\n".join(SUMMARY.summarize(record))
        self.assertIn("itl_samples=0 max_itl_ms=unknown", text)

    def test_invalid_details_fail_instead_of_inventing_success_or_counts(self):
        overrides = [
            {"completed": 4},
            {"errors": []},
            {"errors": ["", "", "", None]},
            {"ttfts": None},
            {"ttfts": [float("inf"), 0, 0, 0]},
            *({"itls": [[x], [], [], []]} for x in (-1, float("nan"), True, "0.3")),
        ]
        for override in overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                SUMMARY.summarize(fixture() | override)

    def test_settings_keep_conflicts_and_ignore_unrelated_fields(self):
        settings = SUMMARY.selected_settings(
            {"tp_size": 2, "nested": {"tp_size": 4}, "errors": ["private-error"]}
        )
        self.assertEqual(settings, {"tp_size": [2, 4]})

    def test_cli_selects_files_explicitly_and_keeps_records_separate(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            for index in range(3):
                path = root / f"run-{index}" / "results.jsonl"
                path.parent.mkdir()
                path.write_text(json.dumps(fixture()) + "\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), "--last", "2"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertNotIn("run-0/", result.stdout)
            self.assertEqual(result.stdout.count("details=present"), 2)
            self.assertIn("NOT independent server stalls", result.stdout)
            self.assertLess(len(result.stdout), 3500)
            self.assertNotIn("private-", result.stdout)
            path.write_text('{"private-generated-text": invalid}\n')
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("private-generated-text", result.stdout + result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
