#!/usr/bin/env python3
"""Collect an existing local SGLang NPU server's prefill/decode traces.

Run separately against BF16 and W4A8, with no competing benchmark/profiler.
Uses the normal serving benchmark; does not launch/switch/patch the server.
Profiling adds overhead: NEVER use this run's throughput as a performance result.
The client/server must share the local filesystem. CPU/GPU profiler activity
names are SGLang's API names; its NPU backend maps GPU to torch_npu NPU activity.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

SAFE_FIELDS = {
    "model_path",
    "tokenizer_path",
    "dtype",
    "quantization",
    "tp_size",
    "ep_size",
    "dp_size",
    "version",
    "startup_time",
    "disable_cuda_graph",
    "cuda_graph_bs",
    "cuda_graph_max_bs",
    "attention_backend",
    "moe_runner_backend",
    "moe_a2a_backend",
    "context_length",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "enable_mixed_chunk",
    "disable_cuda_graph_padding",
    "max_running_requests",
    "max_total_tokens",
    "mem_fraction_static",
    "enable_torch_compile",
    "backend",
    "max_bs",
    "bs",
    "enabled",
}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def filtered_info(info):
    """Retain selected resolved settings, not API keys or raw launch commands."""
    result = {}
    for key, value in info.items():
        if isinstance(value, dict):
            child = filtered_info(value)
            if child:
                result[key] = child
        elif key in SAFE_FIELDS:
            result[key] = value
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, *, post=False):
    headers = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    elif os.environ.get("API_KEY"):
        headers["Authorization"] = os.environ["API_KEY"]
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(
        Request(url, headers=headers, data=b"{}" if post else None), timeout=60
    ) as response:
        return response.read()


def benchmark_command(args, output):
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        args.server_url.rstrip("/"),
        "--tokenizer",
        str(args.model_path.resolve()),
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        str(args.dataset_path.resolve()),
        "--max-concurrency",
        str(args.concurrency),
        "--num-prompts",
        str(args.num_prompts),
        "--seed",
        "42",
        "--flush-cache",
        "--profile",
        "--profile-by-stage",
        "--profile-num-steps",
        str(args.steps),
        "--profile-activities",
        "CPU",
        "GPU",
        "--profile-output-dir",
        str(output / "traces"),
        "--profile-prefix",
        args.label,
        "--output-file",
        str(output / "benchmark-profiled.jsonl"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=["bf16", "w4a8"], required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Same tokenizer as the measured server",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:6969")
    parser.add_argument(
        "--server-revision",
        help="User-reported running server SHA, NOT inferred from the client",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/mnt/share/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"),
    )
    parser.add_argument("--concurrency", type=positive, default=32)
    parser.add_argument("--num-prompts", type=positive, default=500)
    parser.add_argument(
        "--steps",
        type=positive,
        default=5,
        help="Profile this many prefill AND decode steps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Must not exist; default ./artifacts/npu-profile/<unique-run>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without contacting the server",
    )
    args = parser.parse_args()
    url = urlsplit(args.server_url)
    if (
        url.scheme not in ("http", "https")
        or url.hostname not in ("localhost", "127.0.0.1", "::1")
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        parser.error("Use a local server root URL without credentials, query or path")
    output = (
        args.output_dir
        or Path("artifacts/npu-profile")
        / f"{args.label}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    ).resolve()
    command = benchmark_command(args, output)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    started = False
    process = None
    try:
        for variable, suffix in {
            "TMPDIR": "tmp",
            "XDG_CACHE_HOME": "cache",
            "HF_HOME": "cache/hf",
            "HF_HUB_CACHE": "cache/hf/hub",
            "TRITON_CACHE_DIR": "cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
            "PYTHONPYCACHEPREFIX": "cache/pycache",
            "ASCEND_PROCESS_LOG_PATH": "ascend-logs",
        }.items():
            path = output / suffix
            path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(path)
        info = json.loads(request(args.server_url.rstrip("/") + "/server_info"))
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "label": args.label,
                    "reported_server_revision": args.server_revision,
                    "server": filtered_info(info),
                    "command": command,
                    "warning": "Profiled throughput is not a benchmark result; trace durations may overlap",
                },
                indent=2,
            )
        )
        if "quantization" in info:
            expected = None if args.label == "bf16" else "mxfp_w4a8"
            if info["quantization"] != expected:
                raise ValueError(
                    f"Label {args.label} does not match server quantization={info['quantization']!r}"
                )
        print(
            f"Profile artifacts: {output}\nDo not compare throughput from this profiled run.",
            flush=True,
        )
        with (output / "benchmark.log").open("x") as log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=output,
            )
            for line in process.stdout:
                started = started or "Profiler started" in line
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"Benchmark exited {code}; see benchmark.log")
        if not started:
            raise RuntimeError(
                "Benchmark did not acknowledge profiler startup; do not treat this as a trace"
            )
        # The native finite-step profiler stops itself. Do not stop a new
        # profiler that another client could have started after our capture.
        traces = [
            str(p.relative_to(output))
            for p in (output / "traces").rglob("*")
            if p.is_file()
        ]
        if not traces:
            raise RuntimeError(
                "No local trace files produced; check server profiler logs and filesystem"
            )
        started = False
        (output / "traces.json").write_text(json.dumps(traces, indent=2))
        # Import here also keeps dry-run and metadata inspection dependency-free.
        from summarize_npu_profile import write_summary

        summary, summary_text = write_summary(output)
        print(summary_text, end="", flush=True)
        print(f"Copy only {output / 'summary.txt'}; raw traces stay on this server.")
        if summary["errors"] or not summary["captures"]:
            raise RuntimeError(
                "Capture collected but kernel summary incomplete; copy summary.txt for diagnosis"
            )
        return 0
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- preserve diagnostics and stop owned profiler
        (output / "error.txt").write_text(str(error) or "Interrupted")
        print(f"error: {error}; partial artifacts kept in {output}", file=sys.stderr)
        return 2
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if started:
            try:
                stop_profile(args.server_url)
            except Exception as error:  # noqa: BLE001 -- cleanup must preserve the original failure
                print(
                    f"Profiler cleanup failed: {error}. Check /stop_profile on this server.",
                    file=sys.stderr,
                )


def stop_profile(server_url):
    try:
        request(server_url.rstrip("/") + "/stop_profile", post=True)
    except HTTPError as error:
        # Finite-step profiles normally finish before the benchmark completes.
        message = error.read().decode(errors="replace")
        if "Profiling is not in progress" not in message:
            raise


if __name__ == "__main__":
    raise SystemExit(main())
