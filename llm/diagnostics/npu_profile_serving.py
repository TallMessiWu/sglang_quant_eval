#!/usr/bin/env python3
"""Collect an existing local SGLang NPU server's prefill/decode traces.

Run separately against BF16 and W4A8, with no competing benchmark/profiler.
Uses the normal serving benchmark; does not launch/switch/patch the server.
Profiling adds overhead: NEVER use the profiled run's throughput as a performance result.
--diagnose first runs short prefill/decode workloads and one unprofiled ShareGPT
pass, then captures a shorter profile and produces report.txt/report.json.
The client/server must share the local filesystem. CPU/GPU profiler activity
names are SGLang's API names; its NPU backend maps GPU to torch_npu NPU activity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
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
    "max_mamba_cache_size",
    "mamba_full_memory_ratio",
    "max_total_num_tokens",
    "effective_max_running_requests_per_dp",
    "disable_radix_cache",
    "kv_cache_dtype",
    "mamba_scheduler_strategy",
    "weight",
    "kvcache",
    "startup_available",
    "token_capacity",
    "token_capacity_swa",
    "decode",
    "target_verify",
    "SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT",
    "SGLANG_NPU_MAMBA_CLEAR_USE_INDEX_COPY",
    "SGLANG_NPU_MAMBA_CLEAR_TRACE",
    "SGLANG_MAMBA_SSM_DTYPE",
    "SGLANG_PROFILE_RECORD_SHAPES",
    "SGLANG_PROFILE_WITH_STACK",
}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def filtered_info(info):
    """Retain selected resolved settings, not API keys or raw launch commands."""
    if isinstance(info, list):
        return [filtered_info(item) for item in info if isinstance(item, dict)]
    result = {}
    for key, value in info.items():
        if isinstance(value, dict) or (
            isinstance(value, list) and any(isinstance(v, dict) for v in value)
        ):
            child = filtered_info(value)
            if child:
                result[key] = child
        elif key in SAFE_FIELDS:
            result[key] = value
    return result


def validate_server(info, label):
    expected = None if label == "bf16" else "mxfp_w4a8"
    values = []

    def visit(node):
        if isinstance(node, dict):
            if "quantization" in node:
                values.append(node["quantization"])
            for value in node.values():
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(info)
    if not values or any(value != expected for value in values):
        raise ValueError(
            f"Label {label} does not match reported quantization={values!r}"
        )


def source_identity(path):
    path = Path(path).resolve()
    result = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for key, command in (
        ("git_head", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--porcelain", "--untracked-files=no"]),
    ):
        process = subprocess.run(
            ["git", "-C", str(path.parent), *command],
            capture_output=True,
            text=True,
            check=False,
        )
        result[key] = process.stdout.strip() if process.returncode == 0 else None
    return result


def environment_info(model_path, dataset_path):
    """Client identity only; never infer the running server SHA from this checkout."""
    versions = {}
    for name in (
        "sglang",
        "torch",
        "torch_npu",
        "sgl-kernel-npu",
        "transformers",
        "triton-ascend",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    spec = importlib.util.find_spec("sglang")
    sources = {}
    if spec is not None and spec.origin:
        base = Path(spec.origin).parent
        for suffix in (
            "__init__.py",
            "benchmark/serving.py",
            "srt/mem_cache/memory_pool.py",
            "srt/hardware_backend/npu/quantization/moe_methods.py",
            "srt/hardware_backend/npu/quantization/linear_method_npu.py",
        ):
            path = base / suffix
            if path.is_file():
                sources[suffix] = source_identity(path)
    hashes = {}
    for path in (model_path / "config.json", dataset_path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    cann = {}
    for prefix in (
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/driver",
    ):
        if prefix:
            path = Path(prefix) / "version.info"
            if path.is_file():
                try:
                    cann[str(path)] = path.read_text(errors="replace")[:4096]
                except OSError as error:
                    cann[str(path)] = f"unavailable: {type(error).__name__}"
    return {
        "scope": "client environment, NOT proof of server revision/wheel",
        "packages": versions,
        "sources": sources,
        "input_sha256": hashes,
        "cann_version_files": cann,
    }


def hardware_snapshot(output, name):
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=False
        )
        text = result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        text = f"unavailable: {type(error).__name__}"
    (output / f"npu-smi-{name}.txt").write_text(text)


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


def serving_command(args, result_path, *, count, dataset="sharegpt"):
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
        dataset,
        "--dataset-path",
        str(args.dataset_path.resolve()),
        "--max-concurrency",
        str(args.concurrency),
        "--num-prompts",
        str(count),
        "--seed",
        "42",
        "--flush-cache",
        "--output-file",
        str(result_path),
    ]


def benchmark_command(args, output):
    count = (
        min(args.num_prompts, args.profile_prompts)
        if args.diagnose
        else args.num_prompts
    )
    return serving_command(args, output / "benchmark-profiled.jsonl", count=count) + [
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
    ]


def diagnostic_plan(args, output):
    cases = []
    for name, count, input_len, output_len in (
        ("prefill", 2 * args.concurrency, 1024, 1),
        ("decode", args.concurrency, 128, 256),
        ("sharegpt", args.num_prompts, None, None),
    ):
        command = serving_command(
            args,
            output / name / "results.jsonl",
            count=count,
            dataset="sharegpt" if input_len is None else "random",
        )
        command += ["--output-details"]
        if input_len is not None:
            # Pass token IDs: text decode/re-encode need not preserve the requested length.
            command += [
                "--random-input-len",
                str(input_len),
                "--random-output-len",
                str(output_len),
                "--random-range-ratio",
                "1",
                "--tokenize-prompt",
            ]
        cases.append({"name": name, "count": count, "command": command})
    cases.append({"name": "profile", "command": benchmark_command(args, output)})
    return cases


def run_diagnostic_cases(args, output, cases, report):
    from summarize_serving_tails import summarize

    for case in cases:
        directory = output / case["name"]
        directory.mkdir()
        print(
            f"Collecting {case['name']} (no profiler); log: {directory / 'bench.log'}",
            flush=True,
        )
        with (directory / "bench.log").open("x") as log:
            subprocess.run(
                case["command"],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=directory,
                timeout=args.case_timeout,
                check=True,
            )
        records = [
            json.loads(line)
            for line in (directory / "results.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len(records) != 1 or records[0].get("completed") != case["count"]:
            raise RuntimeError(
                f"{case['name']}: missing or failed requests; see local results.jsonl"
            )
        lines = summarize(records[0])
        # Only derived statistics are copied out; raw text/errors/details stay local.
        report["unprofiled"][case["name"]] = lines
        (directory / "summary.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


def write_diagnostic_report(output, report):
    lines = [
        "=== W4A8 DIAGNOSTIC REPORT ===",
        "One diagnostic pass, not repeated performance/accuracy acceptance.",
        "Synthetic prefill/decode are serving workloads, not isolated kernel measurements.",
    ]
    for name, summary in report["unprofiled"].items():
        lines += [f"[{name}; unprofiled]", *summary]
    if report.get("error"):
        lines.append("INCOMPLETE: " + report["error"])
    if (output / "summary.txt").is_file():
        lines.append((output / "summary.txt").read_text())
    lines.append("=== END W4A8 DIAGNOSTIC REPORT ===")
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"Return {output / 'report.txt'} and {output / 'report.json'}; raw requests/traces stay local.",
        flush=True,
    )


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
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="One pass: short prefill/decode workloads, unprofiled ShareGPT, then profile",
    )
    parser.add_argument(
        "--profile-prompts",
        type=positive,
        default=128,
        help="In --diagnose, cap only the profiled workload (default 128)",
    )
    parser.add_argument(
        "--case-timeout",
        type=positive,
        default=600,
        help="Timeout in seconds for each unprofiled diagnostic case",
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
    plan = diagnostic_plan(args, output) if args.diagnose else None
    if args.dry_run:
        print(json.dumps(plan if args.diagnose else command, indent=2))
        return 0
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    started = False
    process = None
    report = {"unprofiled": {}, "metadata": {}, "profile": None}
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
        metadata = {
            "label": args.label,
            "reported_server_revision": args.server_revision,
            "server": filtered_info(info),
            "command": command,
            "warning": "Profiled throughput is not a benchmark result; trace durations may overlap",
        }
        report["metadata"] = metadata
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        validate_server(info, args.label)
        if args.diagnose:
            metadata["plan"] = plan
            metadata["client"] = environment_info(
                args.model_path.resolve(), args.dataset_path.resolve()
            )
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
            hardware_snapshot(output, "before")
            run_diagnostic_cases(args, output, plan[:-1], report)
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

        summary, summary_text = write_summary(output, all_shapes=args.diagnose)
        print(summary_text, end="", flush=True)
        print(f"Copy only {output / 'summary.txt'}; raw traces stay on this server.")
        if args.diagnose:
            # All operator shapes in compact JSON; read large traces only once.
            report["profile"] = summary
            after = json.loads(request(args.server_url.rstrip("/") + "/server_info"))
            report["server_after"] = filtered_info(after)
            validate_server(after, args.label)
            if after.get("startup_time") != info.get("startup_time"):
                raise RuntimeError(
                    "Server restarted during collection; results are not a paired run"
                )
        if summary["errors"] or not summary["captures"]:
            raise RuntimeError(
                "Capture collected but kernel summary incomplete; copy summary.txt for diagnosis"
            )
        return 0
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- preserve diagnostics and stop owned profiler
        report["error"] = (
            type(error).__name__ + ": see local error.txt / benchmark logs"
        )
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
        if args.diagnose:
            hardware_snapshot(output, "after")
            report["hardware"] = {
                name: (output / f"npu-smi-{name}.txt").read_text()
                for name in ("before", "after")
                if (output / f"npu-smi-{name}.txt").exists()
            }
            write_diagnostic_report(output, report)


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
