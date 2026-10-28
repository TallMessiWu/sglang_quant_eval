#!/usr/bin/env python3
"""Summarize local native SGLang benchmark JSONL without printing request text.

ITLs are client samples, potentially amortized across tokens in a stream chunk.
No request start timestamps are saved by --output-details: samples cannot be
deduplicated into server stalls or attributed to prefill, kernels or allocation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SETTINGS = {
    "quantization",
    "tp_size",
    "dp_size",
    "ep_size",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "enable_mixed_chunk",
    "disable_cuda_graph",
    "mem_fraction_static",
    "max_running_requests",
    "max_total_tokens",
    "max_total_num_tokens",
    "max_mamba_cache_size",
    "mamba_full_memory_ratio",
}
THRESHOLDS_MS = (500, 1000, 3000)


def selected_settings(info):
    """Support flat and grouped server_info; omit commands, credentials and text."""
    found = {}

    def visit(node):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in SETTINGS and isinstance(value, (str, int, float, type(None))):
                values = found.setdefault(key, [])
                if value not in values:
                    values.append(value)
            elif isinstance(value, dict):
                visit(value)

    visit(info)
    return {key: vs[0] if len(vs) == 1 else vs for key, vs in sorted(found.items())}


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def fmt(value):
    return f"{value:.2f}" if number(value) else "unknown"


def summarize(record):
    lines = [
        "settings="
        + json.dumps(
            selected_settings(record.get("server_info")), separators=(",", ":")
        ),
        (
            f"completed={record.get('completed', 'unknown')} "
            f"concurrency_limit={record.get('max_concurrency', 'unknown')} "
            f"input_tokens={record.get('total_input_tokens', 'unknown')} "
            f"output_tokens={record.get('total_output_tokens', 'unknown')} "
            f"duration_s={fmt(record.get('duration'))} "
            f"output_tok_s={fmt(record.get('output_throughput'))}"
        ),
    ]
    for metric in ("ttft", "tpot", "itl"):
        values = " ".join(
            f"{stat}={fmt(record.get(f'{stat}_{metric}_ms'))}"
            for stat in ("mean", "median", "p95", "p99")
        )
        lines.append(f"{metric}_ms: {values}")

    if "itls" not in record:
        lines.append(
            "details=absent; long-ITL counts unknown (rerun with OUTPUT_DETAILS=1)"
        )
        return lines
    itls, errors, ttfts = (record.get(key) for key in ("itls", "errors", "ttfts"))
    if not all(isinstance(values, list) for values in (itls, errors, ttfts)):
        raise ValueError("details require itls, errors and ttfts arrays")
    if not len(itls) == len(errors) == len(ttfts):
        raise ValueError("detail arrays have different request counts")
    if not all(isinstance(error, str) for error in errors):
        raise ValueError("errors must be strings")
    successful = [i for i, error in enumerate(errors) if not error]
    if len(successful) != record.get("completed"):
        raise ValueError(
            "completed count disagrees with detail errors; success is ambiguous"
        )
    samples = []
    maxima = []
    first_tokens = []
    for i in successful:
        row = itls[i]
        if not isinstance(row, list) or not all(number(x) for x in row):
            raise ValueError(
                "successful-request ITLs must be finite nonnegative seconds"
            )
        if not number(ttfts[i]):
            raise ValueError(
                "successful-request TTFTs must be finite nonnegative seconds"
            )
        samples.extend(row)
        maxima.append(max(row, default=0))
        first_tokens.append(ttfts[i])
    maximum = fmt(max(samples) * 1000) if samples else "unknown"
    lines.append(
        f"details=present requests={len(successful)} failed={len(errors) - len(successful)} "
        f"itl_samples={len(samples)} max_itl_ms={maximum}"
    )
    for threshold in THRESHOLDS_MS:
        seconds = threshold / 1000
        count = sum(x > seconds for x in samples)
        affected = sum(x > seconds for x in maxima)
        ttft_count = sum(x > seconds for x in first_tokens)
        lines.append(
            f">{threshold}ms: itl_samples={count}/{len(samples)} "
            f"itl_requests={affected}/{len(successful)} "
            f"ttft_requests={ttft_count}/{len(successful)}"
        )
    return lines


def result_files(paths, last):
    seen = set()
    for path in paths:
        candidates = sorted(path.rglob("results.jsonl")) if path.is_dir() else [path]
        if not candidates:
            raise ValueError(f"no results.jsonl under {path}")
        if last is not None and path.is_dir():
            candidates = candidates[-last:]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="JSONL files or benchmark roots"
    )
    parser.add_argument(
        "--last", type=int, help="last N files per directory, sorted by path"
    )
    args = parser.parse_args()
    if args.last is not None and args.last <= 0:
        parser.error("--last must be positive")
    lines = [
        "=== SERVING TAIL SUMMARY v1 ===",
        "Each JSONL record is separate; no warmup removal or percentile pooling.",
        "ITL samples may be amortized per token; counts are NOT independent server stalls.",
        "No shared timeline: cannot establish phase, kernel or allocation cause.",
        "Server SHA and clear/route flags are not inferred; retain launcher evidence.",
    ]
    if args.last is not None:
        lines.append(f"selection=last {args.last} files per directory (path order)")
    records = 0
    try:
        for path in result_files(args.paths, args.last):
            with path.open() as stream:
                nonempty = False
                for lineno, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    nonempty = True
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"invalid JSON at {path.name}:{lineno}"
                        ) from None
                    if not isinstance(record, dict):
                        raise TypeError("each JSONL record must be an object")
                    records += 1
                    lines.append(
                        f"R{records:02d} {path.parent.parent.name}/{path.parent.name}/{path.name}:{lineno}"
                    )
                    lines.extend("  " + row for row in summarize(record))
                if not nonempty:
                    raise ValueError(f"empty result file: {path}")
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Cannot summarize: {exc}\n")
    lines.append("=== END SERVING TAIL SUMMARY ===")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
