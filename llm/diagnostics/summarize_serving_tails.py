#!/usr/bin/env python3
"""Summarize local native SGLang benchmark JSONL without printing request text.

ITLs may be amortized across tokens in a stream chunk. Plain --output-details
lacks a shared timeline; optional --trace-stalls raw intervals support same-host
correlation. Neither interval overlap nor host timing establishes a device cause.
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


def summarize(record, server_records=()):
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
    lines.extend(summarize_timeline(record, successful, server_records))
    return lines


def interval(start, end):
    if not number(start) or not number(end) or end < start:
        raise ValueError("invalid timeline interval")
    return start, end


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def overlap_groups(spans):
    """Connected overlap groups, NOT distinct causal incidents."""
    groups = []
    for span in sorted(spans):
        if not groups or span[0] >= groups[-1][1]:
            groups.append([span[0], span[1], [span]])
        else:
            groups[-1][1] = max(groups[-1][1], span[1])
            groups[-1][2].append(span)
    return sorted(groups, key=lambda group: group[1] - group[0], reverse=True)


def peak_window(spans):
    events = sorted(
        (time, delta) for a, b, *_ in spans for time, delta in ((a, 1), (b, -1))
    )
    active = peak = 0
    window = None
    for i, (at, delta) in enumerate(events[:-1]):
        active += delta
        following = events[i + 1][0]
        if active > peak and following > at:
            peak, window = active, (at, following)
    return peak, window


def summarize_timeline(record, successful, server_records):
    trace = record.get("stall_trace")
    if trace is None:
        return [
            "timeline=absent; TRACE_STALLS=1 is needed for overlapping-gap analysis"
        ]
    if trace.get("version") != 1 or trace.get("clock") != "perf_counter":
        raise ValueError("unsupported stall timeline version or clock")
    run = interval(trace["started"], trace["ended"])
    starts, first_tokens, gaps = (
        trace[key] for key in ("request_starts", "request_first_tokens", "stream_gaps")
    )
    if not len(starts) == len(record["errors"]) == len(gaps) == len(first_tokens):
        raise ValueError("stall timeline request count mismatch")
    spans = []
    for i in successful:
        a, b = interval(starts[i], first_tokens[i])
        if b - a > 1:
            spans.append((a, b, i, "ttft"))
        for a, b, tokens in gaps[i]:
            interval(a, b)
            if not number(tokens) or tokens <= 0:
                raise ValueError("invalid stream chunk token count")
            if b - a > 1:
                spans.append((a, b, i, "stream"))
    if any(a < run[0] or b > run[1] for a, b, *_ in spans):
        raise ValueError("request interval outside benchmark window")
    lags = [interval(*lag) for lag in trace["client_loop_lags"]]
    # Matching hostnames is a guard, not proof of clock identity. The caller
    # must collect the server and benchmark in the same host/time namespace.
    matching = [row for row in server_records if row.get("host") == trace.get("host")]
    clears = [row for row in matching if overlaps((row["entered"], row["exited"]), run)]
    groups = overlap_groups(spans)
    lines = [
        (
            f"timeline=present raw_gaps_or_ttfts_gt1s={len(spans)} overlap_groups={len(groups)} "
            f"client_lags_gt100ms={len(lags)} clear_records_in_run={len(clears)}"
        ),
        "  Groups join overlapping intervals; they are NOT proven shared server pauses.",
    ]
    if not clears:
        lines.append(
            "  clear_correlation=unknown (no matching server trace in this run)"
        )
    for index, (a, b, group) in enumerate(groups[:4], 1):
        peak, window = peak_window(group)
        loop_ms = (
            max((y - x for x, y in lags if overlaps((x, y), window)), default=0) * 1000
        )
        selected = [
            row for row in clears if overlaps((row["entered"], row["exited"]), window)
        ]
        affected = {span[2] for span in group}
        streams = {span[2] for span in group if span[3] == "stream"}
        ttfts = {span[2] for span in group if span[3] == "ttft"}
        lines.append(
            f"  G{index:02d} run_s={a - run[0]:.3f}..{b - run[0]:.3f} "
            f"requests={len(affected)} stream={len(streams)} ttft={len(ttfts)} "
            f"peak_intervals={peak} common_ms={(window[1] - window[0]) * 1000:.2f} "
            f"client_lag_overlap_max_ms={loop_ms:.2f}"
        )
        if selected:
            lines.append("    " + clear_summary(selected))
    if len(groups) > 4:
        lines.append(f"  omitted_overlap_groups={len(groups) - 4}")
    for pid in sorted({row["pid"] for row in clears}):
        rows = [row for row in clears if row["pid"] == pid]
        lines.append(f"  process={pid} " + clear_summary(rows))
    return lines


def clear_summary(rows):
    def maximum(field):
        return max(field(row) for row in rows) * 1000

    deltas = []
    for row in rows:
        before, after = (
            row[side].get("num_alloc_retries") for side in ("before", "after")
        )
        if number(before) and number(after) and after >= before:
            deltas.append(after - before)
    retries = sum(deltas) if len(deltas) == len(rows) else "unknown"
    return (
        f"clears={len(rows)} methods={','.join(sorted({row['method'] for row in rows}))} "
        f"max_host_ms={maximum(lambda row: row['ended'] - row['started']):.2f} "
        f"max_alloc_host_ms={maximum(lambda row: row.get('allocation_ended', row['started']) - row['started']):.2f} "
        f"max_stats_ms={maximum(lambda row: row['started'] - row['entered'] + row['exited'] - row['ended']):.2f} "
        f"retry_delta_inside_clears={retries} max_zero_MiB={max(row['zero_bytes'] for row in rows) / 2**20:.2f}"
    )


def read_server_records(path):
    records = []
    if path is None:
        return records
    marker = "[NPU_MAMBA_CLEAR_TRACE] "
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker not in line:
                continue
            try:
                # Logging formatters may append an ANSI reset after the JSON.
                row, _ = json.JSONDecoder().raw_decode(
                    line.split(marker, 1)[1].lstrip()
                )
                if row["version"] != 1 or row["clock"] != "perf_counter":
                    raise ValueError("unsupported clear trace")
                interval(row["entered"], row["started"])
                interval(row["started"], row["ended"])
                interval(row["ended"], row["exited"])
                records.append(row)
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "malformed NPU_MAMBA_CLEAR_TRACE record in server log"
                ) from None
    return records


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
    parser.add_argument(
        "--server-log", type=Path, help="Optional same-host clear diagnostic log"
    )
    args = parser.parse_args()
    if args.last is not None and args.last <= 0:
        parser.error("--last must be positive")
    lines = [
        "=== SERVING TAIL SUMMARY v2 ===",
        "Each JSONL record is separate; no warmup removal or percentile pooling.",
        "ITL samples may be amortized per token; counts are NOT independent server stalls.",
        "Trace correlation requires the same host/time namespace; overlap is not causality.",
        "Clear spans are HOST enqueue/wait time, not asynchronous NPU kernel duration.",
        "Server SHA and clear/route flags are not inferred; retain launcher evidence.",
    ]
    if args.last is not None:
        lines.append(f"selection=last {args.last} files per directory (path order)")
    records = 0
    try:
        server_records = read_server_records(args.server_log)
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
                    lines.extend(
                        "  " + row for row in summarize(record, server_records)
                    )
                if not nonempty:
                    raise ValueError(f"empty result file: {path}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Cannot summarize: {exc}\n")
    lines.append("=== END SERVING TAIL SUMMARY ===")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
