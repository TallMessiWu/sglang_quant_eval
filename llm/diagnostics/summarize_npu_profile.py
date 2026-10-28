#!/usr/bin/env python3
"""Summarize local torch_npu Text exports; only summary.txt needs copying.

No torch, network, or extra packages required. Reads kernel_details.csv and
step_trace_time.csv in each capture separately. trace_view.json supplies only
explicit SGLang step labels; kernels are never counted again from the trace.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

MAX_TEXT = 6500
TOP_OPS = 6
SETTINGS = {
    "dtype",
    "quantization",
    "tp_size",
    "ep_size",
    "dp_size",
    "disable_cuda_graph",
    "cuda_graph_max_bs",
    "disable_cuda_graph_padding",
    "attention_backend",
    "moe_runner_backend",
    "moe_a2a_backend",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "enable_mixed_chunk",
    "max_running_requests",
    "max_total_tokens",
    "mem_fraction_static",
}
# These torch_npu step_trace_time.csv columns are in microseconds, including
# versions whose column names have no unit suffix. They are NOT additive.
STEP_COLUMNS = {
    "Computing": "compute",
    "Communication": "comm",
    "Communication(Not Overlapped)": "comm_uncovered",
    "Overlapped": "overlap",
    "Free": "free",
    "Stage": "stage",
}


class JsonReader:
    """Incremental JSON values, so a large trace isn't loaded into memory."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.pos = 0
        self.eof = False

    def fill(self):
        chunk = self.stream.read(65536)
        self.buffer = self.buffer[self.pos :] + chunk
        self.pos = 0
        self.eof = not chunk

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.fill()

    def take(self, char):
        if self.peek() != char:
            raise ValueError("Unexpected JSON delimiter")
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                result, end = json.JSONDecoder().raw_decode(self.buffer, self.pos)
                # A number may continue in the next chunk.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.pos = end
                return result
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError("Incomplete or invalid JSON") from None
                self.fill()

    def array(self):
        self.take("[")
        if self.peek() != "]":
            while True:
                yield self.value()
                if self.peek() == "]":
                    break
                self.take(",")
        self.take("]")


def trace_events(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        reader = JsonReader(stream)
        if reader.peek() == "[":
            yield from reader.array()
        else:
            reader.take("{")
            found = False
            if reader.peek() != "}":
                while True:
                    key = reader.value()
                    reader.take(":")
                    if key == "traceEvents":
                        found = True
                        yield from reader.array()
                    else:
                        reader.value()
                    if reader.peek() == "}":
                        break
                    reader.take(",")
            reader.take("}")
            if not found:
                raise ValueError("No traceEvents array")
        if reader.peek():
            raise ValueError("Trailing JSON data")


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError("Expected a numeric duration") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError("Expected a finite nonnegative duration")
    return result


def compact(value, limit=80):
    return re.sub(r"\s+", " ", str(value))[:limit]


def category(name):
    """Name hints only; fused operator time remains indivisible."""
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    for label, words in (
        ("comm_launch", ("hccllaunch",)),
        ("comm", ("hccl", "hcom", "allreduce", "allgather", "reducescatter")),
        ("gmm", ("groupedmatmul", "groupmatmul", "gmm")),
        ("attention", ("attention",)),
        ("gdn/conv", ("gateddelta", "deltarule", "deltanet", "gdn", "causalconv")),
        ("linear", ("matmul", "gemm")),
        ("route", ("moeinit", "moefinal", "moedispatch", "moecombine")),
        ("quant", ("quant",)),
        ("norm", ("rmsnorm", "layernorm")),
    ):
        if any(word in name for word in words):
            return label
    return "other"


def read_steps(directory):
    path = directory / "trace_view.json"
    if not path.is_file():
        path = directory / "trace_view.json.gz"
    if not path.is_file():
        return [], "trace_missing"
    steps = Counter()
    for event in trace_events(path):
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        match = re.match(
            r"^step\[([A-Z_]+) bs=(\d+)(?: toks=(\d+))?(?:\s|\])",
            str(event.get("name", "")),
        )
        if match:
            mode, bs, tokens = match.groups()
            steps[(mode, int(bs), int(tokens) if tokens is not None else None)] += 1
    return [
        {"mode": mode, "bs": bs, "tokens": tokens, "count": count}
        for (mode, bs, tokens), count in sorted(steps.items(), key=lambda x: str(x[0]))
    ], "step_labels" if steps else "no_step_labels"


def read_capture(path, capture_id, root, operators=()):
    capture = {
        "id": capture_id,
        "source": str(path.relative_to(root)),
        "devices": {},
        "warnings": [],
    }
    devices = capture["devices"]
    # Official Text-export headers; older releases omit Device_id. Do not
    # interpret a process ID, filename order, or device ordinal as TP rank.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"Name", "Duration(us)"} <= set(rows.fieldnames or []):
            raise ValueError("Unsupported kernel_details.csv headers")
        for row in rows:
            device = (row.get("Device_id") or "").strip() or "unknown"
            stats = devices.setdefault(device, {"ops": {}, "rows": 0, "shape_rows": 0})
            duration = number(row["Duration(us)"])
            name = (row.get("Type") or row.get("Name") or "unknown").strip()
            op = stats["ops"].setdefault(name, {"count": 0, "us": 0.0})
            op["count"] += 1
            op["us"] += duration
            stats["rows"] += 1
            shape = (row.get("Input Shapes") or "").strip()
            stats["shape_rows"] += shape.lower() not in ("", "n/a", "na", "none", "--")
            if name in operators:
                # No tensor values or stacks. Keep raw shape strings in the
                # local JSON, and bound their length in the copyable report.
                signature = json.dumps([shape, row.get("Input Data Types") or ""])
                detail = op.setdefault("shapes", {}).setdefault(
                    signature,
                    {"count": 0, "us": 0.0, "min_us": duration, "max_us": duration},
                )
                detail["count"] += 1
                detail["us"] += duration
                detail["min_us"] = min(detail["min_us"], duration)
                detail["max_us"] = max(detail["max_us"], duration)
    try:
        capture["steps"], capture["phase_evidence"] = read_steps(path.parent)
    except (ValueError, OSError, EOFError) as error:
        capture["steps"] = []
        capture["phase_evidence"] = "trace_unreadable"
        capture["warnings"].append(f"trace:{type(error).__name__}")
    phases = {
        {"EXTEND": "prefill", "DECODE": "decode"}.get(
            step["mode"], step["mode"].lower()
        )
        for step in capture["steps"]
    }
    capture["phase"] = "+".join(sorted(phases)) or "unknown"
    step_file = path.parent / "step_trace_time.csv"
    if step_file.is_file():
        try:
            with step_file.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    device = (row.get("Device_id") or "").strip() or "unknown"
                    if device not in devices:
                        capture["warnings"].append("step_csv_device_unmatched")
                        continue
                    totals = devices[device].setdefault("cann_us", {})
                    for column, label in STEP_COLUMNS.items():
                        if column in row:
                            totals[label] = totals.get(label, 0.0) + number(row[column])
                    devices[device]["cann_rows"] = (
                        devices[device].get("cann_rows", 0) + 1
                    )
        except (ValueError, OSError, csv.Error) as error:
            for stats in devices.values():
                stats.pop("cann_us", None)
                stats.pop("cann_rows", None)
            capture["warnings"].append(f"step_csv:{type(error).__name__}")
    return capture


def server_settings(value):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in SETTINGS and (
                child is None or isinstance(child, (str, int, float, bool))
            ):
                result[key] = (
                    child if not isinstance(child, str) else compact(child, 48)
                )
            elif isinstance(child, (dict, list)):
                result.update(server_settings(child))
    elif isinstance(value, list):
        for child in value:
            result.update(server_settings(child))
    return result


def collect(root, operators=()):
    metadata = {}
    if (root / "metadata.json").is_file():
        metadata = json.loads((root / "metadata.json").read_text())
    trace_root = root / "traces" if (root / "traces").is_dir() else root
    paths = sorted(trace_root.rglob("kernel_details.csv"))
    result = {
        "version": 2,
        "label": compact(metadata.get("label", "unknown"), 12),
        "server_revision": compact(
            metadata.get("reported_server_revision") or "unknown", 40
        ),
        "settings": server_settings(metadata.get("server", {})),
        "operators": list(operators),
        "captures": [],
        "errors": [],
    }
    for index, path in enumerate(paths, 1):
        capture_id = f"C{index:02d}"
        try:
            capture = read_capture(path, capture_id, root, operators)
            if not capture["devices"]:
                raise ValueError("Empty kernel export")
            result["captures"].append(capture)
        except (ValueError, OSError, csv.Error) as error:
            # Paths and exception text stay out of the copyable summary.
            result["errors"].append(f"{capture_id}:kernel_csv:{type(error).__name__}")
    if not paths:
        result["errors"].append(
            "No kernel_details.csv; Text export may be missing or unfinished"
        )
    return result


def render(result, capture_id=None):
    lines = [
        "=== NPU PROFILE SUMMARY v2 ===",
        f"label={result['label']} server_revision={result['server_revision']}",
        "settings="
        + json.dumps(result["settings"], sort_keys=True, separators=(",", ":")),
        "Units: ms unless stated. Kernel sums overlap; NOT wall time or throughput.",
        "Each capture/device is separate; device ID is NOT a TP rank. Phase uses step labels only.",
        "Short profiled windows do not establish P95/P99 causes or model accuracy.",
    ]
    operators = result.get("operators", [])
    if operators:
        lines.append("Operator detail filter: " + ", ".join(operators))
    omitted = 0
    shown = 0
    for capture in result["captures"]:
        if capture_id and capture["id"] != capture_id:
            continue
        steps = capture["steps"]
        for device, stats in sorted(capture["devices"].items()):
            total = sum(op["us"] for op in stats["ops"].values())
            categories = Counter()
            for name, op in stats["ops"].items():
                categories[category(name)] += op["us"] / 1000
            block = [
                (
                    f"{capture['id']} device={compact(device, 12)} phase={compact(capture['phase'], 40)}"
                    f" ({capture['phase_evidence']}) kernels={stats['rows']} sum={total / 1000:.3f}"
                ),
            ]
            if steps:
                batches = [s["bs"] for s in steps]
                tokens = [s["tokens"] for s in steps if s["tokens"] is not None]
                block.append(
                    f"  step_labels={sum(s['count'] for s in steps)} bs={min(batches)}..{max(batches)}"
                    + (f" extend_tokens={min(tokens)}..{max(tokens)}" if tokens else "")
                    + " (capture-wide; CPU annotations)"
                )
            block.append(
                "  name_hints_ms: "
                + " ".join(f"{k}={v:.3f}" for k, v in sorted(categories.items()))
            )
            cann = stats.get("cann_us", {})
            block.append(
                "  CANN_ms: "
                + " ".join(f"{k}={v / 1000:.3f}" for k, v in cann.items())
                + f" rows={stats['cann_rows']} (overlap already reported; don't add columns)"
                if cann
                else "  CANN_ms: unavailable (communication coverage unknown)"
            )
            if cann.get("comm") == 0 and (
                categories["comm"] or categories["comm_launch"]
            ):
                block.append(
                    "  CANN comm=0 with HCCL events: communication duration is not established."
                )
            block.append(
                f"  input_shape_rows={stats['shape_rows']}/{stats['rows']}; top types: count sum_ms share avg_us"
            )
            ordered = sorted(stats["ops"].items(), key=lambda x: (-x[1]["us"], x[0]))
            if operators:
                ordered = [(name, op) for name, op in ordered if name in operators]
                if not ordered:
                    block.append("  No matching operator types in this capture/device.")
            for name, op in ordered[:TOP_OPS]:
                fraction = op["us"] / total if total else 0
                block.append(
                    f"  {compact(name, 64)}: {op['count']} {op['us'] / 1000:.3f}"
                    f" {fraction:.1%} {op['us'] / op['count']:.2f}"
                )
                shapes = sorted(op.get("shapes", {}).items(), key=lambda x: -x[1]["us"])
                for signature, detail in shapes[:6]:
                    shape, dtype = json.loads(signature)
                    block.append(
                        f"    shape={compact(shape or 'unknown', 160)} dtype={compact(dtype or 'unknown', 60)}"
                    )
                    block.append(
                        f"    count={detail['count']} sum_ms={detail['us'] / 1000:.3f}"
                        f" avg_us={detail['us'] / detail['count']:.2f}"
                        f" min_us={detail['min_us']:.2f} max_us={detail['max_us']:.2f}"
                    )
                if len(shapes) > 6:
                    block.append(
                        f"    remaining_shape_groups={len(shapes) - 6} (kept in local JSON)"
                    )
            if len(ordered) > TOP_OPS:
                rest = ordered[TOP_OPS:]
                block.append(
                    f"  remaining_types={len(rest)} sum_ms={sum(o['us'] for _, o in rest) / 1000:.3f}"
                )
            if capture["warnings"]:
                block.append("  warnings=" + ",".join(sorted(set(capture["warnings"]))))
            # Leave room for explicit omissions and errors, never truncate a number.
            if len("\n".join(lines + block)) > MAX_TEXT - 500:
                omitted += 1
            else:
                lines.extend(block)
                shown += 1
    lines.append(
        f"captures={len(result['captures'])} device_blocks_shown={shown} omitted={omitted}"
    )
    if omitted:
        lines.append(
            "More details stay in summary.json; use --capture C01 (etc.) for one capture."
        )
    if result["errors"]:
        lines.append(
            f"errors={len(result['errors'])}: " + "; ".join(result["errors"][:3])
        )
    if capture_id and not shown:
        lines.append("Requested capture not found or did not fit the summary.")
    lines.append("=== END NPU PROFILE SUMMARY ===")
    return "\n".join(lines) + "\n"


def write_summary(root, operators=()):
    result = collect(root, operators)
    text = render(result)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "summary.txt").write_text(text, encoding="utf-8")
    return result, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Existing profile run directory")
    parser.add_argument("--capture", help="Print only this capture, e.g. C01")
    parser.add_argument(
        "--operators",
        nargs="+",
        default=[],
        help="Show shape/dtype and duration ranges for up to three exact operator types",
    )
    args = parser.parse_args()
    if len(args.operators) > 3 or any(len(op) > 100 for op in args.operators):
        parser.error("Select at most three operator types, each at most 100 characters")
    try:
        root = args.directory.resolve(strict=True)
        result, text = write_summary(root, args.operators)
        selected = render(result, args.capture) if args.capture else text
        if args.operators or args.capture:
            (root / "summary-detail.txt").write_text(selected, encoding="utf-8")
        print(selected, end="")
        if args.capture and not any(
            c["id"] == args.capture for c in result["captures"]
        ):
            return 2
        return 0 if result["captures"] and not result["errors"] else 2
    except (OSError, ValueError) as error:
        print(
            f"Summary unavailable: {type(error).__name__}; check the local export directory."
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
