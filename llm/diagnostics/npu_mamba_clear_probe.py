#!/usr/bin/env python3
"""Compare eager Mamba slot clearing on one idle NPU, without loading a model.

Shapes represent contiguous [layers, capacity, ...] pools, including padding.
This is a synthetic operator probe, not model accuracy or serving performance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

FEATURES = {"temporal": (16, 128, 128), "conv": (3, 4096)}
METHODS = ("index_fill", "index_copy", "flat_index_copy")


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", type=int, default=0, help="Logical visible NPU ID")
    result.add_argument("--layers", type=positive, default=30)
    result.add_argument("--capacities", type=positive, nargs="+", default=[874, 1583])
    result.add_argument("--counts", type=positive, nargs="+", default=[2, 13, 17])
    result.add_argument("--states", choices=FEATURES, nargs="+", default=list(FEATURES))
    result.add_argument("--warmup", type=positive, default=5)
    result.add_argument("--iterations", type=positive, default=20)
    result.add_argument("--repeats", type=positive, default=3)
    result.add_argument("--max-pool-gib", type=float, default=24)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--dry-run", action="store_true")
    return result


def make_plan(args):
    if (
        args.device < 0
        or not math.isfinite(args.max_pool_gib)
        or args.max_pool_gib <= 0
    ):
        raise ValueError(
            "device must be nonnegative; max-pool-gib must be finite and positive"
        )
    if max(args.counts) > min(args.capacities):
        raise ValueError("counts must not exceed any capacity")
    plan = []
    for state in dict.fromkeys(args.states):
        for capacity in dict.fromkeys(args.capacities):
            shape = (args.layers, capacity, *FEATURES[state])
            pool_bytes = math.prod(shape) * 2
            if pool_bytes > args.max_pool_gib * 2**30:
                raise ValueError(f"{shape} exceeds --max-pool-gib (BF16)")
            # One pool, one largest zero source, bounded validation scratch and
            # 2 GiB headroom. Backend workspaces may require additional memory.
            source_bytes = args.layers * max(args.counts) * math.prod(shape[2:]) * 2
            plan.append(
                {
                    "state": state,
                    "capacity": capacity,
                    "shape": shape,
                    "pool_gib": pool_bytes / 2**30,
                    "minimum_free_gib": (pool_bytes + source_bytes) / 2**30 + 2,
                }
            )
    return plan


def make_cases(torch, pool, indices):
    """No host index reads, full-pool clones, or precomputed zero sources."""
    if not pool.is_contiguous():
        raise ValueError("This probe requires a contiguous pool")
    layers, capacity = pool.shape[:2]
    count = indices.numel()

    def index_fill():
        pool.index_fill_(1, indices.to(torch.int64), 0)

    def index_copy():
        source = pool.new_zeros((layers, count, *pool.shape[2:]))
        pool.index_copy_(1, indices.to(torch.int64), source)

    def flat_index_copy():
        # Layer offsets and zero allocation are intentionally inside timing.
        row_ids = (
            torch.arange(layers, device=pool.device, dtype=torch.int64)[:, None]
            * capacity
            + indices.to(torch.int64)[None, :]
        ).flatten()
        source = pool.new_zeros((layers * count, math.prod(pool.shape[2:])))
        pool.view(layers * capacity, -1).index_copy_(0, row_ids, source)

    return dict(zip(METHODS, (index_fill, index_copy, flat_index_copy)))


def check_result(torch, pool, slots, *, chunk_rows=64):
    """Check EVERY element against ones with selected slots cleared to zero.

    Expected row values are constructed on CPU independently of NPU indexing.
    Only a bounded comparison chunk is allocated; no second full pool is needed.
    """
    layers, capacity = pool.shape[:2]
    expected = torch.ones((layers, capacity), dtype=pool.dtype)
    for slot in slots:
        expected[:, slot] = 0
    expected = expected.flatten().to(pool.device)
    flat = pool.view(layers * capacity, -1)
    checks = [
        (
            flat[start : start + chunk_rows]
            == expected[start : start + chunk_rows, None]
        ).all()
        for start in range(0, flat.shape[0], chunk_rows)
    ]
    if not torch.stack(checks).all().item():
        raise RuntimeError("Selected slots were not zeroed or unselected state changed")


def run_probe(args, plan, emit):
    import torch
    import torch_npu

    # Reuse the existing warmup, alternating-order and event/wall timing harness.
    from mxfp4_perf_probe import measure_cases, source_identity

    npu = torch.npu
    if not npu.is_available():
        raise RuntimeError("NPU unavailable; --dry-run works without torch_npu")
    npu.set_device(args.device)
    runtime = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": args.device,
        "device_name": npu.get_device_name(args.device),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "source": source_identity(__file__),
        "timing_source": source_identity(
            Path(__file__).with_name("mxfp4_perf_probe.py")
        ),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_LAUNCH_BLOCKING",
                "TASK_QUEUE_ENABLE",
                "PYTORCH_NO_NPU_MEMORY_CACHING",
            )
        },
    }
    emit({"kind": "runtime", **runtime})
    rows = []
    with torch.inference_mode():
        for item in plan:
            npu.empty_cache()
            free_bytes, _ = npu.mem_get_info()
            if free_bytes / 2**30 < item["minimum_free_gib"]:
                raise RuntimeError(
                    f"Need at least {item['minimum_free_gib']:.2f} GiB free for "
                    f"{item['shape']}; got {free_bytes / 2**30:.2f}. Use an idle NPU."
                )
            pool = torch.empty(item["shape"], dtype=torch.bfloat16, device="npu")
            for count in dict.fromkeys(args.counts):
                # Unique noncontiguous slots, including both pool boundaries.
                capacity = item["capacity"]
                slots = (
                    [capacity - 1]
                    if count == 1
                    else [
                        i * (capacity - 1) // (count - 1)
                        for i in reversed(range(count))
                    ]
                )
                indices = torch.tensor(slots, dtype=torch.int64, device="npu")
                cases = make_cases(torch, pool, indices)
                print(
                    f"Checking {item['state']} capacity={capacity} count={count}",
                    flush=True,
                )
                for name, fn in cases.items():
                    pool.fill_(1)
                    fn()
                    check_result(torch, pool, slots)
                    emit(
                        {
                            "kind": "check",
                            "state": item["state"],
                            "capacity": capacity,
                            "count": count,
                            "case": name,
                            "status": "passed",
                            "scope": "all selected and unselected elements; synthetic ones",
                        }
                    )
                timings = measure_cases(
                    cases,
                    npu,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    mode="eager",
                )
                row = {**item, "count": count, "timings": timings}
                emit({"kind": "timing", **row})
                rows.append(row)
                del fn, cases, indices
            npu.synchronize()
            del pool
    return runtime, rows


def render(runtime, rows):
    lines = [
        "=== NPU MAMBA CLEAR SUMMARY v1 ===",
        f"torch={runtime['torch']} torch_npu={runtime['torch_npu']} device={runtime['device_name']}",
        f"probe_revision={runtime['source']['git_head']}",
        "environment="
        + json.dumps(runtime.get("environment", {}), separators=(",", ":")),
        f"warmup={runtime.get('warmup')} iterations={runtime.get('iterations')} repeats={runtime.get('repeats')}",
        "shapes="
        + json.dumps(list(dict.fromkeys(tuple(row["shape"]) for row in rows))),
        "Synthetic contiguous BF16 pools; one idle NPU; no model, graph or communication.",
        "All selected/unselected elements checked before timing. NOT model accuracy acceptance.",
        "Each cell is median event_ms/wall_ms per call (launch gaps included, NOT kernel sums).",
        "Source zero allocation, index conversion and flat row IDs are included; pool setup/checks excluded.",
        "state capacity count pool_GiB | index_fill | index_copy | flat_index_copy",
    ]
    for row in rows:
        cells = [
            f"{statistics.median(row['timings'][name]['stream_ms']):.4f}/"
            f"{statistics.median(row['timings'][name]['wall_ms']):.4f}"
            for name in METHODS
        ]
        lines.append(
            f"{row['state']} {row['capacity']} {row['count']} {row['pool_gib']:.3f} | "
            + " | ".join(cells)
        )
    lines.append("=== END NPU MAMBA CLEAR SUMMARY ===")
    return "\n".join(lines) + "\n"


def main():
    args = parser().parse_args()
    try:
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        output = (
            args.output_dir
            or Path("artifacts/npu-mamba-clear")
            / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        ).resolve()
        output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for key, suffix in {
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "cache",
        "TRITON_CACHE_DIR": "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "PYTHONPYCACHEPREFIX": "cache/pycache",
        "ASCEND_CACHE_PATH": "cache/ascend",
        "ASCEND_PROCESS_LOG_PATH": "ascend-logs",
    }.items():
        path = output / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    with (output / "results.jsonl").open("x") as report:

        def emit(record):
            report.write(json.dumps(record, default=str) + "\n")
            report.flush()

        emit({"kind": "plan", "plan": plan, "options": vars(args)})
        try:
            runtime, rows = run_probe(args, plan, emit)
            summary = render(runtime, rows)
            (output / "summary.txt").write_text(summary)
            emit(
                {
                    "kind": "complete",
                    "status": "synthetic_checks_and_measurements_collected",
                }
            )
            print(summary, end="", flush=True)
        except Exception as error:  # noqa: BLE001 -- preserve partial data and fail
            emit({"kind": "error", "error": str(error)})
            (output / "error.log").write_text(traceback.format_exc())
            print(f"error: {error}; details in {output}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
