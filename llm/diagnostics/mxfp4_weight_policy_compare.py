#!/usr/bin/env python3
"""Compare online MXFP4 weight policies on real BF16/FP16 safetensors (one NPU).

A: round + default scale settings (the current W4A8 weight call).
B: rint  + default scale settings (rounding-only ablation).
C: rint  + scale_alg=2, dst_type_max=7.25 (the proposed C7 policy).

Examples, from the repository root in the serving Python environment:
  python llm/diagnostics/mxfp4_weight_policy_compare.py --model-path /weights/BF16 --list
  python llm/diagnostics/mxfp4_weight_policy_compare.py --model-path /weights/BF16 \
      --output ./artifacts/mxfp4-policy/results.jsonl
  # Repeat --tensor for exact checkpoint keys; --rows 0 reads all rows per expert.
  python llm/diagnostics/mxfp4_weight_policy_compare.py --model-path /weights/BF16 \
      --tensor model.language_model.layers.0.mlp.experts.gate_up_proj --rows 0

Defaults sample first/middle/last text layers, first/last experts, and 128 rows
spread over the beginning/middle/end of each matrix (including both gate/up
halves). Each row retains its FULL K axis: no quantization block is truncated.
Only the selected slices are loaded, not the entire checkpoint or expert bank.
2D [N,K] Linears and 3D [E,N,K] expert banks are supported; expert slices retain
an E=1 axis for the NPU call. No SGLang import, model launch, or source patching.

This measures weight reconstruction error, NOT Linear/GMM execution, model
accuracy, loading peak memory, or performance. Exit 0 means measurements were
collected, NOT that C is better. Exit 2 means invalid input/environment/operator
failure; partial results and the error remain in the JSONL. Existing reports are
never overwritten. All outputs default to the current working directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from modelslim_w4a8_check import dequantize_weight, git_head, metrics, ue8m0_scale

POLICIES = {
    "A": {"round_mode": "round"},
    "B": {"round_mode": "rint"},
    "C": {"round_mode": "rint", "scale_alg": 2, "dst_type_max": 7.25},
}
PROJECTION = re.compile(
    r"(?:^|\.)(gate_proj|up_proj|gate_up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)"
    r"(?:\.weight)?$"
)
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
EXPERT = re.compile(r"\.experts\.(\d+)\.")


def spread(values, count):
    """Deterministic, evenly spaced choices, including both endpoints."""
    values = sorted(values)
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[i * (len(values) - 1) // (count - 1)] for i in range(count)]


def row_ranges(size, limit):
    if limit == 0 or size <= limit:
        return [(0, size)]
    # Three contiguous reads instead of hundreds of individual mmap slices.
    chunks = min(3, limit)
    widths = [limit // chunks + (i < limit % chunks) for i in range(chunks)]
    starts = (
        [0]
        if chunks == 1
        else [i * (size - widths[i]) // (chunks - 1) for i in range(chunks)]
    )
    return [(start, start + width) for start, width in zip(starts, widths)]


def checkpoint_index(root):
    from safetensors import safe_open

    if not root.is_dir():
        raise ValueError(f"Not a checkpoint directory: {root}")
    if (root / "quant_model_description.json").exists():
        raise ValueError(
            "Use the original BF16/FP16 checkpoint, not a ModelSlim export"
        )
    index = root / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    mapping = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 -- safe_open is not a mapping
                if name in mapping:
                    raise ValueError(f"Duplicate checkpoint key: {name}")
                mapping[name] = path.name
    if not mapping:
        raise ValueError(f"No safetensors found in {root}")
    return mapping


def select_names(mapping, tensors, layers, experts):
    if tensors:
        missing = sorted(set(tensors) - mapping.keys())
        if missing:
            raise ValueError(f"Checkpoint keys not found: {missing}")
        return list(dict.fromkeys(tensors))
    candidates = [
        name
        for name in mapping
        if PROJECTION.search(name)
        and LAYER.search(name)
        and not ({"visual", "vision", "vision_tower"} & set(name.split(".")))
    ]
    layer_ids = spread({int(LAYER.search(n)[1]) for n in candidates}, layers)
    # Legacy checkpoints store each expert as a separate 2D tensor. Select
    # expert IDs per layer, just as for the fused 3D expert-bank format below.
    expert_ids = {
        layer: spread(
            {
                int(EXPERT.search(n)[1])
                for n in candidates
                if int(LAYER.search(n)[1]) == layer and EXPERT.search(n)
            },
            experts,
        )
        for layer in layer_ids
    }
    names = [
        n
        for n in candidates
        if int(LAYER.search(n)[1]) in layer_ids
        and (
            not EXPERT.search(n)
            or int(EXPERT.search(n)[1]) in expert_ids[int(LAYER.search(n)[1])]
        )
    ]
    if not names:
        raise ValueError(
            "No supported text projection keys found; use --tensor EXACT_KEY"
        )
    return sorted(names, key=lambda n: (int(LAYER.search(n)[1]), n))


def read_samples(handle, name, rows, experts):
    data = handle.get_slice(name)
    shape, dtype = data.get_shape(), data.get_dtype()
    if len(shape) not in (2, 3) or dtype not in ("BF16", "F16"):
        raise ValueError(f"Expected BF16/FP16 [N,K] or [E,N,K], got {dtype} {shape}")
    if any(dim <= 0 for dim in shape) or shape[-1] % 64:
        raise ValueError(
            f"Expected nonempty weights with K divisible by 64, got {shape}"
        )
    ranges = row_ranges(shape[-2], rows)
    ids = spread(range(shape[0]), experts) if len(shape) == 3 else [None]
    for expert in ids:
        pieces = [
            data[start:stop, :]
            if expert is None
            else data[expert : expert + 1, start:stop, :]
            for start, stop in ranges
        ]
        sample = torch.cat(pieces, dim=-2).contiguous()
        yield (
            sample,
            {
                "name": name,
                "checkpoint_shape": shape,
                "dtype": dtype,
                "expert": expert,
                "row_ranges": ranges,
                "sample_shape": list(sample.shape),
            },
        )


def quantize(weight, fp4_dtype, policy):
    # Match production: default tail-axis/block settings, torch_npu FP4 enum,
    # and no scale kwargs for A/B. Do NOT substitute torch.float4_e2m1fn_x2.
    packed, scale = torch.ops.npu.npu_dynamic_mx_quant(
        weight, dst_type=fp4_dtype, **POLICIES[policy]
    )
    torch.npu.synchronize()
    packed_cpu = packed.view(torch.uint8).cpu()
    scale_cpu = scale.view(torch.uint8).cpu()
    n, k = weight.numel() // weight.shape[-1], weight.shape[-1]
    if tuple(packed_cpu.shape) != (*weight.shape[:-1], k // 2):
        raise ValueError(f"Unexpected packed FP4 shape: {tuple(packed_cpu.shape)}")
    if scale_cpu.numel() != n * k // 32:
        raise ValueError(f"Unexpected UE8M0 scale shape: {tuple(scale_cpu.shape)}")
    layout = {
        "packed_shape": list(packed_cpu.shape),
        "scale_shape": list(scale_cpu.shape),
    }
    return packed_cpu.reshape(n, k // 2), scale_cpu.reshape(n, k // 32), layout


def weight_metrics(weight, packed, scale):
    reference = weight.float().reshape(-1, weight.shape[-1])
    reconstructed = dequantize_weight(packed, scale)
    result = metrics(reconstructed, reference)
    if not result["finite"]:
        raise ValueError("Nonfinite input or reconstructed weights")
    diff = reconstructed.double() - reference.double()
    squared_error = float(diff.square().sum())
    energy = float(reference.double().square().sum())
    # NMSE is undefined for an all-zero reference, unlike absolute MSE.
    result.update(
        elements=reference.numel(),
        squared_error=squared_error,
        energy=energy,
        mse=squared_error / reference.numel(),
        nmse=squared_error / energy if energy else None,
        mean_error=float(diff.mean()),
        zero_fraction=float((reconstructed == 0).float().mean()),
        # Being encoded as +/-6 is NOT proof of clipping. Count original values
        # exceeding the scale-dependent representable maximum separately.
        out_of_range_fraction=float(
            (reference.abs() > 6 * ue8m0_scale(scale).repeat_interleave(32, dim=-1))
            .float()
            .mean()
        ),
    )
    return result


def code_differences(left, right):
    lp, ls = left
    rp, rs = right
    changed = ((lp & 15) != (rp & 15)).sum() + ((lp >> 4) != (rp >> 4)).sum()
    return {
        "fp4_code_changed_fraction": float(changed) / (lp.numel() * 2),
        "scale_changed_fraction": float((ls != rs).float().mean()),
    }


def compare_sample(sample, device, fp4_dtype):
    if not torch.isfinite(sample).all():
        raise ValueError("Checkpoint sample contains NaN/Inf")
    device_weight = sample.to(device)
    measured, codes = {}, {}
    for policy in POLICIES:
        packed, scale, layout = quantize(device_weight, fp4_dtype, policy)
        measured[policy] = {**weight_metrics(sample, packed, scale), **layout}
        codes[policy] = (packed, scale)
    changes = {
        f"{right}_vs_{left}": code_differences(codes[left], codes[right])
        for left, right in (("A", "B"), ("A", "C"), ("B", "C"))
    }
    return {"policies": measured, "differences": changes}


def summarize(results):
    totals = {}
    for policy in POLICIES:
        entries = [r["policies"][policy] for r in results]
        error = sum(e["squared_error"] for e in entries)
        energy = sum(e["energy"] for e in entries)
        elements = sum(e["elements"] for e in entries)
        totals[policy] = {
            "pooled_mse": error / elements,
            "pooled_nmse": error / energy if energy else None,
            "pooled_relative_l2": math.sqrt(error / energy) if energy else None,
        }
    comparisons = {}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        counts = {"lower_error": 0, "equal_error": 0, "higher_error": 0}
        for result in results:
            a = result["policies"][left]["squared_error"]
            b = result["policies"][right]["squared_error"]
            key = (
                "equal_error" if a == b else "lower_error" if b < a else "higher_error"
            )
            counts[key] += 1
        comparisons[f"{right}_vs_{left}"] = counts
    return {"samples": len(results), "policies": totals, "comparisons": comparisons}


def record(output, event, **values):
    output.write(json.dumps({"event": event, **values}, allow_nan=False) + "\n")
    output.flush()


def run(args, output, mapping, names):
    import torch_npu
    from safetensors import safe_open

    if not torch.npu.is_available():
        raise RuntimeError("No NPU available; run in the serving Python environment")
    torch.npu.set_device(args.device)
    fp4_dtype = torch_npu.float4_e2m1fn_x2
    record(
        output,
        "environment",
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        device=torch.npu.get_device_name(args.device),
        logical_device=args.device,
        fp4_dtype=str(fp4_dtype),
        script_head=git_head(Path(__file__).resolve()),
        operator_schema=str(torch.ops.npu.npu_dynamic_mx_quant.default._schema),
        env={
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "CANN_VERSION",
            )
        },
    )
    results = []
    print("Relative L2 (lower is better); same source weights for A/B/C", flush=True)
    for name in names:
        with safe_open(
            str(args.model_path / mapping[name]), framework="pt", device="cpu"
        ) as handle:
            for sample, metadata in read_samples(handle, name, args.rows, args.experts):
                label = name + (
                    f" [expert={metadata['expert']}]"
                    if metadata["expert"] is not None
                    else ""
                )
                print(f"{label} {metadata['sample_shape']}", flush=True)
                record(output, "sample_start", **metadata)
                result = compare_sample(sample, f"npu:{args.device}", fp4_dtype)
                record(output, "sample", **metadata, **result)
                results.append(result)
                print(
                    "  "
                    + "  ".join(
                        f"{p}={result['policies'][p]['relative_l2']:.6g}"
                        for p in POLICIES
                    ),
                    flush=True,
                )
    summary = summarize(results)
    record(
        output,
        "summary",
        **summary,
        scope="sampled weight reconstruction only; no model/kernel accuracy claim",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {args.output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Original BF16/FP16 safetensors directory",
    )
    parser.add_argument(
        "--tensor",
        action="append",
        help="Exact checkpoint key; repeat to select multiple tensors",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of evenly spaced text layers (automatic selection)",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=2,
        help="Number of evenly spaced experts per selected layer/bank",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=128,
        help="Output rows sampled per matrix; 0 for all (full K retained)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Logical NPU index within ASCEND_RT_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected checkpoint keys without importing torch_npu",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/mxfp4-policy/results.jsonl")
    )
    args = parser.parse_args()
    if args.layers <= 0 or args.experts <= 0 or args.rows < 0 or args.device < 0:
        parser.error("layers/experts must be positive; rows/device must be nonnegative")
    mapping = checkpoint_index(args.model_path)
    names = select_names(mapping, args.tensor, args.layers, args.experts)
    if args.list:
        for name in names:
            print(name)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        record(
            output,
            "config",
            model_path=str(args.model_path.resolve()),
            names=names,
            layers=args.layers,
            experts=args.experts,
            rows=args.rows,
            policies=POLICIES,
            omitted_scale_settings="A/B use operator defaults; inspect environment.operator_schema",
        )
        try:
            run(args, output, mapping, names)
        except Exception as exc:
            record(output, "error", error=f"{type(exc).__name__}: {exc}")
            raise
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- concise operator/environment diagnostics
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
