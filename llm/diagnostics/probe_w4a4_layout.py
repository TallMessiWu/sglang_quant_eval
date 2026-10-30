#!/usr/bin/env python3
"""Inspect one W4A4 layout on an idle Ascend 950; run from llm/.

Each invocation tests one cast policy in a fresh process, using the production
W4A4 preparation/GMM wrappers. This is a kernel probe, not model accuracy or
startup-memory validation. The offline case uses synthetic packed uint8 weights,
not a ModelSlim checkpoint. No production layout policy is changed by this tool.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


def run(args, report):
    import torch

    from sglang.srt.utils import is_npu

    if not is_npu():
        raise RuntimeError("This probe requires an installed Ascend NPU environment")
    import torch_npu

    # Initialize the registry first: a cold moe_methods import cycles through
    # quantization -> awq -> unquant -> the partially initialized moe_methods.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.hardware_backend.npu.quantization import moe_methods

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(20261030)
    device = f"npu:{args.device}"
    fp4 = moe_methods._get_float4_e2m1fn_x2_dtype()
    report["environment"] = {
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "device": torch.npu.get_device_name(args.device),
        "implementation": moe_methods.__file__,
    }
    h, i, m = args.hidden_size, args.intermediate_size, args.tokens_per_expert
    weights = {
        "w13": (torch.randn(2, 2 * i, h) / math.sqrt(h)).bfloat16().to(device),
        "w2": (torch.randn(2, h, i) / math.sqrt(i)).bfloat16().to(device),
    }
    x = torch.randn(2 * m, h).bfloat16().to(device)
    counts = torch.tensor([m, m], dtype=torch.int64, device=device)
    layer = torch.nn.Module()
    w13 = moe_methods.NPUW4A4MXFP4MoEMethod("w13", use_fused_gmm1=True)
    w2 = moe_methods.NPUW4A4MXFP4MoEMethod("w2")
    original_cast = moe_methods.npu_format_cast

    def cast(weight):
        if args.layout == "current":
            return original_cast(weight)
        if args.layout == "nd":
            return weight
        storage_dtype = torch.float8_e4m3fn if args.layout == "nz-e4m3" else torch.int8
        return torch.ops.npu.npu_format_cast(
            weight, 29, input_dtype=fp4, customize_dtype=storage_dtype
        )

    def describe(tensor):
        return {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "format": torch_npu.get_npu_format(tensor),
        }

    report["prepared"] = {}
    for prefix, kernel in (("w13", w13), ("w2", w2)):
        weight = weights[prefix].clone()
        if args.source == "offline":
            weight, scale = kernel._quantize_weight_online(weight, fp4)
            weight = weight.view(torch.uint8)
            if scale.dim() == 4:
                scale = scale.flatten(-2)
            layer.register_parameter(
                f"{prefix}_weight_scale",
                torch.nn.Parameter(scale.view(torch.uint8), requires_grad=False),
            )
        layer.register_parameter(
            f"{prefix}_weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        # Only substitute the cast policy; all quantization and GMM ops are real.
        with patch.object(moe_methods, "npu_format_cast", side_effect=cast):
            kernel.process_weights_after_loading(layer, prefix)
        torch.npu.synchronize()
        report["prepared"][prefix] = {
            "weight": describe(getattr(layer, f"{prefix}_weight")),
            "scale": describe(getattr(layer, f"{prefix}_weight_scale")),
        }

    reference = []
    for expert in range(2):
        gate, up = torch.nn.functional.linear(
            x[expert * m : (expert + 1) * m].float(), weights["w13"][expert].float()
        ).chunk(2, dim=-1)
        reference.append(
            torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up, weights["w2"][expert].float()
            )
        )
    reference = torch.cat(reference).cpu()

    def error(actual, expected):
        actual, expected = actual.float().cpu(), expected.float().cpu()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        if not finite:
            return {"finite": False}
        delta = actual - expected
        return {
            "finite": True,
            "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-12)),
            "max_abs": float(delta.abs().max()),
        }

    plain_hidden = w13.apply(layer, x, counts, None, torch.bfloat16, "w13", 1)
    plain_hidden = torch.ops.npu.npu_swiglu(plain_hidden)
    plain = w2.apply(layer, plain_hidden, counts, None, torch.bfloat16, "w2", 1)
    torch.npu.synchronize()
    report["unfused_vs_float_reference"] = error(plain, reference)
    fused_hidden, fused_scale = w13.apply_fused_gmm1_swiglu(layer, x, counts, None, 1)
    fused = w2.apply(layer, fused_hidden, counts, fused_scale, torch.bfloat16, "w2", 1)
    torch.npu.synchronize()
    report["fused_output"] = {
        "activation": describe(fused_hidden),
        "scale": describe(fused_scale),
    }
    report["fused_vs_float_reference"] = error(fused, reference)
    report["fused_vs_unfused"] = error(fused, plain)
    if not all(
        report[key]["finite"]
        for key in (
            "unfused_vs_float_reference",
            "fused_vs_float_reference",
            "fused_vs_unfused",
        )
    ):
        raise RuntimeError("Non-finite output; inspect the recorded tensor layouts")
    report["status"] = (
        "executed; numerical errors require review, not a model-accuracy pass"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--layout", choices=("current", "nd", "nz-e4m3", "nz-int8"), default="current"
    )
    parser.add_argument("--source", choices=("online", "offline"), default="online")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--tokens-per-expert", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(k <= 0 or k % 64 for k in (args.hidden_size, args.intermediate_size)):
        parser.error("hidden/intermediate sizes must be positive multiples of 64")
    if args.tokens_per_expert <= 0:
        parser.error("tokens-per-expert must be positive")
    output = args.output or Path("logs") / (
        f"w4a4-layout-{args.source}-{args.layout}-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    report = {
        "case": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    }
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"W4A4 layout report: {output}")


if __name__ == "__main__":
    main()
