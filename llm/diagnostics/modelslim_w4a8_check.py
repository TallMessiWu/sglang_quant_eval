#!/usr/bin/env python3
"""Compare the installed offline W4A8 Linear with a software FP4 reference.

NPU examples (run in the serving Python environment):
  python llm/diagnostics/modelslim_w4a8_check.py
  python llm/diagnostics/modelslim_w4a8_check.py --model-path /path/to/checkpoint
  python llm/diagnostics/modelslim_w4a8_check.py --model-path /path/to/checkpoint \
      --prefix model.language_model.layers.0.mlp.up_proj

The reference uses the SAME dynamically quantized FP8 activations as the NPU
calculation, removing activation quantization error from the comparison. This
checks standalone post-load/apply numerics, not full-model loading, graph replay,
SSM state, checkpoint calibration quality, or end-to-end model accuracy.
Exit codes: 0 = comparisons passed; 1 = numeric mismatch; 2 = unavailable/invalid
environment or checkpoint. --self-test only exercises the CPU reference.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

BLOCK_SIZE = 32
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def emit(label, **values):
    print(json.dumps({"check": label, **values}, ensure_ascii=False), flush=True)


def ue8m0_scale(raw):
    if raw.dtype != torch.uint8:
        raise ValueError(f"Expected raw UE8M0 uint8 bytes, got {raw.dtype}")
    if (raw == 255).any():
        raise ValueError("UE8M0 scale contains NaN encoding 255")
    return torch.exp2(raw.to(torch.float32) - 127)


def unpack_fp4(packed):
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("Packed FP4 must be a uint8 [out, in/2] matrix")
    # ModelSlim pack_fp4_to_uint8 stores the FIRST element in the LOW nibble.
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    magnitudes = torch.tensor(FP4_VALUES, dtype=torch.float32)[codes.long() & 7]
    return torch.where(codes & 8 != 0, -magnitudes, magnitudes)


def dequantize_weight(packed, scales):
    values = unpack_fp4(packed)
    expected = (values.shape[0], values.shape[1] // BLOCK_SIZE)
    if values.shape[1] % BLOCK_SIZE or tuple(scales.shape) != expected:
        raise ValueError(f"Weight scale shape {tuple(scales.shape)} != {expected}")
    return values * ue8m0_scale(scales).repeat_interleave(BLOCK_SIZE, dim=-1)


def dequantize_activation(quantized, scale):
    values = quantized.detach().cpu().float()
    raw = scale.detach().cpu().view(torch.uint8).reshape(values.shape[0], -1)
    if raw.shape[1] * BLOCK_SIZE != values.shape[1]:
        raise ValueError(
            f"Activation scale {tuple(scale.shape)} does not cover {tuple(values.shape)}"
        )
    return values * ue8m0_scale(raw).repeat_interleave(BLOCK_SIZE, dim=-1)


def metrics(actual, reference):
    actual, reference = actual.float(), reference.float()
    if tuple(actual.shape) != tuple(reference.shape):
        raise ValueError(
            f"Output shape {tuple(actual.shape)} != {tuple(reference.shape)}"
        )
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "relative_l2": None, "max_abs": None, "cosine": None}
    diff = actual - reference
    a, b = actual.flatten().double(), reference.flatten().double()
    denominator = b.norm().clamp_min(1e-30)
    cosine_denominator = (a.norm() * b.norm()).clamp_min(1e-30)
    return {
        "finite": True,
        "relative_l2": float(diff.double().norm() / denominator),
        "max_abs": float(diff.abs().max()),
        "cosine": float(torch.dot(a, b) / cosine_denominator),
    }


def self_test():
    # All 16 FP4 codes, including negative zero, with asymmetric nibble pairs.
    packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
    )
    expected = torch.tensor([list(FP4_VALUES) + [-v for v in FP4_VALUES]])
    torch.testing.assert_close(unpack_fp4(packed), expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(unpack_fp4(packed)), torch.signbit(expected))
    torch.testing.assert_close(
        ue8m0_scale(torch.tensor([125, 126, 127, 128, 129], dtype=torch.uint8)),
        torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0]),
        rtol=0,
        atol=0,
    )
    # Different scales on adjacent blocks and rows catch axis/broadcast mistakes.
    weight = packed.repeat(2, 4)
    scale = torch.tensor([[126, 128], [129, 125]], dtype=torch.uint8)
    restored = dequantize_weight(weight, scale)
    torch.testing.assert_close(restored[0, :32], expected.repeat(1, 2)[0] * 0.5)
    torch.testing.assert_close(restored[1, 32:], expected.repeat(1, 2)[0] * 0.25)
    qx = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    restored_x = dequantize_activation(qx, scale.reshape(2, 1, 2))
    torch.testing.assert_close(restored_x[0, :32], torch.full((32,), 0.5))
    torch.testing.assert_close(restored_x[1, 32:], torch.full((32,), 0.25))
    for actual in (restored, restored + 0.125):
        result = metrics(actual, restored)
        assert result["finite"]
        assert (result["relative_l2"] == 0) == torch.equal(actual, restored)
    assert not metrics(torch.full_like(restored, float("nan")), restored)["finite"]
    try:
        ue8m0_scale(torch.tensor([255], dtype=torch.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a rejected NaN scale")
    emit("cpu_reference", passed=True, hardware_validated=False)


def git_head(source):
    for parent in source.parents:
        if (parent / ".git").exists():
            proc = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return proc.stdout.strip() or None
    return None


def checkpoint_tensors(root, prefix, rows):
    from safetensors import safe_open

    index_file = root / "model.safetensors.index.json"
    index = (
        json.loads(index_file.read_text())["weight_map"]
        if index_file.exists()
        else None
    )
    files = sorted(root.glob("*.safetensors"))

    def read(suffix, required=True):
        name = f"{prefix}.{suffix}"
        candidates = (
            [root / index[name]] if index is not None and name in index else files
        )
        for filename in candidates:
            with safe_open(str(filename), framework="pt", device="cpu") as handle:
                if name in handle.keys():  # noqa: SIM118 -- safe_open is not a mapping
                    data = handle.get_slice(name)
                    tensor = data[:rows]
                    emit(
                        "checkpoint_tensor",
                        name=name,
                        shape=data.get_shape(),
                        dtype=str(tensor.dtype),
                    )
                    return tensor
        if required:
            raise ValueError(f"Checkpoint tensor not found: {name}")
        return None

    return read("weight"), read("weight_scale"), read("bias", required=False)


def run_case(label, packed, scales, bias, kernel, device, args):
    reference_weight = dequantize_weight(packed, scales)
    if packed.shape[0] % 32 or reference_weight.shape[1] % 64:
        raise ValueError(
            "Probe requires output rows divisible by 32 and input features divisible by 64"
        )
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(packed.to(device), requires_grad=False),
        weight_scale=torch.nn.Parameter(scales.to(device), requires_grad=False),
    )
    kernel.process_weights_after_loading(layer)
    torch.npu.synchronize()
    bias_npu = bias.to(device) if bias is not None else None
    emit(
        "prepared_layer",
        label=label,
        weight_shape=list(layer.weight.shape),
        weight_dtype=str(layer.weight.dtype),
        weight_stride=list(layer.weight.stride()),
        scale_shape=list(layer.weight_scale.shape),
        scale_dtype=str(layer.weight_scale.dtype),
    )
    passed = True
    for tokens in args.tokens:
        gen = torch.Generator().manual_seed(args.seed + tokens)
        x_cpu = torch.randn(tokens, reference_weight.shape[1], generator=gen).to(
            torch.bfloat16
        )
        x = x_cpu.to(device)
        # This is the actual offline production apply, not a recreated matmul.
        output = kernel.apply(layer, x, bias_npu)
        torch.npu.synchronize()
        quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x,
            dst_type=torch.float8_e4m3fn,
        )
        reference_x = dequantize_activation(quantized_x, dynamic_scale)
        reference = reference_x @ reference_weight.T
        if bias is not None:
            # The production A8W4 bias contract rounds to BF16.
            reference += bias.to(torch.bfloat16).float().reshape(1, -1)
        comparison = metrics(output.detach().cpu(), reference)
        # Repeat identical inputs to detect unstable execution separately.
        repeated = kernel.apply(layer, x, bias_npu)
        torch.npu.synchronize()
        repeat_error = metrics(repeated.detach().cpu(), output.detach().cpu())
        good = (
            comparison["finite"]
            and comparison["relative_l2"] <= args.tolerance
            and repeat_error["finite"]
            and repeat_error["relative_l2"] <= args.tolerance
        )
        passed &= good
        emit(
            "linear_numeric",
            label=label,
            tokens=tokens,
            passed=good,
            tolerance=args.tolerance,
            comparison=comparison,
            repeat_comparison=repeat_error,
            activation_quantization=metrics(reference_x, x_cpu.float()),
        )
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--prefix",
        action="append",
        help="Checkpoint Linear prefix; repeat to check more layers",
    )
    parser.add_argument(
        "--rows", type=int, default=128, help="Load only this many output rows"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 64])
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Logical NPU index in the current visibility mask",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Standalone relative L2 threshold, not model accuracy",
    )
    args = parser.parse_args()
    if (
        args.rows <= 0
        or args.rows % 32
        or any(n <= 0 for n in args.tokens)
        or args.tolerance < 0
    ):
        parser.error(
            "rows must be positive and divisible by 32; tokens positive; tolerance nonnegative"
        )
    if args.prefix and args.model_path is None:
        parser.error("--prefix requires --model-path")
    self_test()
    if args.self_test:
        return 0

    import torch_npu

    # torch_npu must register its operators/dtypes before importing SGLang.
    methods = importlib.import_module(
        "sglang.srt.hardware_backend.npu.quantization.linear_method_npu"
    )

    if not torch.npu.is_available():
        raise RuntimeError("No NPU available")
    source = Path(methods.__file__).resolve()
    versions = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
    try:
        versions["sgl-kernel-npu"] = importlib.metadata.version("sgl-kernel-npu")
    except importlib.metadata.PackageNotFoundError:
        versions["sgl-kernel-npu"] = "no distribution metadata"
    emit(
        "environment",
        versions=versions,
        linear_source=str(source),
        head=git_head(source),
        env={
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_USE_FIA",
                "SGLANG_MAMBA_SSM_DTYPE",
            )
        },
    )
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    kernel = methods.NPUMXFP4W4A8OfflineLinearMethod()

    # Known FP4 codes and nonuniform scales exercise nibble and block ordering.
    gen = torch.Generator().manual_seed(args.seed)
    packed = torch.randint(0, 256, (128, 256), generator=gen, dtype=torch.uint8)
    scales = torch.randint(120, 126, (128, 16), generator=gen, dtype=torch.uint8)
    passed = run_case("synthetic", packed, scales, None, kernel, device, args)

    if args.model_path:
        description = json.loads(
            (args.model_path / "quant_model_description.json").read_text()
        )
        if description.get("group_size", 32) != 32:
            raise ValueError(
                "This probe and the current SGLang W4A8 scheme require group_size=32"
            )
        prefixes = args.prefix or sorted(
            key.removesuffix(".weight")
            for key, value in description.items()
            if value == "W4A8_MXFP"
            and key.endswith(
                (".layers.0.mlp.up_proj.weight", ".layers.0.mlp.down_proj.weight")
            )
        )
        if not prefixes:
            raise ValueError(
                "No layer-0 Dense W4A8 up/down keys found; specify --prefix"
            )
        for prefix in prefixes:
            if description.get(prefix + ".weight") != "W4A8_MXFP":
                raise ValueError(
                    f"{prefix} is not explicitly W4A8_MXFP in the description"
                )
            packed, scales, bias = checkpoint_tensors(
                args.model_path, prefix, args.rows
            )
            passed &= run_case(prefix, packed, scales, bias, kernel, device, args)
    emit(
        "result",
        passed=passed,
        scope="standalone eager Linear; full model and graph untested",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- retain machine-readable probe failures
        emit("unavailable_or_invalid", error=f"{type(exc).__name__}: {exc}")
        sys.exit(2)
