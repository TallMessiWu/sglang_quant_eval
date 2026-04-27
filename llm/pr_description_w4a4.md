# PR 标题

```
:construction: [llm][npu][quant] Add W4A4 MXFP4 quantization support for Qwen3 Dense on Ascend NPU
```

---

# PR 正文（复制以下内容）

# Summary

> **Dependency**: This PR depends on #22352 (W8A8 MXFP8) and #23650 (W4A8 MXFP4) PR, and should be merged after both land. It builds on the same NPU quantization infrastructure (`_NPULinearMethodBase`, `ModelSlimConfig` dispatch, etc.).

This PR adds W4A4 single-level MXFP4 quantization support for Qwen3 dense LLM models on Ascend NPU. It continues the NPU quantization work tracked in issue #21584.

Two modes are supported:

**Online quantization (`--quantization mxfp4_w4a4_npu`)**

- New `NPUMxfp4W4A4Config` (`layers/quantization/npu_mxfp4_w4a4.py`) dispatches to `NPUSingleLevelMXFP4LinearMethod`.
- At load time, FP16/BF16 weights are quantised online to single-level MXFP4 via `npu_dynamic_mx_quant(dst_type=float4_e2m1fn_x2, round_mode="round")`: produces packed `uint8` weights (shape `[out, in//2]`) and FP8_E8M0 per-block scales.
- Weights and scales are pre-transposed in `process_weights_after_loading` (no `.contiguous()` — see Implementation Notes).
- At inference, activations are dynamically quantised with the same single-level API and the matmul is executed by `npu_quant_matmul` with `x1_dtype=x2_dtype=float4_e2m1fn_x2` and `group_sizes=[1, 1, 32]`.

**Offline quantization (msmodelslim pre-quantized weights, `--quantization modelslim`)**

- Adds `ModelSlimMXFP4Scheme` (`modelslim/schemes/modelslim_mxfp4.py`) for the `W4A4_MXFP4` scheme type.
- The msmodelslim checkpoint stores weights as `float8_e4m3fn` (one FP4 value per byte, shape `[out, in]`) and scales as `uint8` E8M0 (shape `[out, in/32]`).
- At load time: weights are cast from `float8_e4m3fn` to `float4_e2m1fn_x2` (packing 2 FP4 values per byte, shape `[out, in//2]`), then transposed to `[in//2, out]`; scales are reshaped `[out, in/32]` → `[out, in/64, 2]` (3D, required by `npu_quant_matmul`) then transposed to `[in/64, out, 2]`.
- At inference, activations are dynamically quantised to MXFP4 via `npu_dynamic_mx_quant` and the matmul runs via `npu_quant_matmul` with `group_sizes=[1,1,32]`.

# Key NPU APIs used

| API | Purpose |
| --- | ------- |
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=float4_e2m1fn_x2, round_mode="round")` | Single-level MXFP4 quantisation of weights (online) and activations (inference) |
| `torch_npu.npu_dtype_cast(weight, float4_e2m1fn_x2)` | Cast fp8-container FP4 weights to packed `float4_e2m1fn_x2` format (offline) |
| `torch_npu.npu_quant_matmul(..., x1_dtype=float4_e2m1fn_x2, x2_dtype=float4_e2m1fn_x2, group_sizes=[1,1,32])` | Single-level MXFP4 quantised matmul |

# Files Changed

**New files**

| File | Change |
| ---- | ------ |
| `srt/layers/quantization/npu_mxfp4_w4a4.py` | **New** — `NPUMxfp4W4A4Config` for online W4A4 MXFP4 (`--quantization mxfp4_w4a4_npu`) |
| `srt/layers/quantization/modelslim/schemes/modelslim_mxfp4.py` | **New** — offline `ModelSlimMXFP4Scheme` for `W4A4_MXFP4` msmodelslim checkpoints |

**Modified**

| File | Change |
| ---- | ------ |
| `srt/hardware_backend/npu/quantization/linear_method_npu.py` | Add `NPUSingleLevelMXFP4LinearMethod` (online single-level MXFP4 weight quantisation + inference) |
| `srt/layers/quantization/__init__.py` | Register `NPUMxfp4W4A4Config` under `"mxfp4_w4a4_npu"` |
| `srt/layers/quantization/modelslim/modelslim.py` | Add `W4A4_MXFP4` branch → `ModelSlimMXFP4Scheme` in `_get_scheme_from_parts()` |
| `srt/layers/quantization/modelslim/schemes/__init__.py` | Export `ModelSlimMXFP4Scheme` |

# Implementation Notes

## Offline checkpoint weight format

The msmodelslim `W4A4_MXFP4` checkpoint stores weights as `float8_e4m3fn` (one FP4 value per byte, using the fp8 dtype as a container) rather than packed `uint8`. In `process_weights_after_loading`, `npu_dtype_cast(..., float4_e2m1fn_x2)` re-packs them into 2-per-byte format, halving the last dimension (`[out, in]` → `[out, in//2]`).

## Weight scale must be 3D for `npu_quant_matmul` with `float4_e2m1fn_x2`

Unlike the MXFP8 path where `x2Scale` is 2D, `npu_quant_matmul` with `x2_dtype=float4_e2m1fn_x2` requires `x2Scale` to be 3D. The scale is reshaped from `[out, in/32]` to `[out, in/64, 2]` (pairing consecutive E8M0 values) before transposing to `[in/64, out, 2]`.

## `.contiguous()` is not called after transpose

Consistent with the W8A8 and W4A8 PRs:
- **Online** (`NPUSingleLevelMXFP4LinearMethod`): transpose is applied via `.data =` in-place assignment without `.contiguous()`. `npu_quant_matmul` reads strides directly; calling `.contiguous()` would physically reorder the quantized data and break block-scale mapping.
- **Offline** (`ModelSlimMXFP4Scheme`): same approach — `.data =` assignment preserves the non-contiguous view.

# Performance Comparison Report

> **TODO**: Performance numbers are not yet available. This section will be filled in once benchmark runs on Ascend hardware are complete.

| Metric | Baseline (BF16) | Offline (ModelSlim W4A4_MXFP4) | Online (--quantization mxfp4_w4a4_npu) |
| :----- | :-------------- | :----------------------------- | :------------------------------------- |
| **E2E Latency** | TBD | TBD | TBD |
| **Memory (NPU)** | TBD | TBD | TBD |

# Related Issues

Closes part of #21584 (MXFP8/MXFP4 support on Ascend NPU for Qwen3 Dense LLM).

Depends on #22352 (W8A8 MXFP8) and #23650 (W4A8 MXFP4) PR.
