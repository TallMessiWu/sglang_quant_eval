# PR 标题

```
:construction: [diffusion][npu][quant] Add MXFP4 quantization support for Wan2.2 Diffusion on Ascend NPU
```

---

# PR 正文（复制以下内容）

# Summary

This PR adds MXFP4 (Microscaling FP4, dual-level) quantization support for Wan2.2 diffusion models on Ascend NPU. It is a follow-up to #20922 (MXFP8 support) and **must be merged after that PR lands**.

**Hardware requirement:** Ascend A5 series or newer. `npu_dynamic_dual_level_mx_quant` and `npu_dual_level_quant_matmul` are not available on A2/A3.

Two modes are supported:

**Online quantization (`--quantization mxfp4`)**

- Adds `MXFP4Config` + `NPUMXFP4DiffusionLinearMethod` (`multimodal_gen/runtime/layers/quantization/mxfp4_npu.py`) for the diffusion subsystem.
- At load time, FP16/BF16 weights are quantized online to MXFP4 via `npu_dynamic_dual_level_mx_quant`; at inference, activations are quantized per-token and the matmul is executed by `npu_dual_level_quant_matmul` with dual-level block scales (L0 block size = 512, L1 block size = 32).

> **Note:** The online weight quantization path (`npu_dynamic_dual_level_mx_quant` applied to weights) is experimental. MindIE-SD only uses an offline (pre-calibrated) path for MXFP4 weights. The online path quantizes FP16/BF16 weights at load time without calibration, which may produce different numerical results than the offline path.

**Offline quantization (msmodelslim pre-quantized weights)**

- Adds `ModelSlimMXFP4Scheme` (`multimodal_gen/runtime/layers/quantization/modelslim_mxfp4_scheme.py`) for loading weights pre-quantized by msmodelslim.
- Checkpoint tensor formats:
  - `weight`: `[out, in]` — `float8_e4m3fn` container for FP4 data (converted to `float4_e2m1fn_x2` + FRACTAL_NZ at load time)
  - `weight_scale`: `[out, in/32]` — `uint8` L1 block scales (e8m0 + 127 bias), reshaped to `[out, in/64, 2]`
  - `weight_dual_scale`: `[out, in/512, 1]` — `float32` L0 coarse scales, transposed to `[in/512, out]`
  - `mul_scale`: `[in]` — `float32` smooth-quant activation scale from `NonFusionSmoothQuantWrapper`; must be applied to activations before quantization to preserve numerical alignment with the offline-calibrated weights. Defaults to ones (no-op) if absent.

# Key NPU APIs used

| API | Purpose |
| --- | ------- |
| `torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=None)` | Dual-level MX quantization of activations/weights → `(quant, l0_scale, l1_scale)` |
| `torch_npu.npu_dual_level_quant_matmul(x1, x2, x1l0, x2l0, x1l1, x2l1, ...)` | Dual-level MXFP4 quantized matmul |
| `torch_npu.npu_dtype_cast(weight, torch_npu.float4_e2m1fn_x2)` | Cast fp8-container FP4 weights to packed `float4_e2m1fn_x2` dtype |
| `torch_npu.npu_format_cast(w.view(torch.int8), 29, customize_dtype=torch.int8)` | Convert weight tensor to FRACTAL_NZ format (format 29), required by `npu_dual_level_quant_matmul` |

# Files Changed

**New files**

| File | Change |
| ---- | ------ |
| `multimodal_gen/runtime/layers/quantization/mxfp4_npu.py` | **New** — online MXFP4 (`MXFP4Config` + `NPUMXFP4DiffusionLinearMethod`) for Wan2.2 diffusion |
| `multimodal_gen/runtime/layers/quantization/modelslim_mxfp4_scheme.py` | **New** — offline MXFP4 (`ModelSlimMXFP4Scheme`) for msmodelslim pre-quantized weights |

**Modified — MXFP4 registration & dispatch**

| File | Change |
| ---- | ------ |
| `multimodal_gen/runtime/layers/quantization/__init__.py` | Register `MXFP4Config`; add `"mxfp4"` to `QuantizationMethods` literal |
| `multimodal_gen/runtime/layers/quantization/modelslim.py` | Add `W4A4_MXFP4` / `W4A4_MXFP4_DUALSCALE` branch → `ModelSlimMXFP4Scheme` in `_get_scheme_from_parts()`; improve `NotImplementedError` message to include layer name and quant type |

# Implementation Notes

## Dual-Level Scale Layout

MXFP4 uses a two-level block-scale hierarchy:

| Level | Block Size | Tensor | Format in Matmul API |
| ----- | ---------- | ------ | -------------------- |
| L1 (fine) | 32 elements | `weight_scale` | `[out, in/64, 2]` (uint8) |
| L0 (coarse) | 512 elements (= 16 × L1 blocks) | `weight_dual_scale` | `[in/512, out]` (float32) |

The msmodelslim export uses `[out, in/32]` for `weight_scale` and `[out, in/512, 1]` for `weight_dual_scale`. `process_weights_after_loading` reshapes and transposes these to match what `npu_dual_level_quant_matmul` expects, following the MindIE-SD `W4A4MXFP4DualQuantLinear` reference.

## Smooth-Quant `mul_scale`

msmodelslim wraps quantized layers in `NonFusionSmoothQuantWrapper`, which exports a per-channel activation scale `mul_scale` (shape `[in]`). The activation must be multiplied by this scale **before** dual-level quantization to stay aligned with the offline-calibrated weights. Omitting this step causes mosaic / corrupted output.

`mul_scale` is loaded as a `BasevLLMParameter` with `missing_param_init = "ones"` so that models exported without smooth-quant (or repacked without the `.div.` key rename) degrade gracefully to a no-op rather than crashing.

## FRACTAL_NZ Requirement

`npu_dual_level_quant_matmul` requires the weight tensor (`x2`) to be in FRACTAL_NZ memory format (format 29). The conversion is:

```python
weight = torch_npu.npu_dtype_cast(weight_fp8_container, torch_npu.float4_e2m1fn_x2)
weight = torch_npu.npu_format_cast(weight.view(torch.int8), 29, customize_dtype=torch.int8)
```

This matches the `_init_dynamic_quant_param` step in MindIE-SD's `W4A4MXFP4DualQuantLinear`.

# Performance Comparison Report

> **TODO:** Performance numbers are pending. Results will be added before this PR is ready for final review.

Planned benchmark: same script structure as #20922 (baseline BF16, offline msmodelslim, online `--quantization mxfp4`) on Wan2.2-TI2V-5B with 40 denoising steps, 704×1280, 81 frames.

| Metric | Baseline | Offline MXFP4 | Online MXFP4 |
| :----- | :------- | :------------ | :----------- |
| **E2E Latency** | TBD | TBD | TBD |
| **DenoisingStage** | TBD | TBD | TBD |

# Related Issues / PRs

- Closes part of #14424 (MXFP8/MXFP4 support on Ascend NPU for SGLang).
- **Depends on** #20922 (MXFP8 Diffusion support) — this PR must be merged after that one.
