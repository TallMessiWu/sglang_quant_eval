# PR 标题

```
:sparkles: [llm][npu][quant] Add W4A8 MXFP quantization support for Qwen3 Dense on Ascend NPU
```

---

# PR 正文（复制以下内容）

# Summary

> **Dependency**: This PR depends on #22352 (W8A8 MXFP8 for Qwen3 Dense on Ascend NPU) and should be merged after that PR lands, as it builds on the same NPU quantization infrastructure (`_NPULinearMethodBase`, `ModelSlimConfig` dispatch, etc.).

This PR adds W4A8 MXFP quantization support for Qwen3 dense LLM models on Ascend NPU. It continues the NPU quantization work tracked in issue #21584.

**Hardware requirement:** Ascend 950 (Atlas A5)

Two modes are supported:

**Online quantization (`--quantization mxfp4_w4a8_npu`)**

- New `NPUMxfp4Config` (`layers/quantization/npu_mxfp4.py`) dispatches to `NPUMXFP4W4A8LinearMethod`.
- At load time, FP16/BF16 weights are quantised online to dual-level MXFP4 via `npu_dynamic_dual_level_mx_quant`: produces `float4_e2m1fn_x2` weights, L0 per-block scale (FP32), and L1 per-channel scale (FP8_E8M0). The weight is then cast to FRACTAL_NZ format for NPU matmul efficiency.
- At inference, activations are quantised with the same dual-level API and the matmul is executed by `npu_dual_level_quant_matmul`.
- Note: the actual matmul compute is W4A4 (both operands in FP4 with dual-level scales); "A8" refers to the FP8_E8M0 L1 scale format. There is no W4A8 mixed-precision public kernel in the current torch_npu API.

**Offline quantization (msmodelslim pre-quantized weights, `--quantization modelslim`)**

- Adds `ModelSlimMXFP4W4A8Scheme` (`modelslim/schemes/modelslim_mxfp4_w4a8.py`) for the `W4A8_MXFP` scheme type.
- The current msmodelslim checkpoint format for `W4A8_MXFP` stores weights as `float8_e4m3fn` (shape `[out, in]`) with a `uint8` FP8_E8M0-biased scale (shape `[out, in/32]`). At load time, scale is reshaped to `[out, in/64, 2]` and both weight and scale are transposed (no `.contiguous()` — see Implementation Notes).
- At inference, activations are dynamically quantised to FP8 via `npu_dynamic_mx_quant` and the matmul runs via `npu_quant_matmul` with `group_sizes=[1,1,32]` — identical to the MXFP8 offline path.
- Also adds `W4A8_DYNAMIC` dispatch → `ModelSlimW4A8Int8` (INT4 offline scheme for non-MXFP checkpoints).

# Key NPU APIs used

| API | Purpose |
| --- | ------- |
| `torch_npu.npu_dynamic_dual_level_mx_quant(x)` | Dual-level MXFP4 quantisation of activations/weights (FP4 + L0 FP32 scale + L1 FP8_E8M0 scale) |
| `torch_npu.npu_dual_level_quant_matmul(...)` | MXFP4 dual-level quantised matmul (online mode, Ascend 950 only) |
| `torch_npu.npu_format_cast(w.view(torch.int8), 29)` | Convert FP4 weight to FRACTAL_NZ layout (required by dual-level matmul) |
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=float8_e4m3fn)` | Dynamic MXFP8 activation quantisation (offline mode) |
| `torch_npu.npu_quant_matmul(..., group_sizes=[1,1,32])` | MXFP8 quantised matmul (offline mode, block_size=32) |

# Files Changed

**New files**

| File | Change |
| ---- | ------ |
| `srt/layers/quantization/npu_mxfp4.py` | **New** — `NPUMxfp4Config` for online W4A8 MXFP4 (`--quantization mxfp4_w4a8_npu`) |
| `srt/layers/quantization/modelslim/schemes/modelslim_mxfp4_w4a8.py` | **New** — offline `ModelSlimMXFP4W4A8Scheme` for `W4A8_MXFP` msmodelslim checkpoints |

**Modified — online W4A8 NPU dispatch**

| File | Change |
| ---- | ------ |
| `srt/hardware_backend/npu/quantization/linear_method_npu.py` | Add `NPUMXFP4W4A8LinearMethod` (online dual-level MXFP4 weight quantisation + inference) and `NPUW4A8DynamicLinearMethod` (offline INT4 inference via `npu_weight_quant_batchmatmul`) |
| `srt/layers/quantization/__init__.py` | Register `NPUMxfp4Config` under `"mxfp4_w4a8_npu"` |
| `srt/server_args.py` | Add `"mxfp4_w4a8_npu"` and `"mxfp4_w4a4_npu"` to `QUANTIZATION_CHOICES` |

**Modified — offline W4A8 registration & dispatch**

| File | Change |
| ---- | ------ |
| `srt/layers/quantization/modelslim/modelslim.py` | Add `W4A8_MXFP` branch → `ModelSlimMXFP4W4A8Scheme` and `W4A8_DYNAMIC` branch → `ModelSlimW4A8Int8` in `_get_scheme_from_parts()` |
| `srt/layers/quantization/modelslim/schemes/__init__.py` | Register `ModelSlimMXFP4W4A8Scheme` and `ModelSlimW4A8Int8` |

# Implementation Notes

## W4A8_MXFP offline: weight format matches MXFP8

The current msmodelslim `W4A8_MXFP` checkpoint stores weights as `float8_e4m3fn` (identical layout to `W8A8_MXFP8`), **not** as packed FP4 uint8. This corresponds to an older msmodelslim export version. Consequently, `ModelSlimMXFP4W4A8Scheme` is structurally identical to `ModelSlimMXFP8Scheme` — the distinction lies in the quantisation process, not the inference path. A future checkpoint format change (packed FP4) would require a separate scheme.

## Online W4A8: dual-level scale layout requirements

`npu_dual_level_quant_matmul` requires:
- Weight in FRACTAL_NZ format (format=29), cast via `npu_format_cast(w.view(torch.int8), 29)` since only int-dtype tensors are accepted.
- L0 weight scale shape `[in/512, out]`: loaded as `[out, in/512, 1]`, squeezed and transposed with `.contiguous()`. The `.contiguous()` here is safe — the scale is freshly allocated, unlike the offline non-contiguous transpose pattern.

## `.contiguous()` asymmetry between online and offline paths

Consistent with the W8A8 PR (#22352):
- **Online** (`NPUMXFP4W4A8LinearMethod`): uses `.contiguous()` after transpose — safe because weights/scales are freshly allocated from `npu_dynamic_dual_level_mx_quant`.
- **Offline** (`ModelSlimMXFP4W4A8Scheme`): does **not** call `.contiguous()`, using `.data` assignment to preserve the non-contiguous transpose view. `npu_quant_matmul` reads strides correctly; `.contiguous()` would physically reorder pre-quantized data and break block-scale mapping.

# Performance Comparison Report

> **TODO**: Performance numbers are not yet available. This section will be filled in once benchmark runs on Ascend hardware are complete.

| Metric | Baseline (BF16) | Offline (ModelSlim W4A8_MXFP) | Online (--quantization mxfp4_w4a8_npu) |
| :----- | :-------------- | :---------------------------- | :------------------------------------- |
| **E2E Latency** | TBD | TBD | TBD |
| **Memory (GPU)** | TBD | TBD | TBD |

# Related Issues

Closes part of #21584 (MXFP8/MXFP4 support on Ascend NPU for Qwen3 Dense LLM).

Depends on #22352 (W8A8 MXFP8 for Qwen3 Dense on Ascend NPU).
