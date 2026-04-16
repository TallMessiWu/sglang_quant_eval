# PR 标题

```
:sparkles: [llm][npu][quant] Add W8A8 MXFP8 quantization support for Qwen3 Dense on Ascend NPU
```

---

# PR 正文（复制以下内容）

# Summary

> **Dependency**: This PR depends on #20922 (Diffusion MXFP8 on Ascend NPU) and should be merged after that PR lands, as both share the same infrastructure additions to `--quantization` CLI and loader paths.

This PR adds W8A8 MXFP8 (Microscaling FP8) quantization support for Qwen3 dense LLM models on Ascend NPU. It closes part of the NPU quantization gap tracked in issue #21584.

**Hardware requirement:** Ascend A5 series or newer. `npu_dynamic_mx_quant` is not available on A2/A3.

Two modes are supported:

**Online quantization (`--quantization mxfp8`)**

- Reuses the existing `Fp8Config` path (triggered by `--quantization mxfp8` with `use_mxfp8=True`).
- Adds an NPU bypass in `Fp8Config.get_min_capability()` (returns 0, skipping CUDA capability checks that are meaningless on NPU).
- Adds an NPU dispatch in `Fp8Config.get_quant_method()` → `NPUMXFP8LinearMethod` (new class in `hardware_backend/npu/quantization/linear_method_npu.py`).
- At load time, FP16/BF16 weights are quantized online to MXFP8 via `npu_dynamic_mx_quant` and pre-transposed to `[in, out]` layout. At inference, activations are quantized per-token and the matmul is executed by `npu_quant_matmul` with `group_sizes=[1,1,32]` (block_size=32).

**Offline quantization (msmodelslim pre-quantized weights)**

- Adds `ModelSlimMXFP8Scheme` (`srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py`) for loading weights pre-quantized by msmodelslim (`float8_e4m3fn` weights + `uint8` scale in `float8_e8m0fnu` encoding).
- Dispatched via `"W8A8_MXFP8"` scheme type in `ModelSlimConfig.get_quant_method()`.

**Bug fixes**

- `rotary_embedding/base.py`: wrap `fused_rope_qk_mqa` import in `try/except`, falling back to `None` if the kernel is absent. Without this, a missing kernel in `sgl_kernel_npu` causes the entire module import to fail, and `ModelRegistry` silently skips the model — falling back to HF Transformers without quantization awareness, producing garbled output with FP8 weights interpreted as BF16.
- `transformers.py`: pass `prefix` to `replace_linear_class` so that `ModelSlimConfig.get_quant_method()` receives the correct layer name for per-layer quantization dispatch.

# Key NPU APIs used

| API                                                                     | Purpose                                            |
| ----------------------------------------------------------------------- | -------------------------------------------------- |
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)` | Dynamic MXFP8 quantization of activations/weights  |
| `torch_npu.npu_quant_matmul(..., group_sizes=[1,1,32])`               | MXFP8 quantized matmul (block_size=32)             |
| `torch_npu.float8_e4m3fn` / `torch_npu.float8_e8m0fnu`              | FP8 weight dtype / scale factor dtype              |

# Files Changed

**New files**

| File                                                                                          | Change                                                                                              |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py`                             | **New** — offline MXFP8 (`ModelSlimMXFP8Scheme`) for msmodelslim pre-quantized weights     |

**Modified — online MXFP8 NPU dispatch**

| File                                                                                          | Change                                                                                              |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `srt/hardware_backend/npu/quantization/linear_method_npu.py`                               | Add `NPUMXFP8LinearMethod` (online MXFP8 weight quantization + inference)                      |
| `srt/layers/quantization/fp8.py`                                                            | NPU bypass for `get_min_capability()`; dispatch to `NPUMXFP8LinearMethod` on NPU + mxfp8 path |

**Modified — offline MXFP8 registration & dispatch**

| File                                                                                          | Change                                                                                              |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `srt/layers/quantization/modelslim/modelslim.py`                                            | Add `W8A8_MXFP8` branch → `ModelSlimMXFP8Scheme` in `get_quant_method()`                     |
| `srt/layers/quantization/modelslim/schemes/__init__.py`                                     | Register `ModelSlimMXFP8Scheme`; fix import order to avoid circular dependency                  |

**Modified — bug fixes**

| File                                                                                          | Change                                                                                              |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `srt/layers/rotary_embedding/base.py`                                                       | Wrap `fused_rope_qk_mqa` import in `try/except` + `None` fallback (prevents model import failure) |
| `srt/models/transformers.py`                                                                | Pass `prefix` to `replace_linear_class` for correct per-layer quant dispatch                   |

# Implementation Notes

## Online vs. Offline: `.contiguous()` behavior on transpose

The two paths differ intentionally in their transpose handling:

- **Online (`NPUMXFP8LinearMethod`)**: calls `.contiguous()` after transpose. This is safe because the quantized weight is freshly allocated by `npu_dynamic_mx_quant` — there are no block-scale mappings tied to the original memory layout.
- **Offline (`ModelSlimMXFP8Scheme`)**: does **not** call `.contiguous()` after transpose, using `.data` assignment to preserve the non-contiguous transpose view. `npu_quant_matmul` reads strides correctly; calling `.contiguous()` would physically reorder the pre-quantized weight data and break the block-scale mapping, producing garbled output.

This pattern is consistent with the approach in `vllm-ascend`.

## Circular import in `schemes/__init__.py`

`modelslim_mxfp8.py` imports `ModelSlimLinearScheme` from `schemes/__init__.py`. To avoid a circular import, `modelslim_scheme` must be imported before `modelslim_mxfp8` in `__init__.py`. An `# isort: off` block enforces this ordering.

# Performance Comparison Report

> **TODO**: Performance numbers are not yet available. This section will be filled in once benchmark runs on Ascend hardware are complete.

| Metric          | Baseline | Offline (ModelSlim MXFP8) | Online (--quantization mxfp8) |
| :-------------- | :------- | :------------------------ | :---------------------------- |
| **E2E Latency** | TBD      | TBD                       | TBD                           |

# Related Issues

Closes part of #21584 (MXFP8/MXFP4 support on Ascend NPU for Qwen3 Dense LLM).

Extends #20922 (Diffusion MXFP8 on Ascend NPU) — shares the same NPU quantization infrastructure pattern.
