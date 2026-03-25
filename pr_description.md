# PR 标题

```
:sparkles: [NPU][MXFP8] Add MXFP8 quantization support for Ascend NPU (LLM + Wan2.2 Diffusion)
```

---

# PR 正文（复制以下内容）

## Summary

This PR adds MXFP8 (Microscaling FP8) quantization support for Ascend NPU, targeting both LLM serving and Wan2.2 TI2V diffusion models. It closes the NPU MXFP8 gap tracked in issue #14424.

Two modes are supported:

**Online quantization (Path B — `--quantization mxfp8`)**

- Adds `NPUMXFP8LinearMethod` (`srt/hardware_backend/npu/quantization/mxfp8_method_npu.py`) which handles NPU-specific weight processing and inference.
- Integrates with the existing `MXFP8LinearAscendMethod` in `fp8.py` via a clean config / NPU-layer split, keeping CUDA and NPU paths independent.
- At load time, FP16/BF16 weights are quantized online to MXFP8 via `npu_dynamic_mx_quant`; at inference, activations are quantized per-token and the matmul is executed by `npu_quant_matmul` with `group_sizes=[1,1,32]` (block_size=32).

**Offline quantization (Path A — msmodelslim pre-quantized weights)**

- Adds `ModelSlimMXFP8` scheme (`srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py`) for loading weights pre-quantized by msmodelslim (`float8_e4m3fn` weights + `uint8` scale in `float8_e8m0fnu` encoding).
- Registered in `schemes/__init__.py` and dispatched from `modelslim.py` via the `W8A8_MXFP8` quant type.

**Wan2.2 TI2V Diffusion models**

- Adds `MXFP8Config` + `NPUMXFP8DiffusionLinearMethod` (`multimodal_gen/runtime/layers/quantization/mxfp8_npu.py`) mirroring the LLM-side implementation for the diffusion subsystem.
- Adds `ModelSlimMXFP8Scheme` (`multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py`) for offline (pre-quantized) diffusion weight loading.
- Refactors `multimodal_gen/tools/wan_repack.py` into a one-step repack tool: copies the original HF Diffusers model, converts msmodelslim quant weights (renaming keys to Diffusers format), and restores `config.json` — replacing a multi-step manual workflow. Fixes multiple bugs in the original script (glob patterns passed as literal paths, unconditional `quant_config` key update causing `KeyError`). Supports `Wan2.2-TI2V-5B` (single transformer) and `Wan2.2-T2V-A14B` / `Wan2.2-I2V-A14B` (Cascade dual-transformer).

**Refactor**

- Splits the previous monolithic linear method into a config layer (`MXFP8LinearAscendMethod` in `fp8.py`) and an NPU execution layer (`NPUMXFP8LinearMethod`), improving maintainability.

## Key NPU APIs used

| API                                                                     | Purpose                                           |
| ----------------------------------------------------------------------- | ------------------------------------------------- |
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)` | Dynamic MXFP8 quantization of activations/weights |
| `torch_npu.npu_quant_matmul(..., group_sizes=[1,1,32])`               | MXFP8 quantized matmul (block_size=32)            |
| `torch_npu.npu_dtype_cast(x, torch_npu.float8_e4m3fn)`                | Cast int8 → float8_e4m3fn for serialized weights |
| `torch_npu.float8_e4m3fn` / `torch_npu.float8_e8m0fnu`              | FP8 weight dtype / scale factor dtype             |

## Usage

**Online MXFP8 (LLM):**

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --quantization mxfp8 \
    --device npu \
    --attention-backend ascend
```

**Offline MXFP8 (msmodelslim pre-quantized, LLM):**

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/mxfp8-quantized-model \
    --quantization modelslim \
    --device npu
```

## Files Changed

| File                                                                     | Change                                                                                           |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `srt/hardware_backend/npu/quantization/mxfp8_method_npu.py`            | **New** — NPU MXFP8 weight processing and kernel dispatch                                 |
| `srt/layers/quantization/fp8.py`                                       | Add NPU branch in `get_quant_method()` for `MXFP8LinearAscendMethod`                         |
| `srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py`         | **New** — offline MXFP8 scheme for msmodelslim weights                                    |
| `srt/layers/quantization/modelslim/schemes/__init__.py`                | Register `ModelSlimMXFP8`                                                                      |
| `srt/layers/quantization/modelslim/modelslim.py`                       | Dispatch `W8A8_MXFP8` quant type                                                               |
| `multimodal_gen/runtime/layers/quantization/mxfp8_npu.py`              | **New** — online MXFP8 for Wan2.2 diffusion                                               |
| `multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py` | **New** — offline MXFP8 for Wan2.2 diffusion                                              |
| `multimodal_gen/tools/wan_repack.py`                                   | Refactor — one-step repack CLI; fix glob/KeyError bugs; add T2V-A14B, I2V-A14B, TI2V-5B support |
| `test/srt/ascend/test_ascend_mxfp8_quantization.py`                    | **New** — GSM8K accuracy + throughput test on NPU                                         |

## wan_repack.py: Design Details

### Bug Fixes

The original script contained four bugs that made it entirely non-functional:

| # | Location | Bug | Root Cause |
|---|----------|-----|------------|
| 1 | `load_sharded_safetensors()` | `pathlib.Path(dir, "*model*.safetensors")` passed directly to `load_file()` | `pathlib.Path(dir, "*.safetensors")` creates a **literal** path with `*` in the filename — not a glob. `load_file()` does not expand globs, so every run raises `FileNotFoundError`. |
| 2 | `convert_transformer()` | Same pattern applied to `open(pathlib.Path(dir, "*quant_model_description*.json"))` | Same root cause as above. |
| 3 | `get_transformer_config()` | No `else` branch — unknown `model_type` causes `NameError: name 'RENAME_DICT' is not defined` | Local variable only assigned inside `if model_type == "Wan-T2V-14B"`, then referenced unconditionally outside. |
| 4 | `convert_transformer()` | `update_dict_(original_quant_config, key, new_key)` called unconditionally for every key | The quant description JSON only contains entries for quantized Linear layers, not all model keys. Calling `dict.pop()` on a missing key raises `KeyError`. |

**Fix for bugs 1 & 2**: replaced glob-as-literal-path with `directory.glob(pattern)`, which returns a proper file list. Added existence and uniqueness checks with descriptive error messages.

**Fix for bug 3**: added `else: raise ValueError(...)` and extended support to `Wan2.2-I2V-A14B` and `Wan2.2-TI2V-5B`.

**Fix for bug 4**: added `if key in quant_config` guard before updating the quant description dict.

---

### One-Step Repack Workflow

**Before** — users had to run these steps manually:

```bash
# 1. Copy original HF model
cp -r Wan2.2-TI2V-5B-Diffusers Wan2.2-TI2V-5B-Diffusers-MXFP8

# 2. Delete transformer dir(s) from the copy
rm -rf Wan2.2-TI2V-5B-Diffusers-MXFP8/transformer

# 3. Run weight conversion (also broken — see bugs above)
python wan_repack.py \
    --input-path Wan2.2-TI2V-5B-quantized \
    --output-path Wan2.2-TI2V-5B-Diffusers-MXFP8

# 4. Restore config.json that was deleted in step 2
cp Wan2.2-TI2V-5B-Diffusers/transformer/config.json \
   Wan2.2-TI2V-5B-Diffusers-MXFP8/transformer/config.json
```

**After** — single command:

```bash
python wan_repack.py \
    --model-type Wan2.2-TI2V-5B \
    --original-model-path Wan2.2-TI2V-5B-Diffusers \
    --quant-path        Wan2.2-TI2V-5B-quantized \
    --output-path       Wan2.2-TI2V-5B-Diffusers-MXFP8
```

Internally, the new `repack()` orchestrator runs three steps:

1. `shutil.copytree(original, output, ignore=transformer_dirs)` — copies the full model (VAE, text encoder, scheduler, etc.) to the output path, skipping transformer dirs.
2. `convert_transformer()` for each transformer dir — converts quantized weights to `diffusion_pytorch_model.safetensors` and renames keys to HF Diffusers format.
3. `shutil.copy2(original/transformer/config.json, output/transformer/config.json)` — restores the architecture config that was excluded in step 1.

For Cascade models (`Wan2.2-T2V-A14B`, `Wan2.2-I2V-A14B`), steps 2–3 repeat for both `transformer/` (sourced from `quant_path/high_noise_model/`) and `transformer_2/` (sourced from `quant_path/low_noise_model/`). The cascade vs. single-model dispatch is driven by `CASCADE_MODEL_TYPES`.

---

### Summary

| Aspect | Before | After |
|--------|--------|-------|
| Glob file matching | `pathlib.Path(dir, "*.safetensors")` — broken, `FileNotFoundError` on every run | `dir.glob("*.safetensors")` with existence and uniqueness checks |
| `get_transformer_config()` | Only `"Wan-T2V-14B"`; crashes with `NameError` for any other type | Supports `Wan2.2-T2V-A14B`, `Wan2.2-I2V-A14B`, `Wan2.2-TI2V-5B`; raises `ValueError` on unsupported type |
| `quant_config` key update | Unconditional `dict.pop()` — `KeyError` for non-quantized layers | `if key in quant_config` guard |
| Workflow | 4 manual steps (copy model, delete transformer, convert, restore config.json) | Single `repack()` call handles all steps |
| CLI arguments | `--input-path`, `--output-path` only | `--model-type`, `--original-model-path`, `--quant-path`, `--output-path` |

## Related Issues

Closes part of #14424 (MXFP8/MXFP4 support on Ascend NPU for SGLang).
