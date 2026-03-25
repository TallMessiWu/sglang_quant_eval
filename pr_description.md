# PR 标题

```
:sparkles: [NPU][MXFP8] Add MXFP8 quantization support for Wan2.2 Diffusion on Ascend NPU
```

---

# PR 正文（复制以下内容）

# Summary

This PR adds MXFP8 (Microscaling FP8) quantization support for Wan2.2 diffusion models on Ascend NPU. It closes part of the NPU MXFP8 gap tracked in issue #14424.

Two modes are supported:

**Online quantization (`--quantization mxfp8`)**

- Adds `MXFP8Config` + `NPUMXFP8DiffusionLinearMethod` (`multimodal_gen/runtime/layers/quantization/mxfp8_npu.py`) for the diffusion subsystem.
- At load time, FP16/BF16 weights are quantized online to MXFP8 via `npu_dynamic_mx_quant`; at inference, activations are quantized per-token and the matmul is executed by `npu_quant_matmul` with `group_sizes=[1,1,32]` (block_size=32).

**Offline quantization (msmodelslim pre-quantized weights)**

- Adds `ModelSlimMXFP8Scheme` (`multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py`) for loading weights pre-quantized by msmodelslim (`float8_e4m3fn` weights + `uint8` scale in `float8_e8m0fnu` encoding).

**wan_repack.py refactor**

- Refactors `multimodal_gen/tools/wan_repack.py` into a one-step repack tool: copies the original HF Diffusers model, converts msmodelslim quant weights (renaming keys to Diffusers format), and restores `config.json` — replacing a multi-step manual workflow. Fixes multiple bugs in the original script (glob patterns passed as literal paths, unconditional `quant_config` key update causing `KeyError`). Supports `Wan2.2-TI2V-5B` (single transformer) and `Wan2.2-T2V-A14B` / `Wan2.2-I2V-A14B` (Cascade dual-transformer).

# Key NPU APIs used

| API                                                                     | Purpose                                           |
| ----------------------------------------------------------------------- | ------------------------------------------------- |
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)` | Dynamic MXFP8 quantization of activations/weights |
| `torch_npu.npu_quant_matmul(..., group_sizes=[1,1,32])`               | MXFP8 quantized matmul (block_size=32)            |
| `torch_npu.float8_e4m3fn` / `torch_npu.float8_e8m0fnu`              | FP8 weight dtype / scale factor dtype             |

# Files Changed

**New files**

| File                                                                     | Change                                                                                                   |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `multimodal_gen/runtime/layers/quantization/mxfp8_npu.py`              | **New** — online MXFP8 (`MXFP8Config` + `NPUMXFP8DiffusionLinearMethod`) for Wan2.2 diffusion |
| `multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py` | **New** — offline MXFP8 (`ModelSlimMXFP8Scheme`) for msmodelslim pre-quantized weights          |

**Modified — MXFP8 registration & dispatch**

| File                                                        | Change                                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `multimodal_gen/runtime/layers/quantization/__init__.py`  | Register `MXFP8Config`; add `"mxfp8"` to `QuantizationMethods` literal          |
| `multimodal_gen/runtime/layers/quantization/modelslim.py` | Add `W8A8_MXFP8` branch → `ModelSlimMXFP8Scheme` in `_get_scheme_from_parts()` |

**Modified — CLI & loader support**

| File                                                                      | Change                                                                                        |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `multimodal_gen/runtime/server_args.py`                                 | Add `--quantization` CLI arg (explicit method override, e.g. `--quantization mxfp8`)      |
| `multimodal_gen/runtime/loader/component_loaders/transformer_loader.py` | Honor `--quantization` flag; takes priority over auto-detection from config.json / metadata |
| `multimodal_gen/runtime/loader/fsdp_load.py`                            | Add `weight_scale` to FSDP unused-key list (prevents crash on offline MXFP8 weight load)    |
| `multimodal_gen/runtime/utils/quantization_utils.py`                    | Glob fallback for `quant_model_description*.json` (supports repacked filenames)             |

**Modified — tooling**

| File                                   | Change                                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `multimodal_gen/tools/wan_repack.py` | Refactor — one-step repack CLI; fix glob/KeyError bugs; add T2V-A14B, I2V-A14B, TI2V-5B support |

**Modified — minor refactor (srt)**

| File                               | Change                                                                                                                                                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `srt/layers/quantization/fp8.py` | Import cleanup (`apply_fp8_marlin_linear` → `torch.ops.sglang.apply_fp8_marlin_linear`); restructure `Fp8MoEMethod.process_weights_after_loading()` to move weight shuffle inside scale-processing block |

# wan_repack.py: Design Details

## Bug Fixes

The original script contained four bugs that made it entirely non-functional:

| # | Location                       | Bug                                                                                                  | Root Cause                                                                                                                                                                                          |
| - | ------------------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `load_sharded_safetensors()` | `pathlib.Path(dir, "*model*.safetensors")` passed directly to `load_file()`                      | `pathlib.Path(dir, "*.safetensors")` creates a **literal** path with `*` in the filename — not a glob. `load_file()` does not expand globs, so every run raises `FileNotFoundError`. |
| 2 | `convert_transformer()`      | Same pattern applied to `open(pathlib.Path(dir, "*quant_model_description*.json"))`                | Same root cause as above.                                                                                                                                                                           |
| 3 | `get_transformer_config()`   | No `else` branch — unknown `model_type` causes `NameError: name 'RENAME_DICT' is not defined` | Local variable only assigned inside `if model_type == "Wan-T2V-14B"`, then referenced unconditionally outside.                                                                                    |
| 4 | `convert_transformer()`      | `update_dict_(original_quant_config, key, new_key)` called unconditionally for every key           | The quant description JSON only contains entries for quantized Linear layers, not all model keys. Calling `dict.pop()` on a missing key raises `KeyError`.                                      |

**Fix for bugs 1 & 2**: replaced glob-as-literal-path with `directory.glob(pattern)`, which returns a proper file list. Added existence and uniqueness checks with descriptive error messages.

**Fix for bug 3**: added `else: raise ValueError(...)` and extended support to `Wan2.2-I2V-A14B` and `Wan2.2-TI2V-5B`.

**Fix for bug 4**: added `if key in quant_config` guard before updating the quant description dict.

---

## One-Step Repack Workflow

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

## Summary

| Aspect                       | Before                                                                               | After                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Glob file matching           | `pathlib.Path(dir, "*.safetensors")` — broken, `FileNotFoundError` on every run | `dir.glob("*.safetensors")` with existence and uniqueness checks                                               |
| `get_transformer_config()` | Only `"Wan-T2V-14B"`; crashes with `NameError` for any other type                | Supports `Wan2.2-T2V-A14B`, `Wan2.2-I2V-A14B`, `Wan2.2-TI2V-5B`; raises `ValueError` on unsupported type |
| `quant_config` key update  | Unconditional `dict.pop()` — `KeyError` for non-quantized layers                | `if key in quant_config` guard                                                                                 |
| Workflow                     | 4 manual steps (copy model, delete transformer, convert, restore config.json)        | Single `repack()` call handles all steps                                                                       |
| CLI arguments                | `--input-path`, `--output-path` only                                             | `--model-type`, `--original-model-path`, `--quant-path`, `--output-path`                                 |

# Performance Comparison Report

## 1. Scripts

```bash
# Base Model
SGLANG_CACHE_DIT_FN=2
SGLANG_CACHE_DIT_BN=1
SGLANG_CACHE_DIT_WARMUP=4
SGLANG_CACHE_DIT_RDT=0.4 
SGLANG_CACHE_DIT_MC=4 
SGLANG_CACHE_DIT_TAYLORSEER=true 
SGLANG_CACHE_DIT_TS_ORDER=2 
SGLANG_CACHE_DIT_ENABLED=true
sglang generate --model-path /home/weights/Wan2.2-TI2V-5B-Diffusers \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--height 704 --width 1280 --num-gpus 1 --num-frames 81 --num-inference-steps 40  --warmup \
--perf-dump-path baseline.json

sleep 15

# Inference using Offline Modelslim Pre-Quantized Weights
SGLANG_CACHE_DIT_FN=2
SGLANG_CACHE_DIT_BN=1
SGLANG_CACHE_DIT_WARMUP=4
SGLANG_CACHE_DIT_RDT=0.4 
SGLANG_CACHE_DIT_MC=4 
SGLANG_CACHE_DIT_TAYLORSEER=true 
SGLANG_CACHE_DIT_TS_ORDER=2 
SGLANG_CACHE_DIT_ENABLED=true
sglang generate --model-path /home/weights/Wan2.2-TI2V-5B-Diffusers-mxfp8 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--height 704 --width 1280 --num-gpus 1 --num-frames 81 --num-inference-steps 40  --warmup \
--perf-dump-path offline.json

sleep 15

# Online Quantization Inference
SGLANG_CACHE_DIT_FN=2
SGLANG_CACHE_DIT_BN=1
SGLANG_CACHE_DIT_WARMUP=4
SGLANG_CACHE_DIT_RDT=0.4
SGLANG_CACHE_DIT_MC=4
SGLANG_CACHE_DIT_TAYLORSEER=true
SGLANG_CACHE_DIT_TS_ORDER=2
SGLANG_CACHE_DIT_ENABLED=true
sglang generate --model-path /home/weights/Wan2.2-TI2V-5B-Diffusers \
--quantization mxfp8 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--height 704 --width 1280 --num-gpus 1 --num-frames 81 --num-inference-steps 40  --warmup \
--perf-dump-path online.json
```

## 2. High-level Summary

| Metric                | Baseline     | offline.json            | online.json             |
| :-------------------- | :----------- | :---------------------- | :---------------------- |
| **E2E Latency** | 102922.86 ms | 91619.75 ms (-11.0%) ✅ | 91413.05 ms (-11.2%) ✅ |

## 3. Stage Breakdown

| Stage Name               | Baseline | offline.json          | online.json           |
| :----------------------- | :------- | :-------------------- | :-------------------- |
| InputValidationStage     | 0.11     | 0.08 (-30.0%) ⚪️    | 0.06 (-49.2%) ⚪️    |
| TextEncodingStage        | 12573.16 | 12653.25 (+0.6%) ⚪️ | 12476.38 (-0.8%) ⚪️ |
| LatentPreparationStage   | 0.27     | 0.21 (-24.7%) ⚪️    | 0.16 (-42.3%) ⚪️    |
| TimestepPreparationStage | 1.21     | 0.95 (-21.0%) ⚪️    | 0.90 (-25.5%) ⚪️    |
| DenoisingStage           | 83386.55 | 72910.67 (-12.6%) 🟢  | 72868.55 (-12.6%) 🟢  |
| DecodingStage            | 6952.78  | 6046.59 (-13.0%) 🟢   | 6058.86 (-12.9%) 🟢   |

<details>
<summary>Metadata</summary>

- Baseline Commit: `1b273c6b50fac55790e654ad2888acd0c9f7d7b5`
- offline.json Commit: `1b273c6b50fac55790e654ad2888acd0c9f7d7b5`
- online.json Commit: `1b273c6b50fac55790e654ad2888acd0c9f7d7b5`
- Timestamp: 2026-03-25T16:09:56.257827

</details>

# Related Issues

Closes part of #14424 (MXFP8/MXFP4 support on Ascend NPU for SGLang).
