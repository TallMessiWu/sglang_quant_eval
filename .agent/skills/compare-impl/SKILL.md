---
name: compare-impl
description: Compare implementation patterns between SGLang, MindIE-SD, and vllm-ascend for a specific feature (quantization, attention, etc.)
---

# Compare SGLang vs MindIE-SD / vllm-ascend Implementation

When the user wants to compare how a feature is implemented across frameworks:

1. **Find the feature in SGLang**: Search under `sglang/python/sglang/srt/` (for LLM serving) or `sglang/python/sglang/multimodal_gen/` (for diffusion)
2. **Find reference for Diffusion**: Search under `MindIE-SD/mindiesd/`
3. **Find reference for LLM**: Search under `vllm-ascend/vllm_ascend/`
3. **Compare**:
   - Architecture pattern (registry vs layer-replacement, etc.)
   - Core API calls (especially `torch_npu` APIs)
   - Weight format and loading
   - Forward pass logic
   - Configuration mechanism

Key comparison points for quantization:
- SGLang: `QuantizationConfig` → `LinearMethodBase.apply()` pattern
- MindIE-SD: `quantize(model, desc_path)` → layer replacement pattern
- vllm-ascend: Subclasses of `AscendLinearScheme` (e.g., `AscendW8A8MXFP8DynamicLinearMethod`)
- Note differences in weight handling: vllm-ascend often pre-transposes in `process_weights_after_loading` to optimize `apply`.

Output a side-by-side comparison table and highlight what can be directly reused vs what needs adaptation.
