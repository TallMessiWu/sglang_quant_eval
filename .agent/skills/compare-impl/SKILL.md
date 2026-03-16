---
name: compare-impl
description: Compare implementation patterns between SGLang and MindIE-SD for a specific feature (quantization, attention, etc.)
user_invocable: true
---

# Compare SGLang vs MindIE-SD Implementation

When the user wants to compare how a feature is implemented in both frameworks:

1. **Find the feature in SGLang**: Search under `sglang/python/sglang/srt/` (for LLM serving) or `sglang/python/sglang/multimodal_gen/` (for diffusion)
2. **Find the feature in MindIE-SD**: Search under `MindIE-SD/mindiesd/`
3. **Compare**:
   - Architecture pattern (registry vs layer-replacement, etc.)
   - Core API calls (especially `torch_npu` APIs)
   - Weight format and loading
   - Forward pass logic
   - Configuration mechanism

Key comparison points for quantization:
- SGLang: `QuantizationConfig` → `LinearMethodBase.apply()` pattern
- MindIE-SD: `quantize(model, desc_path)` → layer replacement pattern
- Both use the same `torch_npu` APIs for NPU compute

Output a side-by-side comparison table and highlight what can be directly reused vs what needs adaptation.
