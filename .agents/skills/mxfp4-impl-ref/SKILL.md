---
name: mxfp4-impl-ref
description: Review or implement Ascend MXFP4 paths with format-aware checks for online/offline W4A8 and W4A4, packed weights, single/dual-level scales and NPU operator contracts.
---

# MXFP4 implementation checklist

Resolve repository root from this directory (`../../..`). Read `docs/quantization-overview.md`, relevant `docs/known-pitfalls.md` sections, and select the real worktree from `docs/branches.md` plus `git worktree list`. Follow the `sglang-quant-lookup` workflow for source tracing; this skill is not a version-independent API recipe.

- Identify **W4A8 vs W4A4**, **Linear vs MoE**, **online vs ModelSlim**, and **single-level vs dual-level** before choosing an operator. Dense online dual-level FP4 is not the MoE single-level format.
- Offline MXFP4 is packed `uint8 [out,in/2]`; preserve loader metadata and do not requantize/repack. Online preparation must account for TP/EP shard completion, source storage release and reload.
- Weight policy and activation policy are separate. Verify actual `round_mode`, `scale_alg`, `dst_type_max` in the selected branch rather than propagating one policy to both.
- Check payload/scale dtype, enum-vs-torch dtype, ND/NZ, transpose views, group-list COUNT/CUMSUM, output dtype and bias against the installed API. Do not add `.contiguous()` or force NZ as a universal fix.
- Preserve scale axes: dual-level L0 needs explicit shape, not an unconditional `squeeze(-1)`; single-level pair packing and odd-block padding are separate. For dual-level background read `docs/npu-api/`, whose archived excerpts are not current-version guarantees.
- Verify unaligned Linear fallback, fused gate/up precision, router exclusions, DeepEP normal/low-latency and source/output dtype. Use K≠V / singleton dimensions where they expose layout mistakes.
- Guard NPU-only imports and defer op lookup so CPU contract tests can import/construct classes.

Prefer the existing NPU methods and vllm-ascend semantics. Use focused regression tests for the changed contract, then separately report any actual NPU/graph/model evidence. No new generic framework or model-name-specific quantization class without a demonstrated need.
