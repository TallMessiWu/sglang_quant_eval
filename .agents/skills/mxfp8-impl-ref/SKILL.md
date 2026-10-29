---
name: mxfp8-impl-ref
description: Review or implement Ascend MXFP8 online/offline paths, including partial-scale and TP slicing, Linear/MoE layouts, prepared tensors and separate Diffusion attention selection.
---

# MXFP8 implementation checklist

Resolve repository root from this directory (`../../..`). Read `docs/quantization-overview.md`, `docs/known-pitfalls.md`, and use `sglang-quant-lookup` to select/trace the actual worktree. Verify reference checkout versions; do not copy stale pseudo-code from a different backend.

- Separate FP8 payload from UE8M0 scale storage and from prepared scale layout. Dynamic quant output may be pair-packed; do not assume every API returns a flat `[M,K/32]` tensor.
- MX block size is 32. Use ceil for partial blocks and pad odd block counts before pair packing. For RowParallel TP, use the global K offset, including any left block offset; local K alone is insufficient.
- Keep offline checkpoint keys, Parameter types and loader metadata intact. Online BF16/FP16 quantization and offline loading may share prepared execution without sharing preparation.
- Confirm weight/scale orientation, stride, format cast and matmul transpose semantics. Neither pretranspose nor `.contiguous()` is universally necessary or faster; check the actual kernel/version.
- In MoE, verify routing payload/scales, group-list convention and output dtype together. Input quantization reuse must not carry stale tensors across forwards or incompatible backends.
- Diffusion Linear quantization and MXFP8 attention are distinct. Automatic backend selection must respect explicit choices and model constraints; video success alone does not prove quantized attention was used.
- Guard NPU-only imports and preserve CPU construction. Use existing partial-scale, loader and apply tests before inventing another test framework.

Report source-backed contracts and narrowly scoped checks. CPU/mock, NPU numerics, graph, model quality and serving performance are separate claims; no blanket hardware support from one successful shape.
