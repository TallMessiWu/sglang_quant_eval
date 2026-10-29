---
name: sglang-quant-lookup
description: Locate and trace SGLang quantization from config through loading and inference, check NPU operator contracts, or compare the implementation with vllm-ascend, MindIE-SD and ModelSlim.
---

# Quantization lookup and comparison

1. Read the repository's `AGENTS.md`, `CONTEXT.md`, `docs/quantization-overview.md` and relevant sections of `docs/known-pitfalls.md`. Resolve the repository root from this skill directory (`../../..`), not from the caller's cwd.
2. Run `git -C <repo>/sglang/qwen3.5_dense_w8a8 worktree list --porcelain`; select the requested branch's checkout using `docs/branches.md`, then verify HEAD/diff. Never assume `<repo>/sglang/python` is a checkout.
3. Identify runtime (SRT/Diffusion), layer (Linear/MoE), weight source (online/ModelSlim), format (MXFP8/W4A8/W4A4) and actual TP/EP/backend. Do not infer precision from model names.
4. Trace only the relevant chain: CLI/config registry → quant method/scheme → parameter creation and native loader → preparation/completion → dispatch → apply/GMM. Include fallback and ignored-layer decisions. Search the selected checkout, not all worktrees.
5. At each NPU call record payload/scale dtype and shape, transpose/stride/ND/NZ, group-list convention, bias, output dtype and ownership. Distinguish checkpoint storage from prepared representation. Check installed API/version before proposing a change.
6. For comparison, use `vllm-ascend/vllm_ascend/quantization/` for SRT, `MindIE-SD/mindiesd/` for Diffusion, and the current `msmodelslim/` exporter for storage. Verify their HEADs; similar names do not prove identical contracts.

Return a short call chain with `path:line`, relevant version, and any contract difference. Separate observed code from assumptions, CPU checks from NPU results, and reusable execution from incompatible loading formats. Do not copy a second implementation for Qwen3.5 or turn a lookup into an unsolicited refactor.
