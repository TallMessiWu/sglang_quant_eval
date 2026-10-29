# SGLang MXFP Quantization on Ascend NPU

[中文](README_zh.md)

Research, implementation and validation of **MXFP8 / MXFP4** for SGLang on Huawei Ascend NPU: LLM serving (`srt`) and Diffusion (`multimodal_gen`), Dense/MoE layers, online preparation and offline ModelSlim checkpoints.

- SGLang fork: <https://github.com/TallMessiWu/sglang>
- Upstream tracking: [Diffusion #14424](https://github.com/sgl-project/sglang/issues/14424), [LLM #21584](https://github.com/sgl-project/sglang/issues/21584).

## Start here

| Need | Entry |
| --- | --- |
| Run or benchmark LLMs | [llm/README.md](llm/README.md) — run commands from `llm/` |
| Generate videos | [diffusion/README.md](diffusion/README.md) |
| Goals, PR dependencies and acceptance evidence | [Refactoring plan](docs/mxfp-refactoring-plan.md) |
| Branch, worktree, HEAD and archive mapping | [Branches](docs/branches.md) |
| Formats and code paths | [Quantization overview](docs/quantization-overview.md) |
| Layout/loading/deployment failure mechanisms | [Known pitfalls](docs/known-pitfalls.md) |
| Build the matching 910/950 kernel wheel | [Kernel build guide](docs/sgl-kernel-npu-build.md) |

## Repository boundaries

`sglang/` is a gitignored container of independent worktrees, **not a submodule**. `sglang/qwen3.5_dense_w8a8/` owns the shared `.git`; never delete it as an obsolete feature checkout. Each modified SGLang branch needs its own direct-child worktree.

`sgl-kernel-npu/` is a development submodule; `vllm-ascend/`, `MindIE-SD/` and `msmodelslim/` are reference submodules. Their checkout revisions are independent of the parent gitlinks; cleanup must not silently stage version changes.

`llm/` and `diffusion/` contain ordinary launchers, evaluation tools, focused diagnostics and sample quantization descriptions. Retired patches, duplicated Python launchers and historical presentations are recoverable through Git rather than maintained as competing instructions.

## Quantization and evidence

Online paths start from BF16/FP16 weights and prepare quantized tensors during initialization. SRT selectors include `mxfp8`, `mxfp_w4a8` and `mxfp4`; Diffusion's NPU FP4 launcher uses `mxfp4_npu`. Offline schemes are normally detected from `quant_model_description.json`, without an explicit ModelSlim flag.

Hardware support depends on the exact operator, format, TP shape and CANN/torch_npu/kernel-wheel combination. Follow the selected SGLang checkout's dependency requirements; there is no repository-wide minimum version that guarantees every path. Implementation, CPU/mock tests, NPU numerics, model quality and serving performance are separate claims.

## Agent guidance

[AGENTS.md](AGENTS.md) defines workflow and publication rules; [CONTEXT.md](CONTEXT.md) defines domain terminology. `CLAUDE.md` and `.claude/skills` are symlinks, not duplicate copies.

Project skills in [.agents/skills/](.agents/skills/) include a consolidated `sglang-quant-lookup` workflow (lookup, tracing, API checks and reference comparisons), MXFP format checklists, kernel development, live PR checks and commit/push validation. Imported engineering skills retain their source metadata in `skills-lock.json`.
