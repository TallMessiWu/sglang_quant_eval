![English](https://img.shields.io/badge/Lang-English-blue) [![中文](https://img.shields.io/badge/语言-中文-red)](./README_zh.md)
# SGLang MXFP Quantization on Ascend NPU

This repository (`sglang_quant_eval`) researches and implements **MXFP8 / MXFP4** quantization adaptation for **SGLang** on **Huawei Ascend NPU** hardware — covering both the LLM serving side (`srt`) and the Diffusion side (`multimodal_gen`), across **Dense and MoE** layers, in **online and offline** modes.

## 🎯 Project Objective

- **Target**: Adapt SGLang's quantization system for Huawei Ascend NPU using microscaling FP formats (MXFP8, MXFP4), spanning **W8A8 / W4A8 / W4A4** schemes.
- **Models**: LLMs (e.g. Qwen3, Qwen3.5) via `srt`, and Diffusion models (e.g. Wan2.2) via the `multimodal_gen` subsystem.
- **Quant modes**: **Online** (quantize FP16/BF16 at load time) and **Offline** (load `msmodelslim` pre-quantized weights).
- **Related Issues**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion), [sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLM Qwen3).
- **Fork**: <https://github.com/TallMessiWu/sglang>

## 📁 Repository Structure

- `sglang/` — Local `git worktree` container for the SGLang fork, checked out across **3 active branches** sharing one `.git`. `qwen3_dense_w4a4/` is the main clone **and a submodule** (clickable on GitHub → fork's `junlin_qwen3_dense_w4a4`); `qwen3_moe_w8a8/` and `qwen3.5_dense_w8a8/` are derived worktrees (local-only, gitignored). Merged features (Diffusion MXFP8/MXFP4, Dense W8A8/W4A8) already live in `upstream/main`. See `AGENTS.md` for the worktree↔branch map and merged-PR list.
- `MindIE-SD/` — Huawei's MindIE-SD source (submodule, tracks `dev`); primary reference for Ascend NPU MXFP8/FP8 operations on the **Diffusion** side.
- `msmodelslim/` — Huawei's msmodelslim source (submodule, tracks `master`); reference for the **offline** MXFP4/MXFP8 weight export format.
- `vllm-ascend/` — vLLM Ascend backend (submodule, tracks `main`); primary reference for **LLM**-side MXFP adaptation.
- `diffusion/` & `llm/` — Run scripts, quant-description JSONs, and PR notes for Diffusion (Wan2.2) and LLM (Qwen3/3.5) inference & quantization.
- `docs/` — Project docs: `branches.md` (branch/PR status), `known-pitfalls.md`, `sgl-kernel-npu-build.md`, `npu-api/` (Ascend kernel API references — `DualLevelQuantBatchMatmul`, `DynamicDualLevelMxQuant`), and `agents/` (agent workflow docs).
- `AGENTS.md` — AI-assistant instructions & project context; the **single source of truth** (`CLAUDE.md` is a symlink to it).
- `.agents/skills/` — Custom agent skills (`.claude/skills` is a symlink to this directory).
- `README.md` / `README_zh.md` — Project guide in English and Chinese.

## 🚀 Quantization Paths

Two orthogonal quantization paths, both leveraging `torch_npu` NPU kernels:

1. **Online quantization**: Load FP16/BF16 weights and quantize on the fly inside `process_weights_after_loading`. Triggered by `--quantization mxfp8` / `mxfp4` / `mxfp_w4a8`.
2. **Offline quantization (msmodelslim)**: Load pre-quantized weights produced by Huawei's `msmodelslim` tool. Triggered by `--quantization modelslim`; the scheme is auto-detected from `quant_model_description.json`.

Core `torch_npu` APIs used include `npu_dynamic_mx_quant` + `npu_quant_matmul` (MXFP8 / W4A8) and `npu_dynamic_dual_level_mx_quant` + `npu_dual_level_quant_matmul` (dual-level MXFP4 / W4A4). See the implementation-status matrix in `AGENTS.md`.

## 💻 Environment Requirements

- **Hardware**: Huawei Ascend NPU. MXFP8 / W8A8 / W4A8 run on Atlas 800I A2/A3; **dual-level MXFP4 (W4A4) requires Ascend 950 (A5)** — the `DualLevelQuantBatchMatmul` op is not supported on A2/A3.
- **Software**: CANN ≥ 8.0.RC3 (required for `npu_dynamic_mx_quant` / MXFP8); MXFP4 needs a recent `torch_npu` (e.g. `2.10.0.postX`).
- **Dependencies**: `torch`, `torch_npu`, and the usual SGLang dependencies.

## 🔧 AI Agent Skills

Custom skills live in `.agents/skills/` (with `.claude/skills` symlinked to it). Highlights:

- `sglang-quant-lookup`, `trace-quant-path` — find / trace SGLang quantization implementations.
- `mxfp4-impl-ref`, `mxfp8-impl-ref` — full MXFP4/MXFP8 implementation references (API signatures, shapes, gotchas).
- `npu-api-check`, `compare-impl` — analyze `torch_npu` API usage; compare against MindIE-SD & vllm-ascend.
- `check-issue` — check the status of SGLang issues/PRs related to this work.
- `gitmoji-commit` — generate Gitmoji-compliant commit messages.
- Plus engineering/workflow skills (`diagnose`, `tdd`, `triage`, `handoff`, …). Browse `.agents/skills/` for the full set.
