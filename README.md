![English](https://img.shields.io/badge/Lang-English-blue) [![中文](https://img.shields.io/badge/语言-中文-red)](./README_zh.md)
# SGLang MXFP8 on Ascend NPU Research

This repository (`sglang_quant_eval`) is dedicated to researching and implementing MXFP8/MXFP4 quantization adaptation for **SGLang** on **Huawei Ascend NPU** hardware.

## 🎯 Project Objective

- **Target**: Adapt SGLang's quantization system to support Huawei Ascend NPU using MXFP8 and MXFP4 data formats.
- **Supported Models**: Both standard LLMs (e.g., Qwen3, Qwen3.5, Llama, DeepSeek) via `srt` and Diffusion models (e.g., Wan2.2) via the `multimodal_gen` subsystem.
- **Related Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion), [sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLMs)

## 📁 Repository Structure

- `sglang/` - Local `git worktree` container with the SGLang fork checked out across 2 active branches. `qwen3_dense_w4a4/` is the main clone **and a submodule** (clickable on GitHub → fork's `junlin_qwen3_dense_w4a4`); `qwen3_moe_w8a8/` is a derived worktree (local-only, gitignored). Merged features (Diffusion MXFP8/MXFP4, Dense W8A8/W4A8) already live in upstream/main. See `AGENTS.md` for the worktree-to-branch map and merged-PR list.
- `MindIE-SD/` - Huawei's MindIE-SD source code (submodule, tracks `dev`), serving as a primary reference implementation for Ascend NPU MXFP8/FP8 operations (Diffusion).
- `msmodelslim/` - Huawei's msmodelslim source code (submodule, tracks `master`), reference for the offline MXFP4/MXFP8 weight export format.
- `vllm-ascend/` - vLLM backend code for Ascend (submodule, tracks `main`), serving as a primary reference for LLM MXFP adaptation.
- `diffusion/` & `llm/` - Run scripts and PR notes for Diffusion (Wan2.2) and LLM (Qwen3) inference / quantization.
- `sglang_mxfp8_ascend_research.md` / `_zh.md` - Comprehensive research report, analysis, and implementation plan for the MXFP8 adaptation in English and Chinese.
- `README.md` / `README_zh.md` - Project description and guide in English and Chinese.
- `CLAUDE.md` - AI assistant system instructions and project context.
- `.agent/` & `.claude/` - Custom agent skills and configurations for AI assistants to help with codebase reading and Gitmoji commits.

## 🚀 Implementation Paths

Based on our research (detailed in the research report), there are two main paths for MXFP8 adaptation:

1. **Offline Quantization (msmodelslim)**: Adapting SGLang to load pre-quantized MXFP8 weights produced by Huawei's `msmodelslim` tool. This involves adding to SGLang's existing msmodelslim scheme framework.
2. **Online Quantization**: Implementing dynamic MXFP8 quantization during inference directly from FP16/BF16 weights using `--quantization mxfp8`.

Both paths leverage core `torch_npu` APIs such as `torch_npu.npu_dynamic_mx_quant` and `torch_npu.npu_quant_matmul`.

## 💻 Environment Requirements

To develop and run the code in this repository, the following environment is required:
- **Hardware**: Huawei Ascend NPU (e.g., Atlas 800I A2/A3)
- **Software**: CANN >= 8.0.RC3 (required for `npu_dynamic_mx_quant` and MXFP8 support)
- **Dependencies**: `torch`, `torch_npu`, and `sglang` dependencies.

## 🔧 AI Agent Skills

This repository includes custom tools in `.agent/skills` to assist with development:
- `sglang-quant-lookup`: Quickly find SGLang quantization implementation details.
- `npu-api-check`: Analyze `torch_npu` API usage patterns.
- `compare-impl`: Compare implementations between SGLang and MindIE-SD.
- `trace-quant-path`: Trace the full code path for a quantization method in SGLang.
- `check-issue`: Check the latest status of SGLang GitHub issues/PRs related to our work.
- `gitmoji_commit`: Automatically generate Gitmoji-compliant commit messages.
