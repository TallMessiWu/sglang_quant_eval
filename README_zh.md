[![English](https://img.shields.io/badge/Lang-English-blue)](./README.md)![中文](https://img.shields.io/badge/语言-中文-red)
# 华为 Ascend NPU 上的 SGLang MXFP 量化适配

本仓库 (`sglang_quant_eval`) 研究并实现在 **华为 Ascend NPU** 硬件上为 **SGLang** 适配 **MXFP8 / MXFP4** 量化——同时覆盖 LLM serving 侧 (`srt`) 与 Diffusion 侧 (`multimodal_gen`)、**Dense 与 MoE** 层、**在线与离线** 两种模式。

## 🎯 项目目标

- **目标**: 适配 SGLang 的量化系统，支持在华为 Ascend NPU 上使用 microscaling FP 格式 (MXFP8、MXFP4)，涵盖 **W8A8 / W4A8 / W4A4** 各 scheme。
- **支持模型**: 通过 `srt` 运行的 LLM (如 Qwen3、Qwen3.5)，以及通过 `multimodal_gen` 子系统运行的 Diffusion 模型 (如 Wan2.2)。
- **量化模式**: **在线** (加载时实时量化 FP16/BF16 权重) 与 **离线** (加载 `msmodelslim` 预量化权重)。
- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion)、[sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLM Qwen3)。
- **Fork**: <https://github.com/TallMessiWu/sglang>

## 📁 仓库结构

- `sglang/` — SGLang fork 的本地 `git worktree` 容器，将 **2 个活跃分支** 同时 checkout、共享同一个 `.git`。`qwen3_moe_w8a8/` 是主 clone，`qwen3.5_dense_w8a8/` 是派生 worktree。整个目录**纯本地、gitignore**——主仓不再跟踪任何 sglang 子模块。已合并功能 (Diffusion MXFP8/MXFP4、Dense W8A8/W4A8/W4A4) 都在 `upstream/main`。worktree 与分支的对应关系、已合并 PR 清单见 `AGENTS.md`。
- `MindIE-SD/` — 华为 MindIE-SD 源码 (子模块，跟踪 `dev`)；**Diffusion** 侧 Ascend NPU MXFP8/FP8 操作的主要参考实现。
- `msmodelslim/` — 华为 msmodelslim 源码 (子模块，跟踪 `master`)；**离线** MXFP4/MXFP8 权重导出格式的参考。
- `vllm-ascend/` — vLLM Ascend 后端 (子模块，跟踪 `main`)；**LLM** 侧 MXFP 量化实现的主要参考标准。
- `sgl-kernel-npu/` — 上游 NPU kernel 仓 (子模块，跟踪 `main`)；提供量化路径运行时 import 的 `sgl_kernel_npu` (RoPE、triton norm 等)。
- `diffusion/` & `llm/` — Diffusion (Wan2.2) 与 LLM (Qwen3/3.5) 推理及量化的运行脚本、量化描述 JSON 和 PR 说明。
- `docs/` — 项目文档：`branches.md` (分支/PR 状态)、`known-pitfalls.md`、`sgl-kernel-npu-build.md`、`npu-api/` (Ascend kernel API 参考——`DualLevelQuantBatchMatmul`、`DynamicDualLevelMxQuant`)、`agents/` (agent 工作流文档)。
- `AGENTS.md` — AI 助手指令及项目上下文；**唯一内容源** (`CLAUDE.md` 为指向它的软链接)。
- `.agents/skills/` — AI Agent 自定义技能 (`.claude/skills` 为指向该目录的软链接)。
- `README.md` / `README_zh.md` — 中英文项目自述文档。

## 🚀 量化路径

两条正交的量化路径，均依赖 `torch_npu` NPU kernel：

1. **在线量化**: 加载 FP16/BF16 权重，在 `process_weights_after_loading` 中实时量化。用 `--quantization mxfp8` / `mxfp4` / `mxfp_w4a8` 触发。
2. **离线量化 (msmodelslim)**: 加载由华为 `msmodelslim` 工具生成的预量化权重。用 `--quantization modelslim` 触发，scheme 由 `quant_model_description.json` 自动检测。

核心 `torch_npu` API 包括 `npu_dynamic_mx_quant` + `npu_quant_matmul` (MXFP8 / W4A8) 与 `npu_dynamic_dual_level_mx_quant` + `npu_dual_level_quant_matmul` (双级 MXFP4 / W4A4)。完整实现进度矩阵见 `AGENTS.md`。

## 💻 环境要求

- **硬件**: 华为 Ascend NPU。MXFP8 / W8A8 / W4A8 可在 Atlas 800I A2/A3 上运行；**双级 MXFP4 (W4A4) 需 Ascend 950 (A5)**——`DualLevelQuantBatchMatmul` 算子在 A2/A3 上不支持。
- **软件**: CANN ≥ 8.0.RC3 (支持 `npu_dynamic_mx_quant` / MXFP8 的必需版本)；MXFP4 需较新的 `torch_npu` (如 `2.10.0.postX`)。
- **依赖**: `torch`、`torch_npu` 以及 SGLang 的相关依赖。

## 🔧 AI Agent 技能

自定义技能位于 `.agents/skills/` (`.claude/skills` 为指向它的软链接)。重点技能：

- `sglang-quant-lookup`、`trace-quant-path` — 查找 / 追踪 SGLang 量化实现。
- `mxfp4-impl-ref`、`mxfp8-impl-ref` — MXFP4/MXFP8 完整实现参考 (API 签名、shape、gotchas)。
- `npu-api-check`、`compare-impl` — 分析 `torch_npu` API 用法；与 MindIE-SD、vllm-ascend 对比。
- `check-issue` — 检查与本项目相关的 SGLang Issue/PR 状态。
- `gitmoji-commit` — 生成符合 Gitmoji 规范的提交信息。
- 以及工程/工作流类技能 (`diagnose`、`tdd`、`triage`、`handoff` 等)。完整列表见 `.agents/skills/` 目录。
