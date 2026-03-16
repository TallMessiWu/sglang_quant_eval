# 华为 Ascend NPU 上的 SGLang MXFP8 量化适配研究

本仓库 (`sglang_quant_eval`) 致力于研究和实现在 **华为 Ascend NPU** 硬件上为 **SGLang** 适配 MXFP8/MXFP4 量化功能。

## 🎯 项目目标

- **目标**: 适配 SGLang 的量化系统，支持在华为 Ascend NPU 上使用 MXFP8（以及可能的 MXFP4）数据格式。
- **支持模型**: 支持通过 `srt` 运行的标准 LLM (例如 Llama, Qwen, DeepSeek) 以及通过 `multimodal_gen` 子系统运行的 Diffusion 模型 (例如 Wan2.2)。
- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424)

## 📁 仓库结构

- `sglang/` - 核心 SGLang 源码仓库 (submodule/clone)，适配代码将在此处修改。
- `MindIE-SD/` - 华为 MindIE-SD 源码 (submodule/clone)，作为 Ascend NPU MXFP8/FP8 操作的主要参考实现。
- `sglang_mxfp8_ascend_research.md` - MXFP8 适配的详细研究报告、分析及实现方案。
- `CLAUDE.md` - AI 助手的指令配置及技术摘要。
- `.agent/skills/` - 为 AI Agent 定制的技能，用于执行诸如查找 SGLang 量化实现、检查 NPU API 使用情况以及生成 gitmoji 提交信息等任务。

## 🚀 实现路径

根据研究报告的详细分析，MXFP8 适配有两条主要路径：

1. **离线量化 (msmodelslim)**: 适配 SGLang 以加载由华为 `msmodelslim` 工具生成的预量化 MXFP8 权重。这需要在 SGLang 现有的 modelslim scheme 框架中添加新功能。
2. **在线量化**: 允许在推理过程中直接从 FP16/BF16 权重动态量化为 MXFP8，通过使用 `--quantization mxfp8` 参数实现。

这两条路径都依赖于核心的 `torch_npu` API，如 `torch_npu.npu_dynamic_mx_quant` 和 `torch_npu.npu_quant_matmul`。

## 💻 环境要求

开发和运行本仓库中的代码需要以下环境：
- **硬件**: 华为 Ascend NPU（例如 Atlas 800I A2/A3）
- **软件**: CANN >= 8.0.RC3 (支持 `npu_dynamic_mx_quant` 和 MXFP8 的必需最低版本)
- **依赖**: `torch`, `torch_npu` 以及 `sglang` 的相关依赖。

## 🔧 AI Agent 技能

本仓库在 `.agent/skills` 目录中包含了一些辅助开发的自定义 AI Agent 技能工具：
- `sglang-quant-lookup`: 快速查找 SGLang 量化实现细节。
- `npu-api-check`: 分析 `torch_npu` API 使用模式。
- `compare-impl`: 比较 SGLang 与 MindIE-SD 之间的实现差异。
- `trace-quant-path`: 追踪 SGLang 中量化方法的完整代码执行路径。
- `check-issue`: 检查与本项目工作相关的 SGLang GitHub Issues 或 PR 的最新状态。
- `gitmoji_commit`: 自动生成符合 Gitmoji 规范的代码提交信息。
