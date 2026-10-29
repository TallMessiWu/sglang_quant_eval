# Ascend NPU 上的 SGLang MXFP 量化

[English](README.md)

研究、实现和验证 SGLang 的 **MXFP8 / MXFP4**：覆盖 LLM serving (`srt`)、Diffusion (`multimodal_gen`)、Dense/MoE，以及在线准备和离线 ModelSlim checkpoint。

- SGLang fork：<https://github.com/TallMessiWu/sglang>
- 上游跟踪：[Diffusion #14424](https://github.com/sgl-project/sglang/issues/14424)、[LLM #21584](https://github.com/sgl-project/sglang/issues/21584)。

## 从这里开始

| 需求 | 唯一入口 |
| --- | --- |
| 启动与评测 LLM | [llm/README.md](llm/README.md)，默认在 `llm/` 运行 |
| 视频生成 | [diffusion/README.md](diffusion/README.md) |
| 目标、PR 依赖与验收证据 | [重构计划](docs/mxfp-refactoring-plan.md) |
| 分支、worktree、HEAD 与归档 | [分支表](docs/branches.md) |
| 格式与源码路径 | [量化概览](docs/quantization-overview.md) |
| 加载、布局与部署故障机制 | [已知陷阱](docs/known-pitfalls.md) |
| 配套 910/950 kernel wheel | [构建说明](docs/sgl-kernel-npu-build.md) |

## 仓库边界

`sglang/` 是被 Git 忽略的独立 worktree 容器，**不是子模块**。`sglang/qwen3.5_dense_w8a8/` 持有共享 `.git`，不能当过期功能目录删除；每个修改分支都要有直属独立 worktree。

`sgl-kernel-npu/` 是开发子模块，`vllm-ascend/`、`MindIE-SD/`、`msmodelslim/` 是参考子模块。实际 checkout 与主仓 gitlink 分开管理，清理不顺手记录版本变化。

`llm/`、`diffusion/` 保留普通启动器、评测、聚焦诊断和量化描述样例。退出使用的补丁、重复 Python 启动器和历史演示由 Git 恢复，不继续作为另一套运行说明。

## 量化与验证

在线从 BF16/FP16 权重出发，在初始化时准备量化张量。SRT 常用 `mxfp8`、`mxfp_w4a8`、`mxfp4`；Diffusion NPU FP4 脚本使用 `mxfp4_npu`。离线通常由 `quant_model_description.json` 自动选择 scheme，无需显式 ModelSlim 参数。

硬件支持取决于实际算子、格式、TP 后 shape 和 CANN/torch_npu/kernel wheel 组合。依赖以选定 SGLang checkout 为准，不用统一最低版本承诺全路径可用。实现存在、CPU/mock、NPU 数值、模型精度和 serving 性能分别记录。

## Agent 约定

[AGENTS.md](AGENTS.md) 维护操作与发布规则，[CONTEXT.md](CONTEXT.md) 维护领域语言；`CLAUDE.md`、`.claude/skills` 是软链接，不维护副本。

[项目技能](.agents/skills/) 中 `sglang-quant-lookup` 合并了查找、调用链、API 契约与参考实现对照；另有 MXFP 检查表、kernel 开发、实时 PR 查询与提交发布技能。外部工程技能保留 `skills-lock.json` 来源，不为精简改写整套导入内容。
