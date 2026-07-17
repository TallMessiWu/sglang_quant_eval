# 分支详情

> 本文档记录活跃分支的开发历史、commit hash、A5 验证节点、调试过程。
> 已合并 PR 的代码都在 upstream/main，概览见 [AGENTS.md](../AGENTS.md#已合并-pr代码已在-upstreammain)，此处不再展开。
> `junlin_qwen3_moe_w4a8` 尚无独立小节，现状见根 [AGENTS.md](../AGENTS.md#实现进度) 的实现进度表。

## 已合并 PR（速查）

| PR | 内容 | 原 head 分支 | 合并日期 | 归档 |
| -- | ---- | ----------- | -------- | ---- |
| [#20922](https://github.com/sgl-project/sglang/pull/20922) | Diffusion MXFP8（Wan2.2） | `junlin` | 2026-05-07 | — |
| [#22338](https://github.com/sgl-project/sglang/pull/22338) | Diffusion MXFP4 | `junlin_mxfp4` | 2026-05-19 | — |
| [#22352](https://github.com/sgl-project/sglang/pull/22352) | Qwen3 Dense W8A8 MXFP8 | `junlin_qwen3_dense` | 2026-06-16 | — |
| [#28505](https://github.com/sgl-project/sglang/pull/28505) | Dense MXFP8 重构（`ModelSlimMXFP8Scheme` 委托 `NPUMXFP8LinearMethod` + op 走 `torch.ops.npu.*`） | `junlin_qwen3_dense_w8a8` | 2026-06-17 | — |
| [#23650](https://github.com/sgl-project/sglang/pull/23650) | Qwen3 Dense W4A8 MXFP（在线 `mxfp_w4a8` 单级真 W4A8 + 离线 `W4A8_MXFP`） | `junlin_qwen3_dense_w4a8` | 2026-07-06 | 见备份表 |
| [#23795](https://github.com/sgl-project/sglang/pull/23795) | Qwen3 Dense W4A4 MXFP4（在线双级 `NPUDualLevelMXFP4LinearMethod` + 离线单级 `ModelSlimMXFP4Scheme`；双级修复了单级 RTN 贪心死循环，跑分 GSM8K 在线 93.76/离线 93.48 vs BF16 95.32） | `junlin_qwen3_dense_w4a4` | 2026-07-17 | 见备份表 |

> W4A8 曾从 PR 中摘除越界的 INT4 `W4A8_DYNAMIC` 离线 scheme（`ModelSlimW4A8Int8` + `NPUW4A8DynamicLinearMethod`），完整存于备份分支 `backup-w4a8-with-int4-dynamic-20260622`；日后要还原直接 cherry-pick/checkout。
> W4A4 rebase 前的旧历史（diff 重含已合并 W8A8/W4A8 代码）存于 `backup-w4a4-pre-rebase-20260707`。**`sglang/qwen3_dense_w4a4/` 目录本身不删**——它是 worktree 主 clone + 主仓子模块，其余 3 个 worktree 的 `.git` 都挂在它下面。

---

## `junlin_qwen3_moe_w8a8` — LLM MoE W8A8 (MXFP8) 🚧 PR [#30768](https://github.com/sgl-project/sglang/pull/30768) WIP OPEN

LLM 侧，Qwen3/3.5 MoE W8A8 MXFP8（FusedMoE/TP，`--quantization mxfp8`）；EPMoE 待实现。此目录（`sglang/qwen3_moe_w8a8/`）是从主 clone `sglang/qwen3_dense_w4a4/` 派生的 worktree（gitignore、纯本地，主仓不跟踪）。

- 在线：`NPUMXFP8FusedMoEMethod`（三段式 `create_weights` / `process_weights_after_loading` / `apply`），A5 e2e 已验证。
- 离线：`W8A8_MXFP8` → `ModelSlimMXFP8MoEScheme` → `NPUMXFP8FusedMoEMethod` 离线分支，e2e **待验证**。
- MXFP8 MoE kernel 契约（torch_npu 2.10.0.post2 + A5 已探针验证）详见 [known-pitfalls.md](known-pitfalls.md) 与 AGENTS.md「已知陷阱」。

---

## `junlin_qwen3.5_dense_w8a8` — Qwen3.5 Dense W8A8 (MXFP8) 🚧

Qwen3.5 Dense W8A8 MXFP8 实验/验证分支。此目录（`sglang/qwen3.5_dense_w8a8/`）是从主 clone `sglang/qwen3_dense_w4a4/` 派生的 worktree（gitignore、纯本地，主仓不跟踪）。

- 基于 upstream/main（`4cec9ef9d`，2026-07-13），已含全部已合并 MXFP8/W8A8/W4A8 量化代码。
- Qwen3.5 与 Qwen3 在 SGLang 内部共用底层 Linear/MoE 算子，故量化实现代码完全一致；此分支仅需验证 Qwen3.5-8B 模型在 A5 上的在线 + 离线 W8A8 效果、跑分。
- 启动脚本：`llm/qwen3.5_dense_bf16.sh`（BF16 基线）。

**状态**：
- 在线 W8A8：待验证
- 离线 W8A8：待验证
- 跑分：待进行

---

## 备份分支

用于回退/还原，勿当工作分支：

| 分支 | 内容 | 远端 |
| ---- | ---- | ---- |
| `backup-w4a4-pre-rebase-20260707` | PR #23795 rebase 重建前的旧历史（diff 重含已合并 W8A8/W4A8） | 已推 fork (`origin`) |
| `backup-w4a8-with-int4-dynamic-20260622` (`8c4e5857f0`) | 含被摘除的 INT4 `W4A8_DYNAMIC`（`ModelSlimW4A8Int8` + `NPUW4A8DynamicLinearMethod`）；要还原 INT4 从此 cherry-pick | 已推 fork (`origin`) |
| `backup-w4a8-pre-squash-20260625` (`a1947f3133`) | W4A8 squash 前的多提交历史（16 commit：feature 初版 + 离线/在线 A5 调试）；看单独 commit 从此 checkout | 已推 fork (`origin`) |
| `junlin_qwen3_dense_w4a8_strided` (`72fa20005`) | w4a8 的 strided-view layout 优化版（NPU 实测变慢已回退） | 已推 fork (`origin`) |
