# 分支详情

> 本文档只记录 **3 个活跃分支** 的开发历史、commit hash、A5 验证节点、调试过程。
> 已合并 PR 的代码都在 upstream/main，概览见 [AGENTS.md](../AGENTS.md#已合并-pr代码已在-upstreammain)，此处不再展开。

## 已合并 PR（速查）

| PR | 内容 | 原 head 分支 | 合并日期 | 归档 |
| -- | ---- | ----------- | -------- | ---- |
| [#20922](https://github.com/sgl-project/sglang/pull/20922) | Diffusion MXFP8（Wan2.2） | `junlin` | 2026-05-07 | — |
| [#22338](https://github.com/sgl-project/sglang/pull/22338) | Diffusion MXFP4 | `junlin_mxfp4` | 2026-05-19 | — |
| [#22352](https://github.com/sgl-project/sglang/pull/22352) | Qwen3 Dense W8A8 MXFP8 | `junlin_qwen3_dense` | 2026-06-16 | — |
| [#28505](https://github.com/sgl-project/sglang/pull/28505) | Dense MXFP8 重构（`ModelSlimMXFP8Scheme` 委托 `NPUMXFP8LinearMethod` + op 走 `torch.ops.npu.*`） | `junlin_qwen3_dense_w8a8` | 2026-06-17 | — |
| [#23650](https://github.com/sgl-project/sglang/pull/23650) | Qwen3 Dense W4A8 MXFP（在线 `mxfp_w4a8` 单级真 W4A8 + 离线 `W4A8_MXFP`） | `junlin_qwen3_dense_w4a8` | 2026-07-06 | 见备份表 |

> W4A8 曾从 PR 中摘除越界的 INT4 `W4A8_DYNAMIC` 离线 scheme（`ModelSlimW4A8Int8` + `NPUW4A8DynamicLinearMethod`），完整存于备份分支 `backup-w4a8-with-int4-dynamic-20260622`；日后要还原直接 cherry-pick/checkout。

---

## `junlin_qwen3_dense_w4a4` — LLM Dense W4A4 (MXFP4) ✅ PR [#23795](https://github.com/sgl-project/sglang/pull/23795) OPEN

LLM 侧，Dense W4A4 在线量化（`--quantization mxfp4`）+ 离线 W4A4（`W4A4_MXFP4` → `ModelSlimMXFP4Scheme`，真 MXFP4）。**此目录（`sglang/qwen3_dense_w4a4/`）现为 worktree 主 clone + 主仓子模块**（主仓跟踪其 commit 指针）。

**在 #22352 + #23650 均合并后已 rebase 重建**（2026-07-07，旧历史存 `backup-w4a4-pre-rebase-20260707`，已推 fork）：
- `git reset --hard upstream/main`（`e85ef5487`，含 #23650 squash）去掉 diff 里重复的已合并 W8A8/W4A8 代码，只留 W4A4 delta（8 文件 442 行，单一 feature 提交 head `d831fb30e`）。
- 对齐 #23650 重构风格：① CLI `mxfp4w4a4_npu` → `mxfp4`（设备无关；`__init__.py` 的 `is_npu()` 块加 `"mxfp4": Mxfp4W4A4Config`，镜像 `GPTQAscendConfig`——NPU 走我们的 W4A4、GPU 走上游 `Mxfp4Config`，零冲突）；② config `NPUMxfp4W4A4Config` → `Mxfp4W4A4Config`，`get_quant_method` 按 `is_npu()` 分发；③ 在线 `NPUSingleLevelMXFP4LinearMethod` + 离线 `NPUSingleLevelMXFP4OfflineLinearMethod`（继承在线、共享 `apply`）全走 `torch.ops.npu.*`、复用惰性 helper `_get_float4_e2m1fn_x2_dtype()`（fp4 dtype 必来自 torch_npu）/`_get_float8_e8m0fnu_dtype()`，无顶层 `import torch_npu`；④ 离线 `ModelSlimMXFP4Scheme` 继承 `ModelSlimLinearScheme` + `self.kernel` 委托，注册进 `get_linear_scheme` 的 `("W4A4_MXFP4", ...)`；⑤ 文档写 `docs_new/`。

**在线 + 离线均已在 A5 e2e 验证输出正常**（2026-07-09）。在线走双级、离线走单级（`npu_quant_matmul(x1=x2=fp4, group_sizes=[1,1,32])`，checkpoint 自带 UE8M0 scale；与 W4A8 的 `[0,0,32]`+NZ 是不同路径）。跑分（PR #23795 body 有完整表）：性能 在线约 +41% 吞吐 / −29% 时延、离线约 +49% / −32%；GSM8K 在线 93.76、离线 93.48 vs BF16 95.32（**在线双级反超离线单级**）。PR 已去 WIP。

🆕 **在线路径改用双级（dual-level）MXFP4**（2026-07-09，head `635d1d8b6`）：单级在线 RTN（UE8M0 幂-2 block scale）精度不足，长文本贪心解码会陷入「只输出 reasoning、发不出 EOS」的死循环（离线校准权重无此问题）。新增 `NPUDualLevelMXFP4LinearMethod`（`npu_dynamic_dual_level_mx_quant` + `npu_dual_level_quant_matmul`，细 FP8 E4M3 L0 scale + 粗 L1 scale，权重 FRACTAL_NZ），移植自 Diffusion `NPUMXFP4DiffusionLinearMethod` / MindIE-SD `W4A4MXFP4DualQuantLinear`，A5 实测死循环消失。**在线现只走双级**（去掉 env 开关与单级分支）；单级 `NPUSingleLevelMXFP4LinearMethod` 仅作离线基类保留。仅 Ascend 950（A5）支持 `DualLevelQuantBatchMatmul`，A2/A3 无此 op。

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
