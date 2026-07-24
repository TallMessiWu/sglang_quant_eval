# 分支详情

> 本文档记录活跃分支的开发历史、commit hash、A5 验证节点、调试过程。
> 已合并 PR 的代码都在 upstream/main，概览见 [AGENTS.md](../AGENTS.md#已合并-pr代码已在-upstreammain)，此处不再展开。
> Qwen MoE W4A8 的旧实现仍保存在 fork 远程 `junlin_qwen3_moe_w4a8`（`924fea916`）；当前开发已转到本地 `junlin_qwen3.5_moe_w4a8` worktree，并以最新 Qwen3.5 W8A8 分支为基线。

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
> W4A4 rebase 前的旧历史（diff 重含已合并 W8A8/W4A8 代码）存于 `backup-w4a4-pre-rebase-20260707`。**`sglang/qwen3_dense_w4a4/` 目录已于 2026-07-21 删除**，主仓子模块条目同步移除；主 clone 角色已转由 `sglang/qwen3_moe_w8a8/` 承担。分支仍在 fork 远程（`ac07c7f9f`），代码也已在 upstream/main。

---

## `codex/fix-modelslim-mxfp4-packed-weight` — Dense W4A4 offline packed checkpoint 修复 🚧

目录：`sglang/fix_modelslim_mxfp4_packed/`。2026-07-22 从最新 `upstream/main` `93cb9a548` 创建。

- 根因：PR #23795 按旧版 ModelSlim `float8_e4m3fn [out,in]` checkpoint 实现并用旧模型完成 e2e；ModelSlim 自 `4557940` 起改为 `pack_fp4_to_uint8`，当前 checkpoint 物理格式为 `uint8 [out,in/2]`。
- 修复：`ModelSlimMXFP4Scheme.create_weights` 按 packed shape/dtype 注册参数；`NPUSingleLevelMXFP4OfflineLinearMethod` 直接 transpose 已 packed 权重，移除 `npu_dtype_cast` 二次打包。
- 验证：新增 CPU 回归测试覆盖 placeholder shape/dtype 和 post-load 不二次打包；静态检查已通过，A5 新版 checkpoint e2e 待跑。Commit `d875f6684`。

---

## `junlin_qwen3_moe_w8a8` — LLM MoE W8A8 (MXFP8) 🚧 PR [#30768](https://github.com/sgl-project/sglang/pull/30768) WIP OPEN

LLM 侧，Qwen3/3.5 MoE W8A8 MXFP8（FusedMoE，`--quantization mxfp8`）；EPMoE 待实现。此目录（`sglang/qwen3_moe_w8a8/`）**现为主 clone**（持有共享 `.git`，2026-07-21 起；gitignore、纯本地，主仓不跟踪）。当前 HEAD `e22b7bef8`（2026-07-21 merge `upstream/main` @ `c0ed009f5`，178 commit）。

- 在线：`NPUMXFP8OnlineMoEMethod`（`online_moe_methods.py`，继承 `UnquantizedFusedMoEMethod`，只 override `create_moe_runner`），A5 e2e 已验证。
- 离线：`W8A8_MXFP8` → `ModelSlimMXFP8MoEScheme` → `NPUMXFP8MoEMethod` 离线分支，A5 e2e 已验证。
- MXFP8 MoE kernel 契约（torch_npu 2.10.0.post2 + A5 已探针验证）详见 [known-pitfalls.md](known-pitfalls.md) 与 AGENTS.md「已知陷阱」。

### OrangeRedeng 评审落地（2026-07-20，逐条增量）

首次以单个大 commit（`6bd7056`）应用全部 8 条，A5 上乱码且无法二分，遂 reset 重来：**每条（或每小组）一个 commit + push + A5 验证**，通过后才做下一条。

| # | 内容 | commit |
| - | ---- | ------ |
| ⑥ | 离线 flagless，文档去掉 `--quantization` | `9efe9998d` |
| ⑦ | `hidden_states_quant.py` → `quant.py` + mxfp8 dtype | `efe50dfa3` |
| ②⑤ | `GroupedMatmulSwigluQuant` wrapper + 按 `weight_prefix` 选 matmul | `2d06d61a5` |
| ① | DeepEP 不再硬拒，dispatcher 发 bf16 时 gmm1 自量化 | `1efd2ca35` |
| ③④ | 继承 `UnquantizedFusedMoEMethod` + 改名 `NPUMXFP8OnlineMoEMethod` | `ea24f248d` |
| ⑧ | NZ `npu_format_cast` | `d419aa41f`（2026-07-21） |

⑥⑦②⑤①③④ 七条的 review thread 已 resolve；⑧ 已在 thread 内回复数据（[discussion_r3620392948](https://github.com/sgl-project/sglang/pull/30768#discussion_r3620392948)），由 OrangeRedeng 决定是否 resolve。**顺带查出三条与评审无关的 merge 移植回归**（`e9b36bb5d` e8m0 dtype / `9ac21f5ff` flashinfer runner override / `969087364` 离线空 weight offset），根因见 AGENTS.md「已知陷阱」。

### ⑧ FRACTAL_NZ 实测（2026-07-21）

一次性 kernel 探针（四轮迭代，脚本已删）+ A5 e2e 双跑（ND / NZ × 在线 / 离线）：

- **kernel 级**（gmm1+gmm2，128 experts）：decode **+1.4%**、prefill **+3.8%**，噪声底噪 0.2~0.3%，cos 与 ND 完全一致。
- **e2e**：吞吐在线 **+2.0%**、离线 **+3.0%**；mean TTFT 两者均 **−6.5%** 左右；P99 TPOT −3.8% / −5.5%。
- 远不及 OrangeRedeng 提到的 int case ~10%，但方向一致，故采纳。
- **cast 必须在 transpose 之前**（`CheckMXTranspose` 断言）；**别用小 expert 数测 layout**（E=4 会给出相反结论）。两条均记入 [known-pitfalls.md](known-pitfalls.md)。

PR 正文的性能表（+44% 吞吐 / −30% 延迟）与 GSM8K（在线 95.93 / 离线 95.78 / 基线 95.91）已于 2026-07-21 更新为重构后 + NZ 的数据。

> 曾用派生分支 `junlin_qwen3_moe_w8a8_nz` 承载一个 NZ 生效日志 commit（`_log_mxfp8_weight_format`），未并入 PR，2026-07-21 已连同 worktree 一并删除。

### 2026-07-21 merge upstream/main（`c0ed009f5`，178 commit）

PR 出现冲突后合并上游，选 merge 而非 rebase——分支历史已有两次 merge，且 PR body / 本文件大量引用具体 commit hash，rebase 会全部失效。

- **仅一处冲突**：`modelslim/schemes/__init__.py` 的 `__all__`，我方 `ModelSlimMXFP8MoEScheme` 与上游 `ModelSlimMXFP4Scheme`（dense W4A4 线）互不冲突，两者都留、按 import 顺序排。
- 上游对 `moe_methods.py` 只加了 11 行且仅涉及 `NPUW4A8Int8MoEMethod`（int8 W4A8 的 `_update_bias`），**未触碰 MXFP8**；`utils.py` / `online_moe_methods.py` / `moe_runner/ascend.py` 均无改动。这次不是 2026-07-16 那种整层重写。
- merge 后已核对：NZ cast、`_is_nz_aligned` fp8 分支、`overrides.py` NPU 豁免、`fp8.py` dispatch、离线 scheme 注册、`unquant.py` 那处删除**全部完好**；相对上游净改动 18 文件 / +627 −42，全是 MXFP8 MoE 相关。
- PR 状态回到 `MERGEABLE`（`BLOCKED` 只是缺 `run-ci` label）。

> ⚠️ **A5 e2e 尚未在 merge 后重跑**。按本仓库教训（见 AGENTS.md「上游大 merge 会静默重写 NPU MoE 层」），大 merge 后应先用 `llm/qwen3_moe_online_w8a8.sh` / `llm/qwen3_moe_offline_w8a8.sh` 起服务跑一遍 e2e 校准 baseline 再叠改动。当前 PR 正文的性能/精度数字测自 merge 前的 `d419aa41f`。

---

## `junlin_qwen3.5_moe_w8a8` — Qwen3.5 Dense/MoE W8A8 (MXFP8) 🚧

Qwen3.5 Dense/MoE W8A8 MXFP8 实验/验证分支。此目录（`sglang/qwen3.5_moe_w8a8/`）是从主 clone `sglang/qwen3_moe_w8a8/` 派生的 worktree（gitignore、纯本地，主仓不跟踪）。

- **2026-07-21 rebase 到 `junlin_qwen3_moe_w8a8`（`e22b7bef8`）**，基线 = upstream/main（`c0ed009f5`）+ MoE W8A8 PR #30768 全部实现，故本分支现在**同时含未合并的 MoE W8A8 代码**。随后加入 Qwen3.5 ModelSlim packed mapping、视觉量化配置和 partial scale 修复，当前 HEAD `fc9cd5bad`，已推 fork。
- 原基线 upstream/main（`4cec9ef9d`，2026-07-13），已含全部已合并 MXFP8/W8A8/W4A8 量化代码。
- Qwen3.5 与 Qwen3 在 SGLang 内部共用底层 Linear/MoE 算子，故量化实现代码完全一致；在线 Qwen3.5 MoE W8A8 A5 精度正常，离线文本+图片 e2e 已验证，partial scale 修复后的图片 e2e 待重跑。
- 启动脚本：`llm/qwen3.5_{dense,moe}_{bf16,online_w8a8,offline_w8a8}.sh`（实际文件名见 `llm/`）。

## `junlin_qwen3.5_moe_w4a8` — Qwen3/3.5 MoE W4A8 (MXFP4/8) 🚧

从 `junlin_qwen3.5_moe_w8a8` `fc9cd5bad` 创建，worktree 为 `sglang/qwen3.5_moe_w4a8/`。Qwen3 与 Qwen3.5 共用 `qwen3_moe.py` 和底层 FusedMoE，因此没有新增模型专属量化类。

- 旧 `origin/junlin_qwen3_moe_w4a8` 的在线+离线 W4A8 两个提交已移植到新基线，保留后续 4D pair-split scale 修复。
- 在线入口 `NPUMXFP4W4A8FusedMoEMethod` 继承当前通用 `UnquantizedFusedMoEMethod`，只负责安装 W4A8 per-GMM kernel 和固定 Ascend runner。
- 离线 `W4A8_MXFP` 复用同一 `NPUMXFP4W4A8MoEMethod`；补注册空 `weight_offset`，避免共享 ModelSlim apply 构造 `AscendQuantInfo` 时抛 `AttributeError`。
- CPU 回归测试覆盖 Qwen 标准 expert scheme 分发、packed weight/scale placeholder、空 offset，以及 3D/4D scale 归一化；本机 Python 无 `torch`，测试逻辑待有依赖环境运行。A5 在线/离线 e2e 待验证。
- 启动脚本：`llm/qwen3.5_moe_online_w4a8.sh` 从 `/mnt/share/weight/Qwen3.5-35B-A3B` 在线量化；`llm/qwen3.5_moe_offline_w4a8.sh` 加载 `/mnt/weight/Qwen3.5-35B-A3B-mxfp-w4a8` 并自动检测 ModelSlim scheme。
- 2026-07-24 首次 A5 在线图片请求在视觉 QKV 报 `The dimension of bias should be 2`：共享 W4A8 Linear 误传 FP32 一维 bias；提交 `1ca00a86e` 已改为 BF16 `[1,N]` 并覆盖在线/离线，Qwen3 无 bias 路径保持不变，待 A5 重跑。
- 当前 HEAD 为 `1ca00a86e`，实现提交已推送到 fork 的 `junlin_qwen3.5_moe_w4a8`。

---

## 备份分支

用于回退/还原，勿当工作分支：

| 分支 | 内容 | 远端 |
| ---- | ---- | ---- |
| `backup-w4a4-pre-rebase-20260707` | PR #23795 rebase 重建前的旧历史（diff 重含已合并 W8A8/W4A8） | 已推 fork (`origin`) |
| `backup-w4a8-with-int4-dynamic-20260622` (`8c4e5857f0`) | 含被摘除的 INT4 `W4A8_DYNAMIC`（`ModelSlimW4A8Int8` + `NPUW4A8DynamicLinearMethod`）；要还原 INT4 从此 cherry-pick | 已推 fork (`origin`) |
| `backup-w4a8-pre-squash-20260625` (`a1947f3133`) | W4A8 squash 前的多提交历史（16 commit：feature 初版 + 离线/在线 A5 调试）；看单独 commit 从此 checkout | 已推 fork (`origin`) |
| `junlin_qwen3_dense_w4a8_strided` (`72fa20005`) | w4a8 的 strided-view layout 优化版（NPU 实测变慢已回退） | 已推 fork (`origin`) |
