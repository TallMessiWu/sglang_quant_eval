# 分支详情

> 本文档记录每个分支的开发历史、commit hash、PR 状态、调试过程。
> 与 CLAUDE.md 中的分支表格配合使用——表格给概要，本文档给细节。

## `junlin_diffusion_w8a8` — 主线（不动）

Diffusion MXFP8 基线。主仓子模块指向此分支。

## `junlin_diffusion_w4a4` — Diffusion MXFP8 + MXFP4 在线量化

基于 `junlin_diffusion_w8a8`，含 MXFP8 + MXFP4 在线量化。

## `junlin_mxfp4_offline` — Diffusion MXFP4 离线加载

在 `junlin_diffusion_w4a4` 基础上增加 MXFP4 离线加载。

## `junlin_qwen3_dense_w8a8` — LLM Dense W8A8 MXFP8 ✅

LLM 侧，Qwen3 / 3.5 dense 模型 MXFP8 量化适配。

**Tamir review follow-up（PR #22352 两条 comment）已开 draft [PR #28505](https://github.com/sgl-project/sglang/pull/28505)**：
- ① `ModelSlimMXFP8Scheme` 委托给 `NPUMXFP8LinearMethod`（`self.kernel`，按 weight dtype 分在线/离线分支，统一 `weight_scale_inv`）
- ② op 改走 `torch.ops.npu.*` + `torch.float8_*`，两文件去掉顶层 `import torch_npu` 与 `current_platform` 守卫

本分支已 rebase 到最新 upstream/main 后 force-push，PR diff 干净。NPU 在线+离线 e2e 已通过（两个验证门 raw `npu_quant_matmul` 吃 MX kwargs、`torch.float8_e8m0fnu` 存在均过），PR 已转正式。

**gemini-code-assist 两条 review 已处理并 resolve**：
- ① 模块级 `_FLOAT8_E8M0FNU_DTYPE` 改惰性 helper → **接纳**（起初判不接纳，因 `torch.float8_e8m0fnu` 是 torch 核心 dtype、import 时即可用且 e2e 已证非 None；但用户指出该模块也在非 NPU CI 被 import，惰性求值 Pareto-safe，遂接纳：commit `473389a4a9` 用 `_get_float8_e8m0fnu_dtype()` 在 `apply` 内求值）
- ② offline 分支删废参数 `weight_scale` → **接纳**（commit `f796a23606` 加 `del layer.weight_scale`）

PR head 现为 `473389a4a9`。

## `junlin_qwen3_dense_w4a8` — LLM Dense W4A8 (MXFP4/8) ✅

LLM 侧，Dense W4A8 在线量化（单级真 W4A8/MXFP8 激活，`--quantization mxfp_w4a8`）；离线 W4A8 已实现（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`）。

**已 merge 同步至 `junlin_qwen3_dense_w8a8` 的 upstream 基线（2026-06-12）**；bias-cache 优化保留，但 strided-view layout 优化因 NPU 实测变慢已回退恢复 `.contiguous()`（2026-06-16，strided 版存档于 `junlin_qwen3_dense_w4a8_strided`）。

**PR [#23650](https://github.com/sgl-project/sglang/pull/23650)（W4A8，draft）** 在 #22352 squash 合并 + #28505 重构落地后已 rebase 重建（2026-06-22，旧历史存档于 `backup-w4a8-pre-rebase-20260622`）：
- `git reset --hard upstream/main` 去掉 diff 里重复的已合并 W8A8 代码
- 对齐 #28505 风格：① `ModelSlimMXFP4W4A8Scheme` 改为 `subclass ModelSlimMXFP8Scheme`（W4A8_MXFP checkpoint 布局与 MXFP8 完全相同，零重复）；② `NPUMXFP4W4A8LinearMethod` 改 `torch.ops.npu.*`，无顶层 `import torch_npu`/`current_platform` 守卫，删掉死常量 `_FLOAT4_E2M1FN_X2_DTYPE`；在线 dual-level 离线 `.contiguous()` 保留。

**再按 owner 决定，从本 PR 摘除越界的 INT4 `W4A8_DYNAMIC` 离线 scheme**（`ModelSlimW4A8Int8` + `NPUW4A8DynamicLinearMethod`，非 MXFP，文档交付清单从未列入）→ 本 PR 现为纯 MXFP4 W4A8（在线 `mxfp4_w4a8_npu` + 离线 `W4A8_MXFP`），7 文件 510 行，head `2efd90a6e5`（2026-06-25 已再 rebase 到最新 upstream/main `efbe67d237` + squash 成单一 feature 提交）。被摘除的 INT4 那套完整存档在分支 `backup-w4a8-with-int4-dynamic-20260622`（本地 + 已推 fork，日后要还原直接 cherry-pick/checkout）。注意 modelslim 是多格式工具（INT8/INT4/MXFP8/MXFP4 由 checkpoint `quant_type` 分发，upstream 本就含 INT8/INT4 scheme）。本地无 torch/NPU，仅过 py_compile + pre-commit。

**离线 W4A8 e2e 已在 A5（950）graph 模式验证输出正常**（2026-06-25，torch_npu `2.10.0.post1.dev20260624`）；**在线 W4A8 e2e 已在 A5 验证输出正常**（2026-07-06，需 fp4 dtype 来源修复，见文末）。

**在线 `mxfp4_w4a8_npu` 已从 dual-level 改为单级真 W4A8**（2026-06-25，原 commit `a1947f3133`，现已 squash 进单一提交 `2efd90a6e5`；旧多提交历史存于 `backup-w4a8-pre-squash-20260625`）：
- apply 完全复用离线 `npu_quant_matmul(x2_dtype=fp4, group_sizes=[0,0,32])`
- 激活走 `npu_dynamic_mx_quant(dst=float8_e4m3fn)`（FP8）
- 权重在线 `npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)` → `npu_format_cast(29)` + transpose，布局与离线一致
- 与离线唯一差别是权重来源（在线 RTN vs msmodelslim 校准）

⚠️ NPU 上默认走 graph 模式，别加 `--disable-cuda-graph`——eager decode 的 `ascend` attention（ATB `_npu_paged_attention`）会 `atb` 段错误（与量化无关，见 known-pitfalls）；若确需 eager decode，加 `ASCEND_USE_FIA=1` 改走 aclnn FIA 绕开，A5 实测可跑。

**PR #23650 review follow-up**（2026-07-02，已 squash 进单一 feature 提交 `b73c30235c` 并 force-push fork；中间增量 commit `ba4735d321` 已被 squash 覆盖）：应 reviewer ping1jing2——
- ① 复用并扩展 `hardware_backend/npu/utils.py:npu_format_cast`（新增 `customize_dtype`/`input_dtype` 转发 + FP4 跳过 `_is_nz_aligned` 的 ND fallback），online/offline 两处 W4A8 权重 cast 不再直调 `torch_npu.npu_format_cast`
- ② W4A8 online/offline 彻底去 `torch_npu`——op 全走 `torch.ops.npu.*`、dtype 走 `getattr(torch,...)` helper（`linear_method_npu.py` 新增 `_get_float4_e2m1fn_x2_dtype`），对齐 MXFP8 标杆，`linear_method_npu.py` + `utils.py` 均无 `import torch_npu`
- ③ online `create_weights` 加 docstring
- ④ `del quant_config, prefix` 保留（与 `ModelSlimMXFP8Scheme.__init__` 一致，仅回复不改）
- ⑤ 新 CLI choice `mxfp4_w4a8_npu` + 离线 `W4A8_MXFP` 已写入文档

✅ **A5 已验证可跑（2026-07-02）**：3 个 `torch.ops.npu` 裸 op 带 kwarg 均正常，`torch.float4_e2m1fn_x2` 在该 torch_npu（`2.10.0.post1.dev20260624`）下存在。gemini 那 6 条锚定 Diffusion `multimodal_gen` 的评论 rebase 后已 out-of-scope 关闭。

**在线 W4A8 权重量化在 torch_npu `2.10.0.post2.dev20260704` 崩「output y must be same shape as input x」/561002**。~~曾误诊为 FP4 kernel 被整体打回、改纯 torch RTN（commit `db4dee06ea`）~~——**2026-07-06 二次诊断推翻：真凶是 fp4 dtype 来源，NOT kernel**。详见 [known-pitfalls.md#fp4-dtype-source](known-pitfalls.md#fp4-dtype-source)。

**PR 文档已从 legacy `docs/` 迁到 `docs_new/docs/`**（`.mdx`；upstream 已迁移目录，CI `lint` 的「reject changes under legacy docs/」会拒 legacy `docs/` 改动、连累 pr-gate）：加 `mxfp_w4a8` method-support 行 + Ascend ModelSlim 表 `MXFP4 W4A8` 行 + 精简「W4A8 for LLM dense」文字节，legacy `docs/` 两文件还原到 base（commit `cab582c88d`）。

**在线 CLI choice 已从 `mxfp4_w4a8_npu` 重命名为 `mxfp_w4a8`**（2026-07-06，commit `136b8b506c`，已推 fork）：去 `_npu` 后缀（设备无关、对齐 `mxfp8`）＋ `mxfp_` 前缀反映全 MX 语义（MXFP4 权重 ＋ MXFP8 激活），与 INT4 权重的 `w4afp8` 区分；同步改 CHOICES／注册键／`get_name()`／docstring／文档及测试脚本；并把 config 类 `NPUMxfp4Config` 重命名为设备无关的 `Mxfp4W4A8Config`、`get_quant_method` 改按 `is_npu` 分发（非 NPU 抛 `NotImplementedError`）。

## `junlin_qwen3_dense_w4a4` — LLM Dense W4A4 (MXFP4) 🚧

LLM 侧，Dense W4A4 在线量化（单级 MXFP4：4-bit 权重 + 4-bit 激活，`--quantization mxfp4`）+ 离线 W4A4（`W4A4_MXFP4` → `ModelSlimMXFP4Scheme`，真 MXFP4）。

**PR [#23795](https://github.com/sgl-project/sglang/pull/23795)（W4A4）** 在 #22352 + #23650 均合并后已 rebase 重建（2026-07-07，旧历史存档于 `backup-w4a4-pre-rebase-20260707`，已推 fork）：
- `git reset --hard upstream/main`（`e85ef5487`，含 #23650 squash）去掉 diff 里重复的已合并 W8A8/W4A8 代码，只留 W4A4 delta（8 文件 442 行，单一 feature 提交 head `d831fb30e`）
- 对齐 #23650 重构风格：① CLI 从 `mxfp4w4a4_npu` 改为 `mxfp4`（设备无关；srt 侧 `mxfp4` 只在 cpu/cuda/hip 注册给上游 `Mxfp4Config`/MoE，NPU 上没注册，故在 `__init__.py` 的 `is_npu()` 块加 `"mxfp4": Mxfp4W4A4Config`，镜像 `GPTQAscendConfig`——NPU 走我们的 W4A4、GPU 走上游，零冲突、不改上游 `Mxfp4Config`）；② config `NPUMxfp4W4A4Config` → 设备无关 `Mxfp4W4A4Config`，`get_quant_method` 按 `is_npu()` 分发；③ 在线 `NPUSingleLevelMXFP4LinearMethod` + 新增离线 `NPUSingleLevelMXFP4OfflineLinearMethod`（继承在线、共享 `apply`）全走 `torch.ops.npu.*`、复用惰性 helper `_get_float4_e2m1fn_x2_dtype()`（fp4 dtype 必来自 torch_npu）/`_get_float8_e8m0fnu_dtype()`，无顶层 `import torch_npu`；④ 离线 `ModelSlimMXFP4Scheme` 改为继承 `ModelSlimLinearScheme` + `self.kernel` 委托，注册进 `get_linear_scheme` 的 `linear_quant_schemes`（`("W4A4_MXFP4", ...)`）；⑤ 文档写 `docs_new/`。

kernel 数学保持原样（单级 fp4，`npu_quant_matmul(x1=x2=fp4, group_sizes=[1,1,32])`、激活 `npu_dynamic_mx_quant(dst=fp4)`，无 FRACTAL_NZ；与 W4A8 的 `[0,0,32]`+NZ 是不同路径）。本地无 torch/NPU，仅过 py_compile + pre-commit；**W4A4 MXFP4 尚未 A5 e2e 验证，PR 标 WIP，NZ 布局待 A5 bring-up**。

⚠️ 旧进度表曾把离线记为 INT4 `W4A4_DYNAMIC`，与分支实际（`W4A4_MXFP4` 真 MXFP4）不符，已订正。

## `junlin_qwen3_moe_w8a8` — 当前工作分支 🚧

LLM 侧，Qwen3/3.5 MoE W8A8 MXFP8 在线量化（FusedMoE/TP，`--quantization mxfp8`）；EPMoE 待实现。

---

## 备份分支

用于回退/还原，勿当工作分支：

| 分支 | 内容 | 远端 |
| ---- | ---- | ---- |
| `junlin_qwen3_dense_w4a8_strided` (`72fa20005`) | w4a8 的 strided-view layout 优化版（去 `.contiguous()`，NPU 实测变慢已回退） | 已推 fork (`origin`) |
| `backup-w4a8-pre-rebase-20260622` (`e586b832c7`) | PR #23650 **rebase 重建前**的完整旧历史（merge-base 落后、diff 重含已合并 W8A8） | 仅本地 |
| `backup-w4a8-with-int4-dynamic-20260622` (`8c4e5857f0`) | rebase 重建后、但**仍含被摘除的 INT4 `W4A8_DYNAMIC`** 那套（`ModelSlimW4A8Int8` + `NPUW4A8DynamicLinearMethod`）。日后要还原 INT4 直接从此 cherry-pick/checkout | 已推 fork (`origin`) |
| `backup-w4a8-pre-squash-20260625` (`a1947f3133`) | **本轮 squash+rebase 前**的多提交历史（16 个 commit：feature 初版 `8ff7856846` + 离线 A5 调试 fix/rewind/debug + 在线真 W4A8 修正）。要看调试过程或单独 commit 从此 checkout | 已推 fork (`origin`) |

当前 `junlin_qwen3_dense_w4a8` head 为 `b73c30235c`（`2efd90a6e5` 的 feature + PR #23650 review follow-up：全脱 torch_npu + 复用 npu_format_cast util + 文档，**已 squash 成单一 feature 提交并 force-push fork**，2026-07-02，**A5 已验证在线+离线可跑**）。

另有更早的 `backup-w4a8-pre-merge`（pre-merge 快照，已过时）。
