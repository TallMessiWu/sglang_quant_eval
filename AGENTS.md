# AGENTS.md

本仓库用于研究和实现 SGLang **Diffusion 侧**在华为 Ascend NPU 上的 MXFP8/MXFP4 量化适配（Wan2.2 等 Diffusion 模型）。
**如果涉及 LLM serving (`srt`) 侧的功能开发（如 MXFP8/MXFP4），请务必参考 `vllm-ascend` 的实现模式。**
*注：Qwen3 和 Qwen3.5 模型在 SGLang 内部共用底层 Linear/MoE 算子，因此量化实现代码完全一致。*

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion), [sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLM Qwen3)
- **Fork**: https://github.com/TallMessiWu/sglang

## 分支规则

| 分支                     | 说明                                                                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `junlin_diffusion_w8a8`               | 主线，**不动**                                                                                                                         |
| `junlin_diffusion_w4a4`         | Diffusion MXFP8 + MXFP4 在线量化                                                                                                             |
| `junlin_mxfp4_offline` | Diffusion 在 `junlin_diffusion_w4a4` 基础上增加 MXFP4 离线加载                                                                                      |
| `junlin_qwen3_dense_w8a8`   | LLM 侧，Qwen3 / 3.5 dense 模型 MXFP8 量化适配。**Tamir review follow-up（PR #22352 两条 comment）已开 draft [PR #28505](https://github.com/sgl-project/sglang/pull/28505)：① `ModelSlimMXFP8Scheme` 委托给 `NPUMXFP8LinearMethod`（`self.kernel`，按 weight dtype 分在线/离线分支，统一 `weight_scale_inv`）；② op 改走 `torch.ops.npu.*` + `torch.float8_*`，两文件去掉顶层 `import torch_npu` 与 `current_platform` 守卫。本分支已 rebase 到最新 upstream/main 后 force-push，PR diff 干净。NPU 在线+离线 e2e 已通过（两个验证门 raw `npu_quant_matmul` 吃 MX kwargs、`torch.float8_e8m0fnu` 存在均过），PR 已转正式。gemini-code-assist 两条 review 已处理并 resolve：① 模块级 `_FLOAT8_E8M0FNU_DTYPE` 改惰性 helper → **接纳**（起初判不接纳，因 `torch.float8_e8m0fnu` 是 torch 核心 dtype、import 时即可用且 e2e 已证非 None；但用户指出该模块也在非 NPU CI 被 import，惰性求值 Pareto-safe，遂接纳：commit `473389a4a9` 用 `_get_float8_e8m0fnu_dtype()` 在 `apply` 内求值）；② offline 分支删废参数 `weight_scale` → **接纳**（commit `f796a23606` 加 `del layer.weight_scale`）。PR head 现为 `473389a4a9`。** |
| `junlin_qwen3_dense_w4a8` | LLM 侧，Dense W4A8 在线量化（单级真 W4A8/MXFP8 激活，`--quantization mxfp_w4a8`）；离线 W4A8 已实现（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`）。**已 merge 同步至 `junlin_qwen3_dense_w8a8` 的 upstream 基线（2026-06-12）；bias-cache 优化保留，但 strided-view layout 优化因 NPU 实测变慢已回退恢复 `.contiguous()`（2026-06-16，strided 版存档于 `junlin_qwen3_dense_w4a8_strided`）。**<br>**PR [#23650](https://github.com/sgl-project/sglang/pull/23650)（W4A8，draft）在 #22352 squash 合并 + #28505 重构落地后已 rebase 重建（2026-06-22，旧历史存档于 `backup-w4a8-pre-rebase-20260622`）：`git reset --hard upstream/main` 去掉 diff 里重复的已合并 W8A8 代码，并对齐 #28505 风格——① `ModelSlimMXFP4W4A8Scheme` 改为 `subclass ModelSlimMXFP8Scheme`（W4A8_MXFP checkpoint 布局与 MXFP8 完全相同，零重复）；② `NPUMXFP4W4A8LinearMethod` 改 `torch.ops.npu.*`，无顶层 `import torch_npu`/`current_platform` 守卫，删掉死常量 `_FLOAT4_E2M1FN_X2_DTYPE`；在线 dual-level 离线 `.contiguous()` 保留。**<br>**再按 owner 决定，从本 PR 摘除越界的 INT4 `W4A8_DYNAMIC` 离线 scheme（`ModelSlimW4A8Int8` + `NPUW4A8DynamicLinearMethod`，非 MXFP，文档交付清单从未列入）→ 本 PR 现为纯 MXFP4 W4A8（在线 `mxfp4_w4a8_npu` + 离线 `W4A8_MXFP`），7 文件 510 行，head `2efd90a6e5`（2026-06-25 已再 rebase 到最新 upstream/main `efbe67d237` + squash 成单一 feature 提交）。被摘除的 INT4 那套完整存档在分支 `backup-w4a8-with-int4-dynamic-20260622`（本地 + 已推 fork，日后要还原直接 cherry-pick/checkout）。注意 modelslim 是多格式工具（INT8/INT4/MXFP8/MXFP4 由 checkpoint `quant_type` 分发，upstream 本就含 INT8/INT4 scheme）。本地无 torch/NPU，仅过 py_compile + pre-commit。**离线 W4A8 e2e 已在 A5（950）graph 模式验证输出正常（2026-06-25，torch_npu `2.10.0.post1.dev20260624`）；在线 W4A8 e2e 已在 A5 验证输出正常（2026-07-06，需 fp4 dtype 来源修复，见文末）。**<br>**在线 `mxfp4_w4a8_npu` 已从 dual-level（实为 W4A4，激活被压 FP4、精度差）改为单级真 W4A8（2026-06-25，原 commit `a1947f3133`，现已 squash 进单一提交 `2efd90a6e5`；旧多提交历史存于 `backup-w4a8-pre-squash-20260625`）：apply 完全复用离线 `npu_quant_matmul(x2_dtype=fp4, group_sizes=[0,0,32])`、激活走 `npu_dynamic_mx_quant(dst=float8_e4m3fn)`（FP8）；权重在线 `npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)` → `npu_format_cast(29)` + transpose，布局与离线一致。与离线唯一差别是权重来源（在线 RTN vs msmodelslim 校准）。在线 W4A8 精度待 A5 验。**⚠️ NPU 上默认走 graph 模式，别加 `--disable-cuda-graph`——eager decode 的 `ascend` attention（ATB `_npu_paged_attention`）会 `atb` 段错误（与量化无关，见「已知陷阱」②）；若确需 eager decode，加 `ASCEND_USE_FIA=1` 改走 aclnn FIA 绕开，A5 实测可跑。**<br>**PR #23650 review follow-up（2026-07-02，已 squash 进单一 feature 提交 `b73c30235c` 并 force-push fork；中间增量 commit `ba4735d321` 已被 squash 覆盖）：应 reviewer ping1jing2——① 复用并扩展 `hardware_backend/npu/utils.py:npu_format_cast`（新增 `customize_dtype`/`input_dtype` 转发 + FP4 跳过 `_is_nz_aligned` 的 ND fallback），online/offline 两处 W4A8 权重 cast 不再直调 `torch_npu.npu_format_cast`；② W4A8 online/offline 彻底去 `torch_npu`——op 全走 `torch.ops.npu.*`、dtype 走 `getattr(torch,...)` helper（`linear_method_npu.py` 新增 `_get_float4_e2m1fn_x2_dtype`），对齐 MXFP8 标杆，`linear_method_npu.py` + `utils.py` 均无 `import torch_npu`；③ online `create_weights` 加 docstring；④ `del quant_config, prefix` 保留（与 `ModelSlimMXFP8Scheme.__init__` 一致，仅回复不改）；⑤ 新 CLI choice `mxfp4_w4a8_npu` + 离线 `W4A8_MXFP` 已写入 `docs/advanced_features/quantization.md` + `docs/platforms/ascend/ascend_npu_quantization.md`。✅ **A5 已验证可跑（2026-07-02）**：3 个 `torch.ops.npu` 裸 op 带 kwarg（`npu_format_cast` 的 `customize_dtype/input_dtype`、`npu_quant_matmul` 的 `x2_dtype`、`npu_dynamic_mx_quant` 的 `round_mode`）均正常，`torch.float4_e2m1fn_x2` 在该 torch_npu（`2.10.0.post1.dev20260624`）下存在——**故 W4A8 可完全脱 `torch_npu`，op 走 `torch.ops.npu.*`、dtype 走 `getattr(torch,...)`，无需 guarded import（这条结论以后 W4A4/MoE 脱 torch_npu 可复用）**。gemini 那 6 条锚定 Diffusion `multimodal_gen` 的评论 rebase 后已 out-of-scope 关闭。**<br>**在线 W4A8 权重量化在 torch_npu `2.10.0.post2.dev20260704` 崩「output y must be same shape as input x」/561002。~~曾误诊为 FP4 kernel 被整体打回、改纯 torch RTN（commit `db4dee06ea`）~~——**2026-07-06 二次诊断推翻：真凶是 fp4 dtype 来源，NOT kernel**。NPU op 的 fp4 dtype 参数（`npu_dynamic_mx_quant(dst_type=)`/`npu_quant_matmul(x2_dtype=)`/`npu_format_cast(input_dtype=)`）只认 `torch_npu.float4_e2m1fn_x2`（int `296`），拒 `torch.float4_e2m1fn_x2`（torch dtype 对象）；生产 helper 当时 `getattr(torch,...)` 拿 dtype 对象 → 崩。A5 probe（同 shape 分别用 `torch_npu.float4`=296 vs `torch.float4` 当 dst_type）同机坐实 `dst=296` 全链路 PASS、`dst=torch.float4` FAIL，kernel 没坏。**修法**：`_get_float4_e2m1fn_x2_dtype()` 改 `is_npu()` 时优先 `getattr(torch_npu,"float4_e2m1fn_x2")`（惰性 import，不炸 CI），恢复 op 版 `process_weights_after_loading`、删纯 torch `_mxfp4_quantize_weight`（改动落在 op 版基线 `136b8b506c` 之上 + helper 一处）。**A5 e2e 已验证在线能跑、输出连贯（2026-07-06，用户实测）。****PR 文档已从 legacy `docs/` 迁到 `docs_new/docs/`（`.mdx`；upstream 已迁移目录，CI `lint` 的「reject changes under legacy docs/」会拒 legacy `docs/` 改动、连累 pr-gate）：加 `mxfp_w4a8` method-support 行 + Ascend ModelSlim 表 `MXFP4 W4A8` 行 + 精简「W4A8 for LLM dense」文字节，legacy `docs/` 两文件还原到 base（commit `cab582c88d`）。**详见「已知陷阱」FP4 dtype 来源条 + 下面 docs 迁移陷阱。 |
| `junlin_qwen3_dense_w4a4` | LLM 侧，Dense W4A4 在线量化（单级 MXFP4：4-bit 权重 + 4-bit 激活，`--quantization mxfp4`）+ 离线 W4A4（`W4A4_MXFP4` → `ModelSlimMXFP4Scheme`，真 MXFP4）。**PR [#23795](https://github.com/sgl-project/sglang/pull/23795)（W4A4）在 #22352 + #23650 均合并后已 rebase 重建（2026-07-07，旧历史存档于 `backup-w4a4-pre-rebase-20260707`，已推 fork）：`git reset --hard upstream/main`（`e85ef5487`，含 #23650 squash）去掉 diff 里重复的已合并 W8A8/W4A8 代码，只留 W4A4 delta（8 文件 442 行，单一 feature 提交 head `d831fb30e`），对齐 #23650 重构风格——① CLI 从 `mxfp4w4a4_npu` 改为 `mxfp4`（设备无关；srt 侧 `mxfp4` 只在 cpu/cuda/hip 注册给上游 `Mxfp4Config`/MoE，NPU 上没注册，故在 `__init__.py` 的 `is_npu()` 块加 `"mxfp4": Mxfp4W4A4Config`，镜像 `GPTQAscendConfig`——NPU 走我们的 W4A4、GPU 走上游，零冲突、不改上游 `Mxfp4Config`）；② config `NPUMxfp4W4A4Config` → 设备无关 `Mxfp4W4A4Config`，`get_quant_method` 按 `is_npu()` 分发；③ 在线 `NPUSingleLevelMXFP4LinearMethod` + 新增离线 `NPUSingleLevelMXFP4OfflineLinearMethod`（继承在线、共享 `apply`）全走 `torch.ops.npu.*`、复用惰性 helper `_get_float4_e2m1fn_x2_dtype()`（fp4 dtype 必来自 torch_npu）/`_get_float8_e8m0fnu_dtype()`，无顶层 `import torch_npu`；④ 离线 `ModelSlimMXFP4Scheme` 改为继承 `ModelSlimLinearScheme` + `self.kernel` 委托，注册进 `get_linear_scheme` 的 `linear_quant_schemes`（`("W4A4_MXFP4", ...)`）；⑤ 文档写 `docs_new/`（Ascend 表 `MXFP4 W4A4` 行 + 文字节 + method-support 表更新 `mxfp4` 行）。kernel 数学保持原样（单级 fp4，`npu_quant_matmul(x1=x2=fp4, group_sizes=[1,1,32])`、激活 `npu_dynamic_mx_quant(dst=fp4)`，无 FRACTAL_NZ；与 W4A8 的 `[0,0,32]`+NZ 是不同路径）。本地无 torch/NPU，仅过 py_compile + pre-commit；**W4A4 MXFP4 尚未 A5 e2e 验证，PR 标 WIP，NZ 布局待 A5 bring-up**。⚠️ 旧进度表曾把离线记为 INT4 `W4A4_DYNAMIC`，与分支实际（`W4A4_MXFP4` 真 MXFP4）不符，本次订正。** |
| `junlin_qwen3_moe_w8a8`   | **当前工作分支**，LLM 侧，Qwen3/3.5 MoE W8A8 MXFP8 在线量化（FusedMoE/TP，`--quantization mxfp8`）；EPMoE 待实现 |

> **备份分支（用于回退/还原，勿当工作分支）**：
> | 分支 | 内容 | 远端 |
> | ---- | ---- | ---- |
> | `junlin_qwen3_dense_w4a8_strided` (`72fa20005`) | w4a8 的 strided-view layout 优化版（去 `.contiguous()`，NPU 实测变慢已回退） | 已推 fork (`origin`) |
> | `backup-w4a8-pre-rebase-20260622` (`e586b832c7`) | PR #23650 **rebase 重建前**的完整旧历史（merge-base 落后、diff 重含已合并 W8A8） | 仅本地 |
> | `backup-w4a8-with-int4-dynamic-20260622` (`8c4e5857f0`) | rebase 重建后、但**仍含被摘除的 INT4 `W4A8_DYNAMIC`** 那套（`ModelSlimW4A8Int8` + `NPUW4A8DynamicLinearMethod`）。日后要还原 INT4 直接从此 cherry-pick/checkout | 已推 fork (`origin`) |
> | `backup-w4a8-pre-squash-20260625` (`a1947f3133`) | **本轮 squash+rebase 前**的多提交历史（16 个 commit：feature 初版 `8ff7856846` + 离线 A5 调试 fix/rewind/debug + 在线真 W4A8 修正）。要看调试过程或单独 commit 从此 checkout | 已推 fork (`origin`) |
>
> 当前 `junlin_qwen3_dense_w4a8` head 为 `b73c30235c`（`2efd90a6e5` 的 feature + PR #23650 review follow-up：全脱 torch_npu + 复用 npu_format_cast util + 文档，**已 squash 成单一 feature 提交并 force-push fork**，2026-07-02，**A5 已验证在线+离线可跑**）。**在线 CLI choice 已从 `mxfp4_w4a8_npu` 重命名为 `mxfp_w4a8`（2026-07-06，commit `136b8b506c`，已推 fork）：去 `_npu` 后缀（设备无关、对齐 `mxfp8`）＋ `mxfp_` 前缀反映全 MX 语义（MXFP4 权重 ＋ MXFP8 激活），与 INT4 权重的 `w4afp8` 区分；同步改 CHOICES／注册键／`get_name()`／docstring／`docs/advanced_features/quantization.md` 及测试脚本 `llm/qwen3_dense_online_w4a8.sh`；并把 config 类 `NPUMxfp4Config` 重命名为设备无关的 `Mxfp4W4A8Config`、`get_quant_method` 改按 `is_npu` 分发（非 NPU 抛 `NotImplementedError`）。**另有更早的 `backup-w4a8-pre-merge`（pre-merge 快照，已过时）。

## SGLang worktree 目录规则

SGLang 代码以 `git worktree` 容器形式放在 `sglang/` 下。**`sglang/diffusion_w8a8/` 是主仓子模块**（路径 `sglang/diffusion_w8a8`，指向 fork 的 `junlin_diffusion_w8a8` 分支，GitHub 上可点击跳转）。其余分支都是从它派生的 worktree（共享同一个 `.git`），被 `.gitignore` 精准忽略（非子模块、纯本地）。**需要修改哪个分支，就直接进入对应目录修改；不要在现有目录里用 `git checkout` 切分支。**

| 路径 | 对应分支 | 用途 |
| ---- | -------- | ---- |
| `sglang/diffusion_w8a8/` | `junlin_diffusion_w8a8` | **主 clone**（其余 worktree 由它派生）；Diffusion MXFP8 基线。 |
| `sglang/diffusion_w4a4/` | `junlin_diffusion_w4a4` | Diffusion MXFP4 主工作树（基于 `junlin_diffusion_w8a8`，含 MXFP8 + MXFP4 在线量化）。 |
| `sglang/qwen3_dense_w8a8/` | `junlin_qwen3_dense_w8a8` | LLM Qwen3/Qwen3.5 Dense W8A8 MXFP8 分支。 |
| `sglang/qwen3_dense_w4a8/` | `junlin_qwen3_dense_w4a8` | LLM Dense W4A8 分支。 |
| `sglang/qwen3_dense_w4a4/` | `junlin_qwen3_dense_w4a4` | LLM Dense W4A4 分支。 |
| `sglang/qwen3_moe_w8a8/` | `junlin_qwen3_moe_w8a8` | LLM MoE W8A8 MXFP8 分支。 |

开发约定：
- 改 Diffusion MXFP8 / `junlin_diffusion_w8a8` 基线相关内容：进入 `sglang/diffusion_w8a8/`
- 改 Diffusion MXFP4：进入 `sglang/diffusion_w4a4/`
- 改 Dense W8A8：进入 `sglang/qwen3_dense_w8a8/`
- 改 Dense W4A8：进入 `sglang/qwen3_dense_w4a8/`
- 改 Dense W4A4：进入 `sglang/qwen3_dense_w4a4/`
- 改 MoE W8A8：进入 `sglang/qwen3_moe_w8a8/`
- 若未来要维护 `junlin_mxfp4_offline` 等没有固定目录的分支，在 `sglang/` 下从主 clone（`sglang/diffusion_w8a8`）用 `git worktree add` 新建独立目录，不要复用已有 worktree checkout。
- 5 个 worktree 子目录（`diffusion_w4a4/`、`qwen3_dense_*/`、`qwen3_moe_w8a8/`）被 `.gitignore` 忽略；新增/删除 worktree 不影响主仓。**`sglang/diffusion_w8a8/` 是子模块**（主仓跟踪其 commit 指针）。

> **本地 vs fork 远端命名**：本地分支名已全部对齐 `junlin_<文件夹>`，fork（`TallMessiWu/sglang`）默认分支已改为 `junlin_diffusion_w8a8`。PR [#22352](https://github.com/sgl-project/sglang/pull/22352)（Dense W8A8）已于 2026-06-16 合并，fork 远端已同步改名为 `junlin_qwen3_dense_w8a8`，本地 tracking 已更新（2026-06-17）。注意：跨 fork 重命名 head 分支会关闭对应 PR（已踩坑），已合并后改名安全。

## 在线/离线量化模式

- **在线量化（Online）**：加载 FP16/BF16 权重，`process_weights_after_loading` 中实时量化，用 `--quantization mxfp8/mxfp4` 触发
- **离线量化（Offline ModelSlim）**：加载 msmodelslim 预量化权重，用 `--quantization modelslim` 触发，scheme 由 `quant_model_description.json` 自动检测

## 实现进度

| 功能                                   | 分支                   | 在线实现状态            | 离线实现状态            |
| -------------------------------------- | ---------------------- | ----------------------- | ----------------------- |
| Diffusion MXFP8                        | `junlin_diffusion_w8a8`             | ✅                      | ✅                      |
| Diffusion MXFP4                        | `junlin_diffusion_w4a4`       | ✅                      | ✅                      |
| LLM (Qwen3 & 3.5) Dense W8A8 (MXFP8)   | `junlin_qwen3_dense_w8a8` | ✅ (已对齐 vllm-ascend) | ✅ (已对齐 vllm-ascend) |
| LLM (Qwen3 & 3.5) Dense W4A8 (MXFP4/8) | `junlin_qwen3_dense_w4a8` | ✅ 在线已实现（`mxfp_w4a8`，单级真 W4A8/MXFP8 激活） | ✅ 离线已实现（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`，权重格式同 MXFP8：`float8_e4m3fn`） |
| LLM (Qwen3 & 3.5) Dense W4A4 (MXFP4)   | `junlin_qwen3_dense_w4a4` | ✅ 在线已实现（`mxfp4`，NPU 设备分发，单级 MXFP4；A5 e2e 待验证/WIP） | ✅ 离线已实现（`W4A4_MXFP4` → `ModelSlimMXFP4Scheme` → `NPUSingleLevelMXFP4OfflineLinearMethod`，真 MXFP4，权重 fp8 容器 `float8_e4m3fn`） |
| LLM (Qwen3 & 3.5) MoE W8A8 (MXFP8)     | `junlin_qwen3_moe_w8a8` | ✅ 在线已实现（`mxfp8`，`NPUMXFP8FusedMoEMethod`，仅 FusedMoE/TP；e2e 待 NPU 服务器验证） | ❌ 待实现               |
| LLM (Qwen3 & 3.5) MoE W4A8 (MXFP4/8)   | 待定                   | ❌ 待实现               | ❌ 待实现               |
| LLM (Qwen3 & 3.5) MoE W4A4 (MXFP4)     | 待定                   | ❌ 待实现               | ❌ 待实现               |

## 关键代码路径

> 注：下文 `sglang/python/...` 为 worktree 内相对路径简写，需按上面「worktree 目录规则」加对应前缀——Diffusion 量化代码在 `sglang/diffusion_w4a4/`（基线在 `sglang/diffusion_w8a8/`），LLM 量化代码在对应 `sglang/qwen3_*/` worktree（各分支只含自己那部分实现）。

### Diffusion 侧（multimodal_gen）

量化层目录：`sglang/python/sglang/multimodal_gen/runtime/layers/quantization/`

| 文件                          | 作用                                          |
| ----------------------------- | --------------------------------------------- |
| `__init__.py`               | 注册表 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` |
| `mxfp8_npu.py`              | MXFP8 在线量化                                |
| `mxfp4_npu.py`              | MXFP4 在线量化（双级）                        |
| `modelslim.py`              | ModelSlim 分发 `_get_scheme_from_parts()`   |
| `modelslim_mxfp8_scheme.py` | ModelSlim MXFP8 离线                          |
| `modelslim_mxfp4_scheme.py` | ModelSlim MXFP4 离线（双级）                  |

### LLM 侧（srt）

量化层目录：`sglang/python/sglang/srt/layers/quantization/modelslim/`

| 文件                                  | 作用                                                   |
| ------------------------------------- | ------------------------------------------------------ |
| `modelslim.py`                      | `ModelSlimConfig`：`get_quant_method` 分发、注册       |
| `schemes/modelslim_mxfp8.py`        | ModelSlim MXFP8 离线 scheme（W8A8）                    |
| `schemes/modelslim_mxfp4_w4a8.py`  | ModelSlim W4A8_MXFP 离线 scheme（权重 `float8_e4m3fn`，激活 FP8 动态量化） |
| `schemes/modelslim_w8a8_int8.py`    | ModelSlim W8A8 Int8 离线 scheme                        |

在线量化：
- `--quantization mxfp8` → `linear_method_npu.py` → `NPUMXFP8LinearMethod`（Linear 层）
- `--quantization mxfp8` (MoE 层) → `fp8.py:241` → `NPUMXFP8FusedMoEMethod`（仅 FusedMoE/TP；EPMoE 路径抛 `NotImplementedError`）
- `--quantization mxfp_w4a8` → `layers/quantization/npu_mxfp4.py` → `NPUMxfp4Config` → `NPUMXFP4W4A8LinearMethod`（真 W4A8：单级 FP8 激活 + FP4 权重，apply 复用离线 `npu_quant_matmul(x2_dtype=fp4)`；权重在线 `npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)`）
- `--quantization mxfp4`（NPU 设备分发，`is_npu()` 块注册 `Mxfp4W4A4Config`；非 NPU 为上游 `Mxfp4Config`/MoE）→ `layers/quantization/npu_mxfp4_w4a4.py` → `Mxfp4W4A4Config` → `NPUSingleLevelMXFP4LinearMethod`（单级 W4A4；离线 `W4A4_MXFP4` → `ModelSlimMXFP4Scheme` → `NPUSingleLevelMXFP4OfflineLinearMethod`）

其他关键文件：

- `srt/models/qwen3.py` — Qwen3 / 3.5 模型定义，`EntryClass = Qwen3ForCausalLM`
- `srt/models/qwen3_moe.py` — Qwen3 MoE 模型定义，`EntryClass = Qwen3MoeForCausalLM`
- `srt/hardware_backend/npu/quantization/moe_method_npu.py` — `NPUMXFP8FusedMoEMethod`（MoE 在线 MXFP8，三段式：`create_weights` / `process_weights_after_loading` / `apply`）
- `srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` — MoE NPU kernel 函数集（`npu_fused_experts_mxfp8` / `npu_fused_experts_w4a4` / `npu_fused_experts` 等）
- `srt/models/registry.py` — `ModelRegistry`，扫描 `sglang.srt.models` 注册所有 `EntryClass`
- `srt/layers/rotary_embedding/base.py` — RoPE 实现，NPU 路径 import `sgl_kernel_npu`
- `srt/model_loader/loader.py` — `DefaultModelLoader`：`_get_quantization_config` → `_initialize_model`
- `MindIE-SD/mindiesd/quantization/layer.py` — NPU 量化参考实现 (Diffusion)
- `vllm-ascend/vllm_ascend/quantization/methods/w8a8_mxfp8.py` — NPU 量化参考实现 (LLM)
- `msmodelslim/.../save/ascendv1.py` — MXFP4 权重导出格式

## 注意事项

- **CANN 版本**: MXFP8 需 ≥ 8.0.RC3；MXFP4 最低版本待确认
- **硬件**: Atlas 800I A2/A3（`DualLevelQuantBatchMatmul` 仅支持 Ascend 950，A2/A3 不支持）
- **CPU offload**：`dit_cpu_offload` 默认 True，`process_weights_after_loading` 中需手动 `.to("npu:X")` 后再调用量化 API
- **bias 精度**：量化 matmul 要求 bias 为 `float32`
- **tensor reshape**：diffusion 输入可能是 3D `[batch, seq, hidden]`，NPU 量化 API 需 2D，apply 中先 reshape 后 restore
- 与社区 YChange01 协调 MXFP8/MXFP4 工作分工（已在 Issue #14424 认领）

## 已知陷阱

- **量化不生效/乱码输出**：先验证模型是否注册成功。若 `sgl_kernel_npu` 某 kernel 不存在会导致模型模块 import 失败，`ModelRegistry` 静默跳过，fallback 到 HF Transformers（无量化感知），FP8 权重被当 BF16 解读 → 乱码。
  ```bash
  python3 -c "from sglang.srt.models.registry import ModelRegistry; print(list(ModelRegistry.models.keys()))"
  python3 -c "from sglang.srt.models.qwen3 import Qwen3ForCausalLM; print('OK')"
  ```
  修复：`sgl_kernel_npu` 非核心 kernel 的 import 改为 try/except + `None` fallback（见 `rotary_embedding/base.py`）。

- **模块级 `import torch_npu` 会炸掉全平台 CI**：`quantization/__init__.py` 无条件 import `ModelSlimConfig`，链路上任何文件顶层 `import torch_npu` 都会让 CUDA/CPU/AMD/XPU CI 在 import 时 `ModuleNotFoundError`（PR #22352 踩过两次：`linear_method_npu.py`、`modelslim_mxfp8.py`）。标准写法（见这两个文件）：
  ```python
  from sglang.srt.utils import is_npu
  _is_npu = is_npu()
  if _is_npu:
      import torch_npu
  ```
  模块级用到 `torch_npu` 属性的常量（如 `_FLOAT8_E8M0FNU_DTYPE`）也要用 `if _is_npu else` 三元守卫；函数体内的 `torch_npu.xxx` 调用只在 NPU 运行时执行，无需改。

  > ⚠️ **不要用 `current_platform.is_npu()` 做这个守卫**（旧写法，2026-06-16 已废弃）：新 upstream 把 `sglang/srt/platforms/` 重构成「插件发现的懒单例」，NPU 是 out-of-tree 插件，没装注册 `entry_point`（group `sglang.srt.platforms`）的 NPU 平台插件时，`current_platform` 会 fallback 到 base `SRTPlatform`、`is_npu()` 在**真 NPU 机器上也恒返回 False**（单例缓存、永久卡住）→ `torch_npu` 不 import → 量化哑掉。upstream 自己全用 util 版 `from sglang.srt.utils import is_npu`（直接 `torch.npu.is_available()`，核心文件如 `model_runner.py`/`model_loader/loader.py` 都是模块级 `_is_npu = is_npu()`），这才是可靠且与 upstream 一致的写法。

- **`process_weights_after_loading` 中 transpose 不加 `.contiguous()`**：`npu_grouped_matmul` 通过 strides 感知内存布局，`.contiguous()` 会物理重排数据破坏 block-scale 映射 → 乱码。用 `Parameter(qw.transpose(1, 2), requires_grad=False)` 直接包装非连续 view，不要在 transpose 后加 `.contiguous()`。

- **vllm-ascend MoE MXFP8/MXFP4 均为 offline**：`AscendW8A8MXFP8DynamicFusedMoEMethod`（`w8a8_mxfp8.py:178`）和 `AscendW4A4MXFP4DynamicFusedMoEMethod`（`w4a4_mxfp4.py:119`）的 `process_weights_after_loading` 只做 layout transform，没有 BF16→FP 在线转换。**MoE W4A8_MXFP 在 vllm-ascend 没有 MoE scheme**（`quant_parser.py` 注册了字符串但无对应类）。MoE 在线量化需自实现：online quant 参考 dense `NPUMXFP8LinearMethod`，routing pipeline 参考 `npu_fused_experts_w4a4`。

- **FusedMoE vs EPMoE quant method 共享，但 dispatch_output 类型不同**：`Fp8Config.get_quant_method` 对 FusedMoE 及其所有 EP 子类（`DeepEPMoE`/`NpuFuseEPMoE`/`MoriEPMoE`）返回同一个 method 实例。EPMoE 传入 `apply` 的不是 `StandardDispatchOutput`，需单独处理。当前 `NPUMXFP8FusedMoEMethod.apply` 仅支持 `StandardDispatchOutput`（TP-only），其他类型抛 `NotImplementedError`。

- **W4A8_MXFP checkpoint 权重格式**：`qwen3-8b-dense-w4a8` 检查点的权重存储为 `float8_e4m3fn`（非 packed FP4 uint8），shape 为 `[out, in]`，与 MXFP8 完全相同。这是 msmodelslim 旧版本导出格式（新版 `ascendv1.py` 会 `pack_fp4_to_uint8` → `uint8` shape `[out, in//2]`）。因此 `ModelSlimMXFP4W4A8Scheme` 的 `create_weights` 与 `ModelSlimMXFP8Scheme` 实现一致。

- **离线 W4A8 (`W4A8_MXFP`) 在 A5 上有两个不同报错，根因完全不同，别混为一谈**（`junlin_qwen3_dense_w4a8`，2026-06-24/25，A5 + `Qwen3-8B-mxw4a8-pack-full-0421`，graph 模式 e2e 输出已验证正常）：
  - **① prefill 阶段 `x2 should be in ... nz format, but it is 2` = 旧 torch_npu 的 FP4 `npu_quant_matmul` bug（量化相关，已靠升级解决）**：`NPUMXFP4W4A8OfflineLinearMethod` 照搬 vllm（`npu_format_cast(weight,29,customize_dtype=fp8,input_dtype=fp4)` → `.transpose(-1,-2)` → `npu_quant_matmul(x2_dtype=float4_e2m1fn_x2, group_sizes=[0,0,32])`）在 **`torch_npu 2.10.0.dev20260320`** 报此错；升级到 **`2.10.0.post1.dev20260624`** 后消失，NZ 写法正确。该 A5 强制 `allow_internal_format=False`（设 True 被打回、无 getter）但 NZ 仍能造出且 matmul OK，**不是阻塞点**；`npu_dynamic_mx_quant` 原生返回 3D block scale `[tokens, in//64, 2]`，无需 vllm `maybe_normalize_mxfp_scale_layout`（那个只 MoE 用）。代码保持 vllm 对齐的 NZ 写法（cast29+transpose），**不要切 ND**。
  - **② decode 阶段 `atb::OperationSetup` 段错误（伴 `Cannot create tensor with internal format while allow_internal_format=False`）= eager-decode 走 ATB `_npu_paged_attention`，与量化无关（已坐实根因 + 真修复，2026-06-25）**：升级 torch_npu 后改在 decode 崩。三段同步插桩定位：qkv 的 `npu_quant_matmul` 同步 OK、o_proj 的 matmul **还没跑**就在「进 o_proj 前的入口同步」崩 → fault 来自 qkv→o_proj 之间的 **decode attention**，不是 FP4 matmul。**根因：graph / eager 两模式在 `AscendAttnBackend` dispatch 到不同 attention 算子**——graph decode（`forward_decode_graph`，`ascend_backend.py:2256`）走 `npu_fused_infer_attention_score`（**aclnn** op，吃 ND，不建 internal 张量）；eager decode（`forward_decode` 默认分支 `ascend_backend.py:2617`，`use_fia=use_fa=False`）落到 `torch_npu._npu_paged_attention`（**ATB** op）。这台 A5 把 `allow_internal_format` 强制打回 False（`utils.py:114` 设 True 不生效），ATB `OperationSetup` 要建 FRACTAL_NZ internal workspace 张量、建不出来 → 段错误（报错命名空间 `atb::` 正对 ATB 算子，FIA 是 aclnn 不带此前缀）。**两条解法都已在 A5 验证可跑：**（a）**别加 `--disable-cuda-graph`**（默认 graph 模式，FIA，离线 W4A8 输出正常）；（b）若确需 eager decode，**`ASCEND_USE_FIA=1`** 让 eager 路径也改走 FIA（`ascend_backend.py:2552`，aclnn）绕开 `_npu_paged_attention` → **A5 实测能跑**（这是一整套 FIA 模式，`memory_pool_npu.py:67`/`npu_graph_runner.py:113` 同读此 flag、KV cache 布局随之变，非单纯换算子）。此 attention bug **非 W4A8 特有**（与 linear 量化无关、在线/离线同理），属独立 NPU attention 问题，不在本 PR 交付范围。
  - **误判史**：我中途基于旧 torch_npu 误加的 ND commit `4c1a0f5a0d` 已回退（②的 atb 段错误一度被我也归给 torch_npu 版本/ND，实为 eager-decode attention，已订正）。`--disable-cuda-graph` 是我早期排查时让用户加的，结果它本身才是触发 ② 的开关。

- **在线 W4A8 FP4 崩「output y must be same shape as input x」的真凶 = fp4 dtype 来源用错，NOT kernel regression**（`junlin_qwen3_dense_w4a8`，2026-07-06 二次诊断）：`mxfp_w4a8` 在线量化 `npu_dynamic_mx_quant(weight, dst_type=fp4, round_mode="round")` 在 torch_npu `2.10.0.post2.dev20260704`（CANN 9.1.0，A5）崩 561002。**~~此前误诊为「upstream FP4 kernel 被整体打回、只能纯 torch RTN 绕过」（commit `db4dee06ea`）——已推翻。~~** 真相：**NPU op 的 fp4 dtype 参数（`npu_dynamic_mx_quant(dst_type=)` / `npu_quant_matmul(x2_dtype=)` / `npu_format_cast(input_dtype=)`）只接受 `torch_npu.float4_e2m1fn_x2`（int enum，==`296`），拒绝 `torch.float4_e2m1fn_x2`（torch dtype 对象，存在但喂给 op 会在 op-plugin 层报错；传 `None` 报「Expected a value」）**。生产 `_get_float4_e2m1fn_x2_dtype()` 当时 `getattr(torch, "float4_e2m1fn_x2")` → 拿 torch dtype 对象 → 崩。A5 probe（同 shape 分别拿 `torch_npu.float4_e2m1fn_x2`=296 vs `torch.float4_e2m1fn_x2` 当 dst_type）在**同一台 post2 机器**坐实：`dst=296` 走完整 quant→format_cast→matmul 链 **PASS**（含 weight `[4096,4096]→[4096,2048]`），`dst=torch.float4` 在 quant 第一步 **FAIL**。**kernel 完全没坏。** fp8 不中招是因为 `torch.float8_e8m0fnu`(dtype 对象)作 `scale_dtype` 被接受——**只有 fp4 dtype 挑剔**。post1 能跑 post2 崩：升级收紧的是 op 对 fp4 dst_type 的**类型接受度**，不是 kernel。
  - **修法（恢复 op 版）**：`_get_float4_e2m1fn_x2_dtype()` 改为 `is_npu()` 时优先 `getattr(torch_npu, "float4_e2m1fn_x2")`（函数体内惰性 import torch_npu，不炸跨平台 CI），torch fallback；然后 `git checkout` 回 op 版 `process_weights_after_loading`（`npu_dynamic_mx_quant(dst=fp4)`）、删纯 torch `_mxfp4_quantize_weight`。在线+离线的 matmul/format_cast 一并受益（离线 post2 也没验过、同隐患）。**A5 e2e 已验证：op 版 + dtype 修复后在线 `mxfp_w4a8` 输出连贯（2026-07-06，用户实测）。**
  - **两条教训**：① 第一次误诊「kernel regression」正因 probe **没忠实复现生产的 dtype 来源**——probe v1 用 `torch_npu.float4`(296) 全 PASS、生产用 `torch.float4` 会崩，差异全在 dtype 从哪个模块取。**probe 要逐字节照抄生产 call，dtype 取自哪也算 call 的一部分**；「先 probe 再下结论」还不够，probe 本身必须忠实。② PR #23650 review 定的「W4A8 全脱 torch_npu、dtype 走 `getattr(torch,...)`」对 **fp4 是错的**——fp4 dtype 必须来自 torch_npu（fp8/e8m0 无所谓）。

- **sglang 文档已迁到 `docs_new/docs/`，改 legacy `docs/` 会被 CI 拒**（2026-07-06 踩，`junlin_qwen3_dense_w4a8` PR #23650）：upstream 把文档从 `docs/`（`.md`）迁到 **`docs_new/docs/`（`.mdx`）**，`lint` job 有个 step「reject changes under legacy docs/」用 `git diff origin/main...HEAD | xargs scripts/ci/check_no_docs_changes.py` 检出任何 `docs/` 下改动（allowlist 除外）就 fail，连累 pr-gate/pr-test-*。**改文档一律去 `docs_new/docs/` 对应位置**（路径也重构了，如 `docs/platforms/ascend/ascend_npu_quantization.md` → `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quantization.mdx`，且是 HTML `<table>` + JSX `style={{color:'green'}}`，非 md 表）。已在 legacy `docs/` 加过行的旧 PR 要**把 legacy 文件还原到 base**（`git checkout <merge-base> -- docs/...`，净 diff 不含 docs/ 即过）+ 在 docs_new 补等价内容。注意本地 `pre-commit run --files <legacy doc>` 会误报（hook `pass_filenames:false` 读 `git diff --cached`，只要 legacy 被 staged 就 fail）——以「净 diff 模拟」`git diff <base> --name-only \| xargs python3 scripts/ci/check_no_docs_changes.py` 为准。

- **MoE MXFP8 `npu_grouped_matmul` 必须显式传 `x_dtype` + `weight_dtype`**：仅传 `scale_dtype=FLOAT8_E8M0FNU_DTYPE` + `per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE` 不够——kernel 无法仅从 scale_dtype 推断「权重/激活是 fp8_e4m3fn 且 scales 是 UE8M0 block scale」，会走错 dequant 路径产生乱码。对齐 vllm-ascend `A5DeviceAdaptor.get_quant_gmm2_kwargs` (`device_op.py:460-466`)，gmm1/gmm2 都加：
  ```python
  x_dtype=torch_npu.float8_e4m3fn,
  weight_dtype=torch_npu.float8_e4m3fn,
  scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
  per_token_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
  ```
  注意：dense linear (`npu_quant_matmul`) **不** 需要 `x_dtype/weight_dtype`，它通过 `group_sizes=[1, 1, 32]` 显式带块大小，能从 tensor dtype 推断模式；MoE `npu_grouped_matmul` 没 `group_sizes`，必须靠 `x_dtype/weight_dtype` 显式声明。

- **MoE MXFP8 的 gmm1 需使用 fused gmm+swiglu+quant**：对齐 vllm-ascend A5 MXFP MoE 路径，gmm1 使用 `torch_npu.npu_grouped_matmul_swiglu_quant_v2`，而不是拆成 `npu_grouped_matmul` → `npu_swiglu` → `npu_dynamic_mx_quant`。该 fused op 的 `group_list` 需要从 count-style（`expert_tokens_num_type=1`）转换为 cumulative-style（`group_list.cumsum(dim=0)`）；gmm2 仍使用 `npu_grouped_matmul` 并保留 count-style `expert_tokens`。

- **MoE MXFP8 的 weight + scale 必须 `.transpose(1, 2)` 但 _不要_ `.contiguous()`**：`npu_grouped_matmul` (mx case) 通过 strides 感知 block-scale 布局；`.contiguous()` 会物理重排内存，但 kernel 仍按 strided-view 假设索引 → block-scale 映射错位 → 输出乱码。
  
  正确做法（对齐 vllm-ascend `AscendW8A8MXFP8DynamicFusedMoEMethod.process_weights_after_loading`，`w8a8_mxfp8.py:332-339`）：
  ```python
  layer.w13_weight = Parameter(qw13.transpose(1, 2), requires_grad=False)         # [E, H, 2I] strided view
  layer.w13_weight_scale = Parameter(s13.transpose(1, 2), requires_grad=False)    # [E, H//64, 2I, 2] strided view
  ```
  内存仍是 `[E, N, K]` / `[E, N, K_blk//2, 2]` 物理布局，但逻辑 shape 已变为 `[E, K, N]` / `[E, K_blk//2, N, 2]`——同时满足 kernel 的「n-dim 相等」和「transposition 一致」两个约束。

  `npu_dynamic_mx_quant` 对 2D 输入 `[N, K]` 直接吐 **3D** scale `[N, K_blk//2, 2]`（参考 `mxfp8_npu.py:144`，**不要手动 reshape**——已经 3D，stack 后是 4D，再 reshape 会 `too many values to unpack`）；`torch.stack` 拼出 weight 3D `[E, N, K]` + scale 4D `[E, N, K_blk//2, 2]`；transpose(1, 2) 后即为正确布局。
  
  注意：dense linear 路径 (`NPUMXFP8LinearMethod`) 用 `.transpose(0, 1).contiguous()` 是 OK 的，因为 `npu_quant_matmul` 接受 contig 布局；MoE 的 `npu_grouped_matmul` 不接受。
  
  踩坑历史：早先版本错误地加了 `.contiguous()`，跑通但输出乱码——纠正回 vllm-ascend 的 strided-view 布局后修复。

- **strided-view（去 `.contiguous()`）在 w4a8 硬件上反而变慢，已恢复 `.contiguous()`**（`junlin_qwen3_dense_w4a8`，2026-06-16）：strided weight/scale view（w8a8 上实测 -6.6% 提升）在 **w4a8 分支的 NPU 上端到端比更新前 w4a8 慢**（多条路径），且是「慢」非「乱码」。故 commit `9d6e9583e` 把两处 `.contiguous()` 恢复回来：MXFP8 dense（`NPUMXFP8LinearMethod`，merge 带来的 strided）+ W4A8 dual-level（`NPUMXFP4W4A8LinearMethod`，原 perf commit `33dfc0b9b` 去掉的）。**注：`NPUMXFP4W4A8LinearMethod` 的 dual-level 路径已于 2026-06-25（commit `a1947f3133`）整体替换为单级真 W4A8（apply 复用离线 NZ 路径），此条对它已不适用；MXFP8 dense 那半仍有效。****bias 缓存（`layer.bias_fp32`）是纯增益、保留**。strided 优化版存档在分支 **`junlin_qwen3_dense_w4a8_strided`**（72fa20005）。教训:`.contiguous()` 去留是「硬件/kernel 相关」,不要跨分支照搬 w8a8 的 layout 优化,需各自 NPU benchmark。

> 详细 API 参考和实现模式见 `/mxfp4-impl-ref` skill。

## 开发工具

- **pre-commit**：`sglang/` 下每个 worktree 都是独立 git checkout，pre-commit 必须进入**具体 worktree**（如 `sglang/diffusion_w4a4/`）后运行：
  ```bash
  pre-commit run --all-files  # 在对应 worktree 目录下执行，如 sglang/diffusion_w4a4/
  ```
  Windows 上 CI 脚本已修复编码和路径分隔符兼容性问题（`check_workflow_job_names.py`、`check_registered_tests.py`）。

## 网络访问限制（重要）

**Claude 的 Bash 环境到 github.com 的网络不稳定**：TLS 握手会间歇性失败（`gnutls_handshake() failed` / `SSL_ERROR_SYSCALL`），**大传输尤其容易失败**（实测：fork 的大 merge push 连续失败、`curl`/`fetch` 也失败；但主仓的小 commit push 一次成功）。所以这不是硬封锁，而是按传输大小/运气波动的不稳定连接。因此：

- **push 失败先重试几次**；持续失败（尤其是 fork 的大体量 push）就交给用户在自己的终端执行（用户侧网络稳定）。Claude 侧 `git push` 报 TLS 错 **不代表推送真的失败**——用户那边往往已成功；可用 `git log origin/<branch>`（依赖本地 remote-tracking ref）核对，但 TLS 不稳时 `fetch` 也可能刷新不了。
- **CI 日志 / GitHub Actions**：`gh` CLI **已认证可用**（account TallMessiWu），CI 失败时**先用 gh 拉真实日志再动手**，不要凭本地复现或臆测下结论：`gh pr checks <N> --repo sgl-project/sglang`、`gh run view --job <id> --log-failed`、`gh run watch <run-id> --exit-status`。注意 CI 跑的是 `pre-commit run --all-files`，ruff 的实际 autofix（如 UP037 给带 `from __future__ import annotations` 的文件去引号）可能与本地裸 `ruff --select=...` 不一致，会以「files were modified by this hook」失败。
- 需要联网（搜索、抓网页）时走 `web-access` skill，不要用裸 curl。

## 代码提交
代码提交时必须使用gitmoji-commit这个skill。每次提交代码后，更新 AGENTS.md 或相关 agent 指导文档。

### 子模块 / worktree 提交流程
1. **sglang 代码改动**：进入对应 worktree（见上「SGLang worktree 目录规则」）提交，并更新该 worktree 内的 agent 指导文档；推送到 fork（https://github.com/TallMessiWu/sglang）。`sglang/diffusion_w8a8/` 是主仓子模块，**主仓会跟踪其 commit 指针**——在对应 worktree 提交推送后，回到主仓 `git add sglang/diffusion_w8a8` 更新指针快照即可。其余 5 个 worktree 被 gitignore，主仓不跟踪。
2. 回到主仓，更新主仓 AGENTS.md（记录相关变更摘要）。
3. **参考子模块（MindIE-SD / msmodelslim / vllm-ascend）**：`.gitmodules` 已为各自配 `branch=`（dev / master / main）。需要同步上游时在主仓跑 `git submodule update --remote <name>`，再提交主仓记录新指针快照——git 子模块始终记录具体 commit，`branch=` 只是声明跟踪哪条上游分支、供 `--remote` 使用。

## Agent Team 协作模式（本分支：**禁用**）

本分支 **不** 启用 agent team 模式。新会话默认 **单 agent**，**不要** 主动调用 `TeamCreate` / 不要 spawn 命名子 agent / 不要走 planner-generator-evaluator 三角色流水线。即使用户说「开 team」之类的关键词，也请先确认是否要切到 `agent-team` 分支再启用。

如需 agent team 协作（planner → generator → evaluator，带「重生成」与「重规划」回环），请：
1. `git checkout agent-team`
2. 参见该分支 agent 指导文档末尾「Agent Team 协作模式（`pge-team`）」节

> 分支即模式，互不打扰。

## Agent skills

### Issue tracker

本仓库使用 **Local markdown** issue tracker：issue 作为 `.scratch/<feature>/` 下的 markdown 文件存在。详见 `docs/agents/issue-tracker.md`。

### Triage labels

5 个 canonical triage 角色采用中文字符串：待评估 / 待补充信息 / 可交付 agent / 需人工实现 / 不予处理。详见 `docs/agents/triage-labels.md`。

### Domain docs

单上下文布局：根 `CONTEXT.md` + `docs/adr/`。详见 `docs/agents/domain.md`。
