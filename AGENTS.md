# AGENTS.md

本仓库用于研究和实现 SGLang **Diffusion 侧**在华为 Ascend NPU 上的 MXFP8/MXFP4 量化适配（Wan2.2 等 Diffusion 模型）。
**如果涉及 LLM serving (`srt`) 侧的功能开发（如 MXFP8/MXFP4），请务必参考 `vllm-ascend` 的实现模式。**
*注：Qwen3 和 Qwen3.5 模型在 SGLang 内部共用底层 Linear/MoE 算子，因此量化实现代码完全一致。*

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion), [sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLM Qwen3)
- **Fork**: https://github.com/TallMessiWu/sglang

## 分支规则

Diffusion、Dense W8A8/W4A8 及 **Dense W4A4（PR #23795，2026-07-17 合入）** 侧功能均已合入上游 `sgl-project/sglang`（见下「已合并 PR」）。后续分支都基于 upstream/main rebase，**已含全部已合并代码**。当前本地有 **5 个活跃功能工作目录**：

| 分支 | 目录 | PR | 状态 |
| ---- | ---- | -- | ---- |
| `junlin_qwen3_moe_w8a8` | `sglang/qwen3_moe_w8a8/`（**主 clone**） | [#30768](https://github.com/sgl-project/sglang/pull/30768) | WIP OPEN，LLM MoE W8A8 MXFP8，在线+离线 A5 已 e2e 验证；OrangeRedeng 评审 **8 条全部落地**（⑧ NZ 已合入并回复，2026-07-21）。PR body 性能/精度数据已更新为 NZ 版。**2026-07-21 merge upstream/main（178 commit，`c0ed009f5`）解冲突，PR 已回到 MERGEABLE——⚠️ merge 后 A5 e2e 尚未重跑**。HEAD `e22b7bef8` |
| `junlin_qwen3.5_moe_w8a8` | `sglang/qwen3.5_moe_w8a8/`（派生 worktree） | [#32155](https://github.com/sgl-project/sglang/pull/32155) | 🚧 DRAFT，Qwen3.5 Dense/MoE W8A8 MXFP8 实验/验证。在线 Qwen3.5 MoE W8A8 A5 精度正常；离线 ModelSlim 的 Qwen3.5 GDN packed mapping 修复 `e6ffbc02a` 已经 A5 文本+图片 e2e 验证正常。视觉塔量化配置修复 `ccc9d841a` 已推 fork；首次 A5 图片请求暴露 K=4304 的 MXFP8 scale floor/pair 布局错误，`ceil(K/32)` 占位并补齐奇数 scale 的修复 `fc9cd5bad` 与 CPU 回归测试已推 fork，待重跑图片 e2e。PR 当前 stacked on #30768，待其合入后 rebase 清理 diff。**2026-07-21 rebase 到 `junlin_qwen3_moe_w8a8`（`e22b7bef8`）**，故也含未合并的 MoE W8A8 PR #30768 代码。HEAD `fc9cd5bad` |
| `junlin_qwen3.5_moe_w4a8` | `sglang/qwen3.5_moe_w4a8/`（派生 worktree） | 待创建 | 🚧 WIP，基于 `junlin_qwen3.5_moe_w8a8` `fc9cd5bad` 创建；已移植 Qwen3 MoE W4A8 在线+离线能力，并按当前基线让在线入口继承 `UnquantizedFusedMoEMethod`、为离线纯 scale scheme 注册空 offset。视觉 QKV 的 A8W4 bias 契约修复（BF16 `[1,N]`）`1ca00a86e`。**2026-07-24 A5 在线图片请求又暴露视觉 MLP `linear_fc2`（K=4304）不是 32 倍数导致 FP4 `npu_quant_matmul` 报 `k dim must to be aligned to 32`**；`Mxfp4W4A8Config.get_quant_method` 现对 K 非 32-对齐的 `LinearBase` 回退 BF16（对齐 msmodelslim/vllm-ascend 离线跳过行为），CPU 回归测试已补，提交 `7aa87ddfa` 推 fork，待 A5 重跑。HEAD `7aa87ddfa`。 |
| `junlin_qwen3.5_moe_w4a4` | `sglang/qwen3.5_moe_w4a4/`（派生 worktree） | 待创建 | 🚧 WIP，2026-07-24 基于 `junlin_qwen3.5_moe_w4a8` `7aa87ddfa` 创建；实现共享 Qwen MoE **W4A4 MXFP4**（单级，packed fp4 权重+fp4 激活），在线（`--quantization mxfp4` experts 分支）+离线（ModelSlim `W4A4_MXFP4`）。混合精度：只压 experts，非 expert 层保持 MXFP8/dual-level。gmm1 **融合** `npu_grouped_matmul_swiglu_quant_v2`（对齐 vllm-ascend `AscendW4A4MXFP4DynamicFusedMoEMethod`），gmm2 fp4；dispatcher 设 bf16 + gmm1 自量化 fp4（init_routing 无 fp4 quant_mode）。在线 Linear gate 补 K%32→BF16 回退（Qwen3.5-VL `linear_fc2` K=4304）。CPU 回归测试已补、`py_compile`+lint 通过。**A5 fp4-grouped-matmul 探针 + e2e 待验证**（我们栈首次让 fp4 激活过 grouped matmul；权重 FRACTAL_NZ vs ND 待实测）。HEAD `8d3d4cfb6`。 |
| `codex/fix-modelslim-mxfp4-packed-weight` | `sglang/fix_modelslim_mxfp4_packed/`（派生 worktree） | 待创建 | 🚧 基于 `upstream/main` `93cb9a548` 修复 Dense W4A4 offline ModelSlim 新版 packed checkpoint：`uint8 [out,in/2]` placeholder，移除 post-load 二次打包；CPU 回归测试已补，A5 e2e 待验证。HEAD `d875f6684`。 |

### 只存在于 fork 远程的分支（本地无目录）

| 分支 | 远程 HEAD | 说明 |
| ---- | --------- | ---- |
| `junlin_qwen3_moe_w4a8` | `924fea916` | LLM MoE W4A8 MXFP（MXFP4 权重 + FP8 激活），在线+离线已实现、**A5 未验证、PR 未创建**。2026-07-21 本地目录与分支已删，代码仅存于 fork。需要继续时从主 clone `git worktree add sglang/qwen3_moe_w4a8 junlin_qwen3_moe_w4a8` 拉回。 |
| `junlin_qwen3_dense_w4a4` | `ac07c7f9f` | PR #23795 已合并，代码已在 upstream/main。原为主 clone + 主仓子模块，2026-07-21 目录已删、子模块条目已从 `.gitmodules` 移除，无需再拉回。 |

### 已合并 PR（代码已在 upstream/main）

| PR | 内容 | 原 head 分支 | 合并日期 |
| -- | ---- | ----------- | -------- |
| [#20922](https://github.com/sgl-project/sglang/pull/20922) | Diffusion MXFP8（Wan2.2） | `junlin` | 2026-05-07 |
| [#24918](https://github.com/sgl-project/sglang/pull/24918) | docs：Diffusion MXFP8 | `junlin` | 2026-05-11 |
| [#22338](https://github.com/sgl-project/sglang/pull/22338) | Diffusion MXFP4 | `junlin_mxfp4` | 2026-05-19 |
| [#25904](https://github.com/sgl-project/sglang/pull/25904) | docs：Diffusion MXFP4 | `codex/diffusion-mxfp4-docs` | 2026-05-25 |
| [#22352](https://github.com/sgl-project/sglang/pull/22352) | Qwen3 Dense W8A8 MXFP8 | `junlin_qwen3_dense` | 2026-06-16 |
| [#28505](https://github.com/sgl-project/sglang/pull/28505) | Dense MXFP8 重构（委托 kernel + `torch.ops.npu`） | `junlin_qwen3_dense_w8a8` | 2026-06-17 |
| [#23650](https://github.com/sgl-project/sglang/pull/23650) | Qwen3 Dense W4A8 MXFP | `junlin_qwen3_dense_w4a8` | 2026-07-06 |
| [#23795](https://github.com/sgl-project/sglang/pull/23795) | Qwen3 Dense W4A4 MXFP4（在线双级 + 离线单级） | `junlin_qwen3_dense_w4a4` | 2026-07-17 |

> 活跃分支的详细状态（commit hash、A5 验证节点、备份分支清单）见 [docs/branches.md](docs/branches.md)。

## SGLang worktree 目录规则

SGLang 代码以 `git worktree` 形式放在 `sglang/` 下，各目录**共享同一个 `.git`**：`sglang/qwen3_moe_w8a8/` 是**主 clone**（持有真正的 `.git` 目录），其余目录是从它派生的 worktree。**整个 `sglang/` 目录已被 `.gitignore` 忽略——主仓不再跟踪任何 sglang 子模块**（旧的 `sglang/qwen3_dense_w4a4` 子模块已于 2026-07-21 移除）。**需要修改哪个分支，就直接进入对应目录修改；不要在现有目录里用 `git checkout` 切分支。**

| 路径 | 对应分支 | 用途 |
| ---- | -------- | ---- |
| `sglang/qwen3_moe_w8a8/` | `junlin_qwen3_moe_w8a8` | **主 clone**（持有共享 `.git`）；LLM MoE W8A8 MXFP8（PR #30768）。 |
| `sglang/qwen3.5_moe_w8a8/` | `junlin_qwen3.5_moe_w8a8` | 派生 worktree；Qwen3.5 Dense/MoE W8A8 MXFP8 实验/验证。 |
| `sglang/qwen3.5_moe_w4a8/` | `junlin_qwen3.5_moe_w4a8` | 派生 worktree；Qwen3/Qwen3.5 共享 MoE W4A8 MXFP 在线+离线适配。 |
| `sglang/qwen3.5_moe_w4a4/` | `junlin_qwen3.5_moe_w4a4` | 派生 worktree；Qwen3/Qwen3.5 共享 MoE **W4A4 MXFP4** 在线+离线适配（单级、混合精度只压 experts）。 |
| `sglang/fix_modelslim_mxfp4_packed/` | `codex/fix-modelslim-mxfp4-packed-weight` | 派生 worktree；修复 Dense W4A4 offline 新版 ModelSlim packed checkpoint 加载。 |
| `sglang/32745_platform_rework/` | `codex/32745-platform-rework` | 派生 worktree；PR [#32745](https://github.com/sgl-project/sglang/pull/32745)，Gemma RMSNorm 改走 `sgl_kernel_npu` 稳定 API（配对 sgl-kernel-npu#638）。 |

开发约定：
- 改 MoE W8A8：进入 `sglang/qwen3_moe_w8a8/`（NZ 已在其中）；改 Qwen3.5 Dense/MoE W8A8：进入 `sglang/qwen3.5_moe_w8a8/`；改 Qwen3/Qwen3.5 MoE W4A8：进入 `sglang/qwen3.5_moe_w4a8/`。
- 改 Dense W4A4 offline packed checkpoint 修复：进入 `sglang/fix_modelslim_mxfp4_packed/`。
- 已合并的 Diffusion / Dense W8A8/W4A8/W4A4 代码都在 upstream/main（两个目录 rebase 后均含），无需单独 checkout。要基于某个远程分支（如 `junlin_qwen3_moe_w4a8`）继续开发，从主 clone 用 `git worktree add sglang/<名字> <分支>` 新建独立目录，不要复用已有 worktree。
- **动 worktree 前先跑 `git worktree list` 确认实际状态**，别照本表假设——本表曾因目录被手工删除而失真（2026-07-21 已订正）。注意 `git worktree list` 必须在 `sglang/` 下的目录里跑，在主仓跑只会列出主仓自己。旧 `sglang/diffusion_w8a8` 子模块随 Diffusion 合并上游后已移除。

> **命名与陷阱**：本地分支名对齐 `junlin_<文件夹>`；fork（`TallMessiWu/sglang`）默认分支为 `junlin_diffusion_w8a8`。注意：跨 fork 重命名 **未合并** PR 的 head 分支会关闭对应 PR（已踩坑），已合并后改名才安全。

## 在线/离线量化模式

- **在线量化（Online）**：加载 FP16/BF16 权重，`process_weights_after_loading` 中实时量化，用 `--quantization mxfp8/mxfp4` 触发
- **离线量化（Offline ModelSlim）**：加载 msmodelslim 预量化权重，用 `--quantization modelslim` 触发，scheme 由 `quant_model_description.json` 自动检测

## 实现进度

| 功能                                   | 归属                   | 在线实现状态            | 离线实现状态            |
| -------------------------------------- | ---------------------- | ----------------------- | ----------------------- |
| Diffusion MXFP8                        | 已合并 #20922          | ✅                      | ✅                      |
| Diffusion MXFP4                        | 已合并 #22338          | ✅                      | ✅                      |
| LLM (Qwen3 & 3.5) Dense W8A8 (MXFP8)   | 已合并 #22352 / #28505 | ✅ (已对齐 vllm-ascend) | ✅ (已对齐 vllm-ascend) |
| LLM (Qwen3 & 3.5) Dense W4A8 (MXFP4/8) | 已合并 #23650          | ✅ 在线（`mxfp_w4a8`，单级真 W4A8/MXFP8 激活） | ✅ 离线（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`，权重为 packed FP4：`uint8 [out,in/2]`） |
| LLM (Qwen3 & 3.5) Dense W4A4 (MXFP4)   | 已合并 #23795；packed 修复进行中 | ✅ 在线已实现（`mxfp4`，NPU 设备分发，**双级 MXFP4** `NPUDualLevelMXFP4LinearMethod`；A5 e2e 已验证，双级修复了单级 RTN 贪心死循环） | 🚧 原实现兼容旧版 `float8_e4m3fn [out,in]` checkpoint；当前 ModelSlim 导出为 packed `uint8 [out,in/2]`。`codex/fix-modelslim-mxfp4-packed-weight` 已修 placeholder 与 post-load 二次打包，A5 e2e 待验证。 |
| LLM (Qwen3 & 3.5) MoE W8A8 (MXFP8)     | `junlin_qwen3_moe_w8a8`（#30768 WIP） | ✅ 在线已实现（`mxfp8`，`NPUMXFP8OnlineMoEMethod`（继承 `UnquantizedFusedMoEMethod`），A5 e2e 已验证） | ✅ 离线已实现（`W8A8_MXFP8` → `ModelSlimMXFP8MoEScheme` → `NPUMXFP8MoEMethod` 离线分支，A5 e2e 已验证） |
| LLM (Qwen3 & 3.5) MoE W4A8 (MXFP4/8)   | `junlin_qwen3.5_moe_w4a8`（HEAD `7aa87ddfa`，基于 `fc9cd5bad`；旧实现来源 `origin/junlin_qwen3_moe_w4a8` `924fea916`） | ✅ 在线（`mxfp_w4a8` → `NPUMXFP4W4A8FusedMoEMethod`，继承 `UnquantizedFusedMoEMethod` 并换入 W4A8 per-GMM kernel；视觉 MLP K=4304 不对齐层已回退 BF16，`7aa87ddfa`；A5 e2e 待重跑） | ✅ 离线（`W4A8_MXFP` → `ModelSlimMXFP4W4A8MoEScheme` → `NPUMXFP4W4A8MoEMethod`，packed FP4 `uint8` 权重且无 offset；A5 e2e 待验证） |
| LLM (Qwen3 & 3.5) MoE W4A4 (MXFP4)     | `junlin_qwen3.5_moe_w4a4`（HEAD `8d3d4cfb6`，基于 `7aa87ddfa`） | 🚧 在线已实现（`mxfp4` experts 分支 → `NPUMXFP4W4A4FusedMoEMethod` → `NPUMXFP4W4A4MoEMethod`，单级 packed fp4 + fp4 激活，融合 gmm1；**A5 探针/e2e 待验证**） | 🚧 离线已实现（`W4A4_MXFP4` → `ModelSlimMXFP4W4A4MoEScheme` → `NPUMXFP4W4A4MoEMethod`，packed fp4 `uint8` 无 offset；**A5 e2e 待验证**） |

## 关键代码路径

> 注：下文 `sglang/python/...` 为 worktree 内相对路径简写，需加对应目录前缀（如 `sglang/qwen3_moe_w8a8/`）。已合并的 Diffusion / Dense 量化代码在 upstream/main，各 worktree 均含；各未合并分支只额外含自己那部分实现。

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
| `schemes/modelslim_mxfp4.py`        | ModelSlim W4A4_MXFP4 离线 scheme（packed `uint8 [out,in/2]`；激活 MXFP4） |
| `schemes/modelslim_mxfp4_w4a8.py`  | ModelSlim W4A8_MXFP 离线 scheme（packed FP4 权重 `uint8 [out,in/2]`，激活 FP8 动态量化） |
| `schemes/modelslim_w8a8_int8.py`    | ModelSlim W8A8 Int8 离线 scheme                        |

在线量化：
- `--quantization mxfp8` → `linear_method_npu.py` → `NPUMXFP8LinearMethod`（Linear 层）
- `--quantization mxfp8` (MoE 层) → `fp8.py` → `hardware_backend/npu/quantization/online_moe_methods.py` → `NPUMXFP8OnlineMoEMethod`（在线 MXFP8；**继承 `UnquantizedFusedMoEMethod`**，只 override `create_moe_runner` 换上 `NPUMXFP8MoEMethod("w13"/"w2")` kernel，`create_weights` / `process_weights_after_loading` / `apply` 全部继承）。单独成文件是因为 `unquant.py` 顶层 import 了 `moe_methods.py`，放同一文件会成 import 环
- `--quantization mxfp_w4a8` (MoE 层) → `npu_mxfp4.py` → `hardware_backend/npu/quantization/online_moe_methods.py` → `NPUMXFP4W4A8FusedMoEMethod`（继承 `UnquantizedFusedMoEMethod`，只替换为 `NPUMXFP4W4A8MoEMethod` per-GMM kernel；三段式，仅 FusedMoE/TP）
- `--quantization mxfp4` (MoE experts 层) → `npu_mxfp4_w4a4.py` → `Mxfp4W4A4Config` 的 FusedMoE 分支 → `online_moe_methods.py` → `NPUMXFP4W4A4FusedMoEMethod`（继承 `UnquantizedFusedMoEMethod`，换入 `NPUMXFP4W4A4MoEMethod("w13"/"w2")` per-GMM kernel）。**单级** packed fp4 权重 + fp4 激活；w13 走**融合** gmm1 `npu_grouped_matmul_swiglu_quant_v2`（`quant_dtype=fp4`，同 MXFP8 融合形状而非 W4A8 非融合），w2 走 fp4 `npu_grouped_matmul`；dispatcher 设 `bf16`（init_routing 无 fp4 quant_mode）→ gmm1 自量化 fp4。TP-only。权重处理复用 W4A8 packed-fp4 helper。非 expert Linear 仍走 `NPUDualLevelMXFP4LinearMethod`（混合精度）
- `--quantization mxfp_w4a8` → `layers/quantization/npu_mxfp4.py` → `NPUMxfp4Config` → `NPUMXFP4W4A8LinearMethod`（真 W4A8：单级 FP8 激活 + FP4 权重，apply 复用离线 `npu_quant_matmul(x2_dtype=fp4)`；权重在线 `npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)`）
- `--quantization mxfp4`（NPU 设备分发，`is_npu()` 块注册 `Mxfp4W4A4Config`；非 NPU 为上游 `Mxfp4Config`/MoE）→ `layers/quantization/npu_mxfp4_w4a4.py` → `Mxfp4W4A4Config` → **`NPUDualLevelMXFP4LinearMethod`（在线唯一路径，双级 MXFP4：细 FP8 E4M3 L0 scale + 粗 L1 scale，`npu_dynamic_dual_level_mx_quant` + `npu_dual_level_quant_matmul`，权重 FRACTAL_NZ，仅 A5/Ascend 950）**。单级 `NPUSingleLevelMXFP4LinearMethod` 仅作离线基类保留（离线 `W4A4_MXFP4` → `ModelSlimMXFP4Scheme` → `NPUSingleLevelMXFP4OfflineLinearMethod`，单级）。移植自 Diffusion `NPUMXFP4DiffusionLinearMethod`/MindIE-SD `W4A4MXFP4DualQuantLinear`

其他关键文件：

- `srt/models/qwen3.py` — Qwen3 / 3.5 模型定义，`EntryClass = Qwen3ForCausalLM`
- `srt/models/qwen3_moe.py` — Qwen3 MoE 模型定义，`EntryClass = Qwen3MoeForCausalLM`
- `srt/hardware_backend/npu/quantization/moe_methods.py` — 所有 per-gmm MoE kernel 类（`NPUUnquantMoEMethod`、`NPUMXFP8MoEMethod`、`NPUMXFP4W4A8MoEMethod`、`NPUMXFP4W4A4MoEMethod`、`NPUW4A4Int4MoEMethod`、`NPUW8A8Int8MoEMethod`、`NPUW4A8Int8MoEMethod` 等）。`NPUMXFP4W4A4MoEMethod` 复用 `NPUMXFP4W4A8MoEMethod` 的 packed-fp4 权重 helper（`_quantize_weight_online`/`_process_weight_fp4`/`_process_scale_fp4`），只在激活侧不同（fp4 + 融合 gmm1）
- `srt/hardware_backend/npu/quantization/online_moe_methods.py` — 在线量化 FusedMoE 入口类（`NPUMXFP8OnlineMoEMethod`、`NPUMXFP4W4A8FusedMoEMethod`、`NPUMXFP4W4A4FusedMoEMethod`），与 `moe_methods.py` 分开以避开 `unquant.py` 的 import 环
- `srt/hardware_backend/npu/moe/matmul.py` — MoE matmul kernel wrapper（`GroupedMatmul`、`GroupedMatmulSwigluQuant`（gmm1 融合 gate/up+swiglu+requant，返回 (激活, block scale)））
- `srt/hardware_backend/npu/moe/quant.py` — MoE 量化 kernel wrapper（`HiddenStatesDynamicQuant`：int8/quint4x2 走 `npu_dynamic_quant`，`float8_e4m3fn` 走 `npu_dynamic_mx_quant`）。**原名 `hidden_states_quant.py`，按评审建议改名**
- `srt/hardware_backend/npu/moe/activation.py` — MoE 激活函数库（`NPUSwiglu`、`NPUSwigluQuant`、`NPUSwigluMXFP8Quant`、`NPUSwigluDeepEPKernel` 等）
- `srt/layers/moe/moe_runner/ascend.py` — Ascend MoE runner 编排（gmm1→activation→gmm2），含 W4A8 MXFP 非融合分支
- `srt/layers/quantization/modelslim/schemes/modelslim_mxfp4_w4a8_moe.py` — `ModelSlimMXFP4W4A8MoEScheme`（离线 W4A8_MXFP MoE）
- `srt/layers/quantization/modelslim/schemes/modelslim_mxfp4_w4a4_moe.py` — `ModelSlimMXFP4W4A4MoEScheme`（离线 `W4A4_MXFP4` MoE，packed fp4 `uint8` + e8m0，无 offset，委托 `NPUMXFP4W4A4MoEMethod`）
- `srt/models/registry.py` — `ModelRegistry`，扫描 `sglang.srt.models` 注册所有 `EntryClass`
- `srt/layers/rotary_embedding/base.py` — RoPE 实现，NPU 路径 import `sgl_kernel_npu`
- `srt/model_loader/loader.py` — `DefaultModelLoader`：`_get_quantization_config` → `_initialize_model`
- `MindIE-SD/mindiesd/quantization/layer.py` — NPU 量化参考实现 (Diffusion)
- `vllm-ascend/vllm_ascend/quantization/methods/w8a8_mxfp8.py` — NPU 量化参考实现 (LLM)
- `msmodelslim/.../save/ascendv1.py` — MXFP4 权重导出格式
- `sgl-kernel-npu/` — 上游 NPU kernel 仓（`sgl_kernel_npu`，RoPE / triton norm 等；2026-07-21 起为主仓子模块，跟踪 `main`）。缺 kernel 会导致 import 失败 → 静默 fallback HF Transformers → 乱码，见「已知陷阱」首条。
- `docs/npu-api/DualLevelQuantBatchMatmul.md`、`docs/npu-api/DynamicDualLevelMxQuant.md` — Ascend 双级量化 kernel API 参考（原在根目录，现归入 `docs/npu-api/`）

## 注意事项

- **CANN 版本**: MXFP8 需 ≥ 8.0.RC3；MXFP4 最低版本待确认
- **硬件**: Atlas 800I A2/A3（`DualLevelQuantBatchMatmul` 仅支持 Ascend 950，A2/A3 不支持）
- **CPU offload**：`dit_cpu_offload` 默认 True，`process_weights_after_loading` 中需手动 `.to("npu:X")` 后再调用量化 API
- **bias 精度/shape**：多数量化 matmul 使用 `float32` bias；A8W4 `npu_quant_matmul`（MXFP4 权重 + MXFP8 激活）是例外，要求 BF16 二维 bias `[1,N]`。Qwen3 文本层通常无 bias，Qwen3.5 视觉 QKV 会触发该契约
- **tensor reshape**：diffusion 输入可能是 3D `[batch, seq, hidden]`，NPU 量化 API 需 2D，apply 中先 reshape 后 restore
- 与社区 YChange01 协调 MXFP8/MXFP4 工作分工（已在 Issue #14424 认领）

## 已知陷阱

- **量化不生效/乱码输出**：先验证模型注册（`ModelRegistry.models.keys()`），`sgl_kernel_npu` 缺失 kernel → import 失败 → 静默 fallback HF Transformers → 乱码。修复：非核心 kernel import 改 try/except + `None`。
- **模块级 `import torch_npu` 炸全平台 CI**：用 `from sglang.srt.utils import is_npu; if is_npu(): import torch_npu` 守卫，**不要**用 `current_platform.is_npu()`（NPU 插件未装时恒 False）。
- **transpose 不加 `.contiguous()`**：`npu_grouped_matmul` 靠 strides 感知 block-scale 布局，`.contiguous()` 物理重排 → 乱码。dense 路径 `.contiguous()` 则 OK。
- **vllm-ascend MoE 量化均为 offline**：无 BF16→FP 在线转换，MoE 在线量化需自实现。
- **FusedMoE vs EPMoE dispatch_output 类型不同**：当前仅支持 `StandardDispatchOutput`（TP-only）。
- **W4A8_MXFP 权重是 packed FP4**：当前 ModelSlim `ascendv1.py` 用 `pack_fp4_to_uint8` 导出 `uint8 [out,in/2]`；`ModelSlimMXFP4W4A8Scheme.create_weights` 必须按相同物理 shape/dtype 注册，post-load 只做 FRACTAL_NZ layout 转换和 transpose，不能再次打包。
- **离线 W4A8 A5 两报错**：① prefill NZ format 错 → 升级 torch_npu 到 `2.10.0.post1` 解决；② decode ATB 段错误 → 与量化无关，别加 `--disable-cuda-graph`（或用 `ASCEND_USE_FIA=1`）。
- **A5 默认 ATB 注意力算子崩溃（warmup 固定挂，与量化无关）**：`--device npu` 在 A5(Ascend 950) 上默认 prefill 走 ATB `SelfAttentionOperation`（`_npu_flash_attention_qlens`，`ascend_backend.py:1472`）；服务起来后 SGLang 自动 warmup 请求（一次 prefill）触发 `RuntimeError: SelfAttentionOperation CreateOperation failed!`。`CreateOperation` 是 ATB 按 SoC 能力表构图/校验的步骤，A5 不支持该算子（decode 默认 `_npu_paged_attention` 同源）；A2/A3(910B/C) 支持故不受影响——**崩溃是 A5 特有**。修复：`export ASCEND_USE_FIA=1`，让 prefill+decode 都走 A5 原生 FIA 算子 `npu_fused_infer_attention_score`（`forward_extend` 走 `use_fia` 分支，永不到达 `_npu_flash_attention_qlens`）。`llm/` 全部 serve 脚本已默认带此 env。**别**用注释掉 `qk_head_dim<=128 and ...` 条件强制 `False`→native SDPA 的脏改（慢、无融合 kernel、且只补 prefill）。与上条 decode ATB 段错误同根。
- **在线 W4A8 FP4 dtype 来源**：fp4 dtype 参数必须来自 `torch_npu.float4_e2m1fn_x2`（int 296），**不能**用 `torch.float4_e2m1fn_x2`（torch dtype 对象会被 op 拒绝）。`_get_float4_e2m1fn_x2_dtype()` 在 NPU 时优先 `getattr(torch_npu, ...)`。
- **在线 W4A8 的 K 必须 32-对齐，不对齐层回退 BF16**：FP4（A8W4）`npu_quant_matmul`（packed FP4 权重 + group_size=32 block scale）**硬性要求 reduction 维 K 是 32 的倍数**，无末尾 partial block 支持——这与 MXFP8（A8W8）不同（MXFP8 能吃 K=4304，靠 `ceil(K/32)` scale + pad 成 pair，见 `NPUMXFP8LinearMethod`）。Qwen3.5 视觉 MLP `linear_fc2` 的 K=4304（4304/32=134.5）会让 kernel 报 `AclNN_Parameter_Error: the k dim must to be aligned to 32, which is 4304`。vllm-ascend 的 W4A8 是纯离线，这类层由 msmodelslim 导出时留进 `ignored_layers` 不量化；在线路径需自己复刻：`Mxfp4W4A8Config.get_quant_method`（`npu_mxfp4.py`）对 `layer.input_size % 32 != 0` 的 `LinearBase` 返回 `UnquantizedLinearMethod`（打 warning）。检查放在 `is_npu()` 之前，故可在 CPU CI 测（`test_npu_mxfp4_w4a8_linear.py`）。LLM 各层维度本就 32-对齐、视觉塔 tp=1，不会误跳。
- **sglang 文档迁到 `docs_new/docs/`**：改 legacy `docs/` 会被 CI lint 拒，新文档一律写 `docs_new/docs/`（`.mdx`）。
- **MoE `npu_grouped_matmul` 需显式 `x_dtype` + `weight_dtype`**：缺少则 kernel 走错 dequant 路径 → 乱码。dense `npu_quant_matmul` 不需要（有 `group_sizes`）。
- **MoE gmm1 用 fused `npu_grouped_matmul_swiglu_quant_v2`**：勿拆三步；`group_list` 需 count→cumulative 转换。
- **MoE weight+scale `.transpose(1,2)` 不要 `.contiguous()`**：真实机制是 gmm1 的 `CheckMXTranspose` 断言——**weight 与 weight_scale 的 transpose 标志必须一致**，只动一边直接报错（非乱码）。两边都 contiguous 数值正确但更慢（128-expert decode −6.2%）。dense 路径 contig 则 OK。**注意小 expert 数 micro-bench 结论相反**（E=4 时 contiguous 快 58%），定 layout 必须用真实 expert 数。
- **MoE MXFP8 转 FRACTAL_NZ 必须 cast 在 transpose 之前**：`npu_format_cast` 产出非 transposed 张量，先 transpose 再 cast 会触发上条断言。正确顺序同 dense W4A8（`linear_method_npu.py:443`/`:578`），**不能**照抄同文件 int8 MoE 的 `npu_format_cast(w.transpose(1,2))`（int8 无 MX scale 要同步）。A5 实测 decode +1.4% / prefill +3.8%（一次性 kernel 探针跑出，脚本已删）。
- **strided-view 在 w4a8 上变慢**：已恢复 `.contiguous()`。`.contiguous()` 去留是硬件相关的，勿跨分支照搬。
- **MXFP8 MoE kernel 契约（torch_npu 2.10.0.post2 + A5，已探针验证）**：
  - `npu_dynamic_mx_quant(x[N,K], dst=float8_e4m3fn)` → scale `[N, K//64, 2]` uint8（**3D pair-split**，无需 normalize）；**3D 输入 [E,N,K] 被 kernel 直接接受** → 免逐 expert 循环 + uint8 stack。
  - gmm1 `npu_grouped_matmul_swiglu_quant_v2`：weight/scale 单元素 list，group_list **cumulative**；`x_dtype=None, weight_dtype=None`（e4m3 隐式），`weight_scale_dtype=x_scale_dtype=float8_e8m0fnu`（**必须显式**）。
  - gmm2 `npu_grouped_matmul`：weight/scale 单元素 list，group_list **COUNT + 显式 `group_list_type=1`**；scale dtype 同上，**无 group_sizes**。
  - weight `transpose(1,2)` → **strided view**（NO contiguous）—— probe 证实 strided 和 contiguous 都 PASS（cos≈0.997），但 strided 匹配 vllm-ascend 性能。
  - E8M0 = `getattr(torch_npu, "float8_e8m0fnu")` = int 293；函数内 lazy import torch_npu，不模块级引入。
  - **融合激活量化（quant_mode=3，已 A5 探针验证 Q9 + e2e 验证）**：`npu_moe_init_routing_v2` 直接在 routing 内做 MXFP8 激活量化——`quant_mode=3` → 排序输出 e4m3、第 4 返回值给 e8m0 block scale，省掉单独一次 `npu_dynamic_mx_quant`（对齐 vllm-ascend A5：原生 v2，非 v3/custom op）。`npu_fused_experts_mxfp8` **唯一路径**就是融合（无两步 fallback、无 env 开关——A5 e2e 验证通过后已移除，简化维护）。**Q9 实测（torch_npu 2.10.0.post2 + A5）**：`quant_mode=3` 接受且 **x_dtype 传 None**（不需要）；ret[3] scale 是 **2D `[N, K/32]` float8_e8m0fnu**（与 `npu_dynamic_mx_quant` 直出 3D 不同）→ **必须 `_normalize_mxfp_scale` 2D→3D `[N,K/64,2]`**；融合 vs 两步 **cos=1.0、qx 字节完全一致**（非近似），在线/离线 e2e 均可跑。
- **DeepEP 没有 mxfp8 dispatch dtype**：`token_dispatcher/deepep.py` 的 `config_map` 只有 BF16/FP8/INT8/NVFP4，把 `dispatcher_output_dtype` 设成 `"mxfp8"` 会 **KeyError**。所以 `NPUMXFP8MoEMethod.process_weights_after_loading` 对 w13 按 `get_moe_a2a_backend().is_deepep()` 分流：DeepEP 设 `"bf16"`（dispatcher 不量化），`apply_fused_gmm1_swiglu` 里 `pertoken_scale is None` 时用 `HiddenStatesDynamicQuant(float8_e4m3fn)` 自己量化再进 gmm1；ascend_tp 才设 `"mxfp8"` 走融合路由量化。**DeepEP 路径代码上已打通但未在 A5 上 e2e 验证**（尤其 low-latency 的 3D hidden_states 是否被融合 gmm1 接受未知）。

- **`--quantization mxfp8` 会强制 flashinfer runner → w1/w3 shard swap（静默乱码）**：`arg_groups/overrides.py::_moe_runner_backend_quant_constraints` 见到 `mxfp8` 就把 `moe_runner_backend` 从 `auto` 顶成 `flashinfer_trtllm`，**不看平台**。这个值不只选 runner：`fused_moe_triton/layer.py:939` 的 **w1/w3 shard swap**（"flashinfer assumes w31 format"）和 `:253` 的 128 对齐 round-up 都挂在它上面。shard swap 的触发名单是 `ModelOptNvFp4FusedMoEMethod` / `Fp8MoEMethod` / **`UnquantizedFusedMoEMethod`** / `CompressedTensorsMxInt4MoE` —— 一旦 MoE method 继承了 `UnquantizedFusedMoEMethod`（本 PR 评审第 ③ 条要求），就命中名单，**每个 expert 的 gate/up 被交换**，gmm1 融合 swiglu 算成 `silu(up)*gate`：**不报错，只是退化重复输出**。若尚未继承（继承自 `FusedMoEMethodBase`）则表现为响亮的 `TypeError: Unexpected quant_info type for flashinfer_trtllm: AscendQuantInfo`。修复：`if view.quantization == "mxfp8" and not is_npu():`，让 backend 停在 `auto` 由 `create_moe_runner` 解析成 ASCEND；并在 `create_moe_runner` 里对非 `auto`/`ascend` 的 backend 显式 raise，堵住手动 `--moe-runner-backend`。
- **MoE 侧 e8m0 dtype 必须取 `torch_npu`，dense 侧 `torch` 也行**：`npu_grouped_matmul*` 按 torch_npu 自己的枚举（A5 上 293）校验 scale dtype 参数，torch 的 dtype 对象会被拒（`weight_scale_dtype only supports float8_e8m0fnu or None, but the actual value is Float8_e8m0fnu`）→ 每次 gmm1 直接死。dense 的 `npu_quant_matmul` 两种都吃，所以 `linear_method_npu.py::_get_float8_e8m0fnu_dtype`（只读 torch）**不能**给 MoE 复用；`moe_methods.py::_require_e8m0_dtype` 需 torch_npu 优先 + lazy import + 模块级缓存。
- **离线 MXFP8 / MXFP4 W4A8 MoE 必须注册空 `weight_offset`**：`modelslim.py::ModelSlimMoEMethod.apply` 照 int8 scheme 无条件读 `layer.{w13,w2}_weight_offset` 来建 `AscendQuantInfo`。两种 MXFP 都是纯 scale 格式（e8m0 block 指数即全部，无零点），scheme 不创建 offset 就会触发 `AttributeError: 'FusedMoE' object has no attribute 'w13_weight_offset'`。修复：`layer.register_parameter(f"{prefix}_weight_offset", None)`（`AscendQuantInfo` 该字段本就是 `Optional`；None 参数不出现在 `named_parameters()`，无 loader 会找它）。
- **Qwen3.5 离线 ModelSlim 必须保留模型专属 packed mapping**：NPU loader 会在 `packed_modules_mapping` 中加入 scoped `model`/`visual` 映射，但 Qwen3.5 的 `in_proj_qkvz` → `in_proj_qkv`+`in_proj_z`、`in_proj_ba` → `in_proj_b`+`in_proj_a` 位于顶层。若 `ModelSlimConfig.get_quant_method` 只读取 scoped mapping，这两个 GDN 投影会静默回退到 unquantized Linear，checkpoint 的 `weight_scale` 随后因参数未注册而被忽略，服务不报错但输出乱码/复读。修复：先收集顶层 list mapping，再用对应 scoped mapping 覆盖；回归测试见 `test_modelslim_config.py`。Qwen3.5 在线 MXFP8 不走 ModelSlim，故精度正常。
- **Qwen3.5 视觉塔也必须接收离线量化配置**：`Qwen3VLForConditionalGeneration` 若构造视觉塔时硬编码 `quant_config=None`，视觉 blocks 的 QKV/proj/MLP 和 merger 会按 BF16 Linear 注册，checkpoint 中 110 个 `visual.*.weight_scale` 随后全部告警并被跳过。vLLM 的 Qwen3-VL 直接把同一个 `quant_config` 传给视觉塔；SGLang 的 blocks/attention/MLP/merger 也已完整透传该参数，因此修复入口应在 `qwen3_vl.py`，并同时验证视觉 packed QKV 映射和图片 e2e，不能仅静默过滤告警。
- **ModelSlim MXFP8 必须保留 K 维末尾的 partial block scale**：scale placeholder 应按 `ceil(K/32)` 注册，不能用 `K//32`。Qwen3-VL 视觉 FC2 的 K=4304，checkpoint 有 135 个 scale；旧 placeholder 只有 134 个，row-parallel loader 会据此静默截断，post-load 再配对成 67，而 `npu_quant_matmul` 要求 `ceil(4304/64)=68`，最终报 `k_x2Scale is 67`。应对奇数 scale count 末尾补零后 reshape 为 pair 布局（135 → 136 → 68），与 vllm-ascend 实现一致。
- **上游大 merge 会静默重写 NPU MoE 层**：2026-07-16 那次 merge 删掉 `fused_moe_method_npu.py`（-1581 行）、新建 `quantization/moe_methods.py`（+994）/ `moe_runner/ascend.py`（+338）/ `npu/moe/*.py`，本仓库的 MXFP8 MoE 实现是**在 merge commit 内部被重新移植的**。上面三条回归全部出自这次移植。**教训：大 merge 后必须先单独跑一遍 A5 e2e 校准 baseline，再往上叠任何改动**，否则后续每一步的验证都被污染、无法二分。
- **A5 无 `torch_npu.npu_gemma_rms_norm`，sgl-kernel-npu wheel 已按 SoC 分构建**：Qwen3.5 用 Gemma RMSNorm，A2/A3 有 native kernel（性能更好，故保留），A5(Ascend 950) 未注册该算子。修复是 [sgl-kernel-npu#638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) + [sglang#32745](https://github.com/sgl-project/sglang/pull/32745) 配对，**分发在编译期决定**：`build.sh -a kernels 910|950` 经 `SGL_KERNEL_NPU_BUILD_TARGET` 在 wheel staging 阶段把 `_gemma_rmsnorm_native.py`（910 走 native）或 `_gemma_rmsnorm_aclnn.py`（950 走 `npu_rms_norm(input, 1+weight, eps)`）**拷贝**成 `norm/gemma_rmsnorm.py`，SGLang 侧只调这个稳定 API、不查 SoC。要点：① **官方 release 只构建 910 target**（`build_and_release.yml` 里 `./build.sh -a kernels` 不带 SOC，matrix 仅 910b/a3），**A5 必须自行 `bash build.sh -a kernels 950`**；② 910/950 wheel 只差这一个模块，但 target 既不进文件名也不进 version，装错时 910 上是静默走慢的 ACLNN 路径、950 上要到首次调用 Gemma RMSNorm 才报错；③ **源码树里不存在 `norm/gemma_rmsnorm.py`**（只有两个 `_gemma_rmsnorm_*.py` 私有源），所以源码态 / `pip install -e .` 会响亮 ImportError，而不是在 A5 上静默拿到 910 实现；④ 残差路径 `add_gemma_rms_norm` 在**所有 target** 上统一用 `npu_add_rms_norm(x, residual, 1+weight, eps)`——因为 torch_npu 没有 native 的 add+gemma 融合算子（vllm-ascend `ops/layernorm.py` 同样只在 plain 路径用 gemma 算子），这个不对称不是遗漏；⑤ 所有 A5 拼写（`950`/`Ascend950`/`Ascend950PR_*`/`Ascend950DT_*`）在 `normalize_soc_version` 里折叠成 `Ascend950`，且 `-a kernels` 在 A5 上一律编 `Ascend910_9382` 兼容目标（LoRA 等模块未移植 A5 pipeline），精确 PR/DT target **不再透传给 cmake**；⑥ SGLang 侧 `srt/layers/layernorm.py` 的 provider import 带 try/except 回落 `torch_npu.npu_gemma_rms_norm`，所以装旧 wheel 的 A2/A3 用户行为完全不变（否则模块级硬 import 会让**所有**模型在 import 期就炸）；`sglang/kernels` registry 那条路径则**故意**硬报错——点名选中的 backend 不能静默换成别的 provenance；⑦ `1.0 + weight` 在 weight 自身 dtype 里算（kernel 要求 gamma 与 input 同 dtype），bf16 下比 910 native 略差，与 vllm-ascend 和 SGLang CUDA 的 `gemma_weight` buffer 一致，是既有取舍不是 bug。

> 详细根因分析、调试过程、代码示例、修复方法见 [docs/known-pitfalls.md](docs/known-pitfalls.md)。

## 开发工具

- **pre-commit**：`sglang/` 下每个 worktree 都是独立 git checkout，pre-commit 必须进入**具体 worktree**（如 `sglang/qwen3_moe_w8a8/`）后运行：
  ```bash
  pre-commit run --all-files  # 在对应 worktree 目录下执行，如 sglang/qwen3_moe_w8a8/
  ```
  Windows 上 CI 脚本已修复编码和路径分隔符兼容性问题（`check_workflow_job_names.py`、`check_registered_tests.py`）。

## 网络访问限制（重要）

**Claude 的 Bash 环境到 github.com 的网络不稳定**：TLS 握手会间歇性失败（`gnutls_handshake() failed` / `SSL_ERROR_SYSCALL`），**大传输尤其容易失败**（实测：fork 的大 merge push 连续失败、`curl`/`fetch` 也失败；但主仓的小 commit push 一次成功）。所以这不是硬封锁，而是按传输大小/运气波动的不稳定连接。因此：

- **push 是 Claude 的活，不要甩给用户**（用户 2026-07-20 明确要求）。失败就挂后台重试循环（`git push` → 拿 `git rev-parse refs/remotes/origin/<branch>` 和 HEAD 比对 → `sleep 20` → 重来，60~90 次），期间继续干别的活。实测断网可持续 10+ 分钟，但重试到第 2~5 次成功是常态。只有循环耗尽才告诉用户，并说明试了多少次。同理适用于 `gh api`（发评论、resolve thread、改 PR 正文）。
- Claude 侧 `git push` 报 TLS 错 **不代表推送真的失败**——先用 `git rev-parse refs/remotes/origin/<branch>` 核对再下结论（注意 `git rev-parse --short A B` 一次传两个 ref 在 ref 不存在时会报 `Needed a single revision`，分开查）。
- **CI 日志 / GitHub Actions**：`gh` CLI **已认证可用**（account TallMessiWu），CI 失败时**先用 gh 拉真实日志再动手**，不要凭本地复现或臆测下结论：`gh pr checks <N> --repo sgl-project/sglang`、`gh run view --job <id> --log-failed`、`gh run watch <run-id> --exit-status`。注意 CI 跑的是 `pre-commit run --all-files`，ruff 的实际 autofix（如 UP037 给带 `from __future__ import annotations` 的文件去引号）可能与本地裸 `ruff --select=...` 不一致，会以「files were modified by this hook」失败。
- 需要联网（搜索、抓网页）时走 `web-access` skill，不要用裸 curl。

## 代码提交
代码提交时必须使用gitmoji-commit这个skill。每次提交代码后，更新 AGENTS.md 或相关 agent 指导文档。

### 子模块 / worktree 提交流程
1. **sglang 代码改动**：进入对应 worktree（见上「SGLang worktree 目录规则」）提交，并更新该 worktree 内的 agent 指导文档；推送到 fork（https://github.com/TallMessiWu/sglang）。**`sglang/` 整个目录被 gitignore、主仓不跟踪任何 sglang 子模块**，所以推完 fork 就结束，主仓侧不需要更新指针快照。
2. 回到主仓，更新主仓 AGENTS.md（记录相关变更摘要）。
3. **参考子模块（MindIE-SD / msmodelslim / vllm-ascend / sgl-kernel-npu）**：`.gitmodules` 已为各自配 `branch=`（dev / master / main / main）。需要同步上游时在主仓跑 `git submodule update --remote <name>`，再提交主仓记录新指针快照——git 子模块始终记录具体 commit，`branch=` 只是声明跟踪哪条上游分支、供 `--remote` 使用。

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

## 启动脚本权重路径

`llm/` 下的 Qwen3/Qwen3.5 启动脚本使用当前共享存储上的模型路径；offline 脚本依赖自动检测量化 scheme，不显式传 `--quantization modelslim`。Qwen3 MoE W4A8 启动脚本已移除；Qwen3.5 MoE W4A8 使用 `qwen3.5_moe_online_w4a8.sh`（BF16 权重 + `--quantization mxfp_w4a8`）和 `qwen3.5_moe_offline_w4a8.sh`（ModelSlim 权重自动检测）。**Qwen3.5 MoE W4A4** 使用 `qwen3.5_moe_online_w4a4.sh`（BF16 权重 + `--quantization mxfp4`）和 `qwen3.5_moe_offline_w4a4.sh`（ModelSlim 权重自动检测，checkpoint 由 `~/Downloads/qwen_mxfp_quant/qwen3_5_35b_moe_w4a4_mxfp4.yaml` 混合精度导出）。
