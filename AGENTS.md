# AGENTS.md

本仓库用于研究和实现 SGLang **Diffusion 侧**在华为 Ascend NPU 上的 MXFP8/MXFP4 量化适配（Wan2.2 等 Diffusion 模型）。
**如果涉及 LLM serving (`srt`) 侧的功能开发（如 MXFP8/MXFP4），请务必参考 `vllm-ascend` 的实现模式。**
*注：Qwen3 和 Qwen3.5 模型在 SGLang 内部共用底层 Linear/MoE 算子，因此量化实现代码完全一致。*

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion), [sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLM Qwen3)
- **Fork**: https://github.com/TallMessiWu/sglang

## 分支规则

Diffusion、Dense W8A8/W4A8 及 **Dense W4A4（PR #23795，2026-07-17 合入）** 侧功能均已合入上游 `sgl-project/sglang`（见下「已合并 PR」）。后续分支都基于 upstream/main rebase，**已含全部已合并代码**，故本地只保留 **3 个活跃工作分支**（各对应一个未合并 PR）：

| 分支 | 目录 | PR | 状态 |
| ---- | ---- | -- | ---- |
| `junlin_qwen3_moe_w8a8` | `sglang/qwen3_moe_w8a8/`（派生 worktree） | [#30768](https://github.com/sgl-project/sglang/pull/30768) | WIP OPEN，LLM MoE W8A8 MXFP8，在线+离线 A5 已 e2e 验证；OrangeRedeng 评审 **8 条全部落地**（⑧ NZ 已合入并回复，2026-07-21）。PR body 性能/精度数据已更新为 NZ 版。**2026-07-21 merge upstream/main（178 commit，`c0ed009f5`）解冲突，PR 已回到 MERGEABLE——⚠️ merge 后 A5 e2e 尚未重跑**。HEAD `e22b7bef8` |
| `junlin_qwen3_moe_w4a8` | `sglang/qwen3_moe_w4a8/`（派生 worktree） | 待创建 | 🚧 WIP，LLM MoE W4A8 MXFP（MXFP4 权重 + FP8 激活），在线+离线已实现，A5 待验证 |
| `junlin_qwen3.5_dense_w8a8` | `sglang/qwen3.5_dense_w8a8/`（派生 worktree） | 待创建 | 🚧 WIP，Qwen3.5 Dense W8A8 MXFP8 实验/验证（代码已合入 upstream/main，此分支用于 A5 在线+离线验证、跑分、模型适配）。**2026-07-21 rebase 到 `junlin_qwen3_moe_w8a8`（`e22b7bef8`）**，故也含未合并的 MoE W8A8 PR #30768 代码。HEAD `3ee602835` |

> `junlin_qwen3_dense_w4a4` 的 PR #23795 已合并，但 `sglang/qwen3_dense_w4a4/` 目录**不能删除**——它是 worktree 主 clone（持有共享 `.git`）+ 主仓子模块，其余 3 个 worktree 都挂在它下面，删除会破坏整个 worktree 结构。详见下文「SGLang worktree 目录规则」。

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

SGLang 代码以 `git worktree` 形式放在 `sglang/` 下，各目录**共享同一个 `.git`**：`sglang/qwen3_dense_w4a4/` 是主 clone（持有独立 `.git` 目录）**且是主仓子模块**（GitHub 上可点击跳转到 fork 的 `junlin_qwen3_dense_w4a4`）；其余目录是从它派生的 worktree，被 `.gitignore` 忽略（纯本地、非子模块）。**需要修改哪个分支，就直接进入对应目录修改；不要在现有目录里用 `git checkout` 切分支。**

> ⚠️ **2026-07-21 实况与下表不符，尚未处理**：主仓 `git status` 显示 `sglang/qwen3_dense_w4a4` 为 deleted，`git worktree list` 只列出 `qwen3_moe_w8a8`（现为**主 worktree**，列在首行）和 `qwen3.5_dense_w8a8`——主 clone 和 `qwen3_moe_w4a8` 均不在其中。`.git` 仍可用，各 worktree 的 git 命令正常，`qwen3_moe_w8a8_nz` 即从 `qwen3_moe_w8a8` 派生成功。动 worktree 前先跑 `git worktree list` 确认实际状态，别照下表假设。

| 路径 | 对应分支 | 用途 |
| ---- | -------- | ---- |
| `sglang/qwen3_dense_w4a4/` | `junlin_qwen3_dense_w4a4` | **主 clone + 主仓子模块**（其余 3 个 worktree 共享此处 `.git`）；LLM Dense W4A4 MXFP4，**PR #23795 已合并（2026-07-17）**，此目录作为 worktree 基础设施保留，不再是活跃开发分支。 |
| `sglang/qwen3_moe_w8a8/` | `junlin_qwen3_moe_w8a8` | 派生 worktree（gitignore、纯本地）；LLM MoE W8A8 MXFP8（PR #30768）。 |
| `sglang/qwen3_moe_w4a8/` | `junlin_qwen3_moe_w4a8` | 派生 worktree（gitignore、纯本地）；LLM MoE W4A8 MXFP（MXFP4 权重 + FP8 激活）。 |
| `sglang/qwen3.5_dense_w8a8/` | `junlin_qwen3.5_dense_w8a8` | 派生 worktree（gitignore、纯本地）；Qwen3.5 Dense W8A8 MXFP8 实验/验证。 |

开发约定：
- 改 MoE W8A8：进入 `sglang/qwen3_moe_w8a8/`（NZ 已在其中）；改 MoE W4A8：进入 `sglang/qwen3_moe_w4a8/`；改 Qwen3.5 Dense W8A8：进入 `sglang/qwen3.5_dense_w8a8/`。`sglang/qwen3_dense_w4a4/`（Dense W4A4，PR #23795）已合并，日常开发不再进入，仅作为其余 3 个 worktree 的 `.git` 宿主保留。
- 已合并的 Diffusion / Dense W8A8/W4A8/W4A4 代码都在 upstream/main（各 worktree rebase 后均含），无需单独 checkout。如需基于某已合并 PR 再开发，从主 clone（`sglang/qwen3_dense_w4a4`）用 `git worktree add` 新建独立目录，不要复用已有 worktree。
- **`sglang/qwen3_dense_w4a4/` 是主仓子模块**（主仓跟踪其 commit 指针，见 `.gitmodules`）；`qwen3_moe_w8a8/`、`qwen3_moe_w4a8/` 和 `qwen3.5_dense_w8a8/` 被 `.gitignore` 忽略、主仓不跟踪。旧 `sglang/diffusion_w8a8` 子模块随 Diffusion 合并上游后已移除。

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
| LLM (Qwen3 & 3.5) Dense W4A8 (MXFP4/8) | 已合并 #23650          | ✅ 在线（`mxfp_w4a8`，单级真 W4A8/MXFP8 激活） | ✅ 离线（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`，权重格式同 MXFP8：`float8_e4m3fn`） |
| LLM (Qwen3 & 3.5) Dense W4A4 (MXFP4)   | 已合并 #23795          | ✅ 在线已实现（`mxfp4`，NPU 设备分发，**双级 MXFP4** `NPUDualLevelMXFP4LinearMethod`；A5 e2e 已验证，双级修复了单级 RTN 贪心死循环） | ✅ 离线已实现（`W4A4_MXFP4` → `ModelSlimMXFP4Scheme` → `NPUSingleLevelMXFP4OfflineLinearMethod`，单级真 MXFP4，权重 fp8 容器 `float8_e4m3fn`） |
| LLM (Qwen3 & 3.5) MoE W8A8 (MXFP8)     | `junlin_qwen3_moe_w8a8`（#30768 WIP） | ✅ 在线已实现（`mxfp8`，`NPUMXFP8OnlineMoEMethod`（继承 `UnquantizedFusedMoEMethod`），A5 e2e 已验证） | ✅ 离线已实现（`W8A8_MXFP8` → `ModelSlimMXFP8MoEScheme` → `NPUMXFP8MoEMethod` 离线分支，A5 e2e 已验证） |
| LLM (Qwen3 & 3.5) MoE W4A8 (MXFP4/8)   | `junlin_qwen3_moe_w4a8`（待创建 PR） | ✅ 在线（`mxfp_w4a8`，NPU 设备分发，`NPUMXFP4W4A8FusedMoEMethod`，A5 e2e 待验证） | ✅ 离线（`W4A8_MXFP` → `ModelSlimMXFP4W4A8MoEScheme` → `NPUMXFP4W4A8MoEMethod`，权重 packed fp4 uint8，A5 e2e 待验证） |
| LLM (Qwen3 & 3.5) MoE W4A4 (MXFP4)     | 待定                   | ❌ 待实现               | ❌ 待实现               |

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
| `schemes/modelslim_mxfp4_w4a8.py`  | ModelSlim W4A8_MXFP 离线 scheme（权重 `float8_e4m3fn`，激活 FP8 动态量化） |
| `schemes/modelslim_w8a8_int8.py`    | ModelSlim W8A8 Int8 离线 scheme                        |

在线量化：
- `--quantization mxfp8` → `linear_method_npu.py` → `NPUMXFP8LinearMethod`（Linear 层）
- `--quantization mxfp8` (MoE 层) → `fp8.py` → `hardware_backend/npu/quantization/online_moe_methods.py` → `NPUMXFP8OnlineMoEMethod`（在线 MXFP8；**继承 `UnquantizedFusedMoEMethod`**，只 override `create_moe_runner` 换上 `NPUMXFP8MoEMethod("w13"/"w2")` kernel，`create_weights` / `process_weights_after_loading` / `apply` 全部继承）。单独成文件是因为 `unquant.py` 顶层 import 了 `moe_methods.py`，放同一文件会成 import 环
- `--quantization mxfp_w4a8` (MoE 层) → `npu_mxfp4.py:113` → `NPUMXFP4W4A8FusedMoEMethod`（在线 MXFP4 W4A8，三段式，仅 FusedMoE/TP）
- `--quantization mxfp_w4a8` → `layers/quantization/npu_mxfp4.py` → `NPUMxfp4Config` → `NPUMXFP4W4A8LinearMethod`（真 W4A8：单级 FP8 激活 + FP4 权重，apply 复用离线 `npu_quant_matmul(x2_dtype=fp4)`；权重在线 `npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)`）
- `--quantization mxfp4`（NPU 设备分发，`is_npu()` 块注册 `Mxfp4W4A4Config`；非 NPU 为上游 `Mxfp4Config`/MoE）→ `layers/quantization/npu_mxfp4_w4a4.py` → `Mxfp4W4A4Config` → **`NPUDualLevelMXFP4LinearMethod`（在线唯一路径，双级 MXFP4：细 FP8 E4M3 L0 scale + 粗 L1 scale，`npu_dynamic_dual_level_mx_quant` + `npu_dual_level_quant_matmul`，权重 FRACTAL_NZ，仅 A5/Ascend 950）**。单级 `NPUSingleLevelMXFP4LinearMethod` 仅作离线基类保留（离线 `W4A4_MXFP4` → `ModelSlimMXFP4Scheme` → `NPUSingleLevelMXFP4OfflineLinearMethod`，单级）。移植自 Diffusion `NPUMXFP4DiffusionLinearMethod`/MindIE-SD `W4A4MXFP4DualQuantLinear`

其他关键文件：

- `srt/models/qwen3.py` — Qwen3 / 3.5 模型定义，`EntryClass = Qwen3ForCausalLM`
- `srt/models/qwen3_moe.py` — Qwen3 MoE 模型定义，`EntryClass = Qwen3MoeForCausalLM`
- `srt/hardware_backend/npu/quantization/moe_methods.py` — 所有 per-gmm MoE kernel 类（`NPUUnquantMoEMethod`、`NPUMXFP8MoEMethod`、`NPUMXFP4W4A8MoEMethod` / `NPUMXFP4W4A8FusedMoEMethod`、`NPUW4A4Int4MoEMethod`、`NPUW8A8Int8MoEMethod`、`NPUW4A8Int8MoEMethod` 等）
- `srt/hardware_backend/npu/quantization/online_moe_methods.py` — 在线量化 FusedMoE 入口类（`NPUMXFP8OnlineMoEMethod`），与 `moe_methods.py` 分开以避开 `unquant.py` 的 import 环
- `srt/hardware_backend/npu/moe/matmul.py` — MoE matmul kernel wrapper（`GroupedMatmul`、`GroupedMatmulSwigluQuant`（gmm1 融合 gate/up+swiglu+requant，返回 (激活, block scale)））
- `srt/hardware_backend/npu/moe/quant.py` — MoE 量化 kernel wrapper（`HiddenStatesDynamicQuant`：int8/quint4x2 走 `npu_dynamic_quant`，`float8_e4m3fn` 走 `npu_dynamic_mx_quant`）。**原名 `hidden_states_quant.py`，按评审建议改名**
- `srt/hardware_backend/npu/moe/activation.py` — MoE 激活函数库（`NPUSwiglu`、`NPUSwigluQuant`、`NPUSwigluMXFP8Quant`、`NPUSwigluDeepEPKernel` 等）
- `srt/layers/moe/moe_runner/ascend.py` — Ascend MoE runner 编排（gmm1→activation→gmm2），含 W4A8 MXFP 非融合分支
- `srt/layers/quantization/modelslim/schemes/modelslim_mxfp4_w4a8_moe.py` — `ModelSlimMXFP4W4A8MoEScheme`（离线 W4A8_MXFP MoE）
- `srt/models/registry.py` — `ModelRegistry`，扫描 `sglang.srt.models` 注册所有 `EntryClass`
- `srt/layers/rotary_embedding/base.py` — RoPE 实现，NPU 路径 import `sgl_kernel_npu`
- `srt/model_loader/loader.py` — `DefaultModelLoader`：`_get_quantization_config` → `_initialize_model`
- `MindIE-SD/mindiesd/quantization/layer.py` — NPU 量化参考实现 (Diffusion)
- `vllm-ascend/vllm_ascend/quantization/methods/w8a8_mxfp8.py` — NPU 量化参考实现 (LLM)
- `msmodelslim/.../save/ascendv1.py` — MXFP4 权重导出格式
- `docs/npu-api/DualLevelQuantBatchMatmul.md`、`docs/npu-api/DynamicDualLevelMxQuant.md` — Ascend 双级量化 kernel API 参考（原在根目录，现归入 `docs/npu-api/`）

## 注意事项

- **CANN 版本**: MXFP8 需 ≥ 8.0.RC3；MXFP4 最低版本待确认
- **硬件**: Atlas 800I A2/A3（`DualLevelQuantBatchMatmul` 仅支持 Ascend 950，A2/A3 不支持）
- **CPU offload**：`dit_cpu_offload` 默认 True，`process_weights_after_loading` 中需手动 `.to("npu:X")` 后再调用量化 API
- **bias 精度**：量化 matmul 要求 bias 为 `float32`
- **tensor reshape**：diffusion 输入可能是 3D `[batch, seq, hidden]`，NPU 量化 API 需 2D，apply 中先 reshape 后 restore
- 与社区 YChange01 协调 MXFP8/MXFP4 工作分工（已在 Issue #14424 认领）

## 已知陷阱

- **量化不生效/乱码输出**：先验证模型注册（`ModelRegistry.models.keys()`），`sgl_kernel_npu` 缺失 kernel → import 失败 → 静默 fallback HF Transformers → 乱码。修复：非核心 kernel import 改 try/except + `None`。
- **模块级 `import torch_npu` 炸全平台 CI**：用 `from sglang.srt.utils import is_npu; if is_npu(): import torch_npu` 守卫，**不要**用 `current_platform.is_npu()`（NPU 插件未装时恒 False）。
- **transpose 不加 `.contiguous()`**：`npu_grouped_matmul` 靠 strides 感知 block-scale 布局，`.contiguous()` 物理重排 → 乱码。dense 路径 `.contiguous()` 则 OK。
- **vllm-ascend MoE 量化均为 offline**：无 BF16→FP 在线转换，MoE 在线量化需自实现。
- **FusedMoE vs EPMoE dispatch_output 类型不同**：当前仅支持 `StandardDispatchOutput`（TP-only）。
- **W4A8_MXFP 权重格式同 MXFP8**：`float8_e4m3fn`（非 packed uint8），`create_weights` 实现一致。
- **离线 W4A8 A5 两报错**：① prefill NZ format 错 → 升级 torch_npu 到 `2.10.0.post1` 解决；② decode ATB 段错误 → 与量化无关，别加 `--disable-cuda-graph`（或用 `ASCEND_USE_FIA=1`）。
- **A5 默认 ATB 注意力算子崩溃（warmup 固定挂，与量化无关）**：`--device npu` 在 A5(Ascend 950) 上默认 prefill 走 ATB `SelfAttentionOperation`（`_npu_flash_attention_qlens`，`ascend_backend.py:1472`）；服务起来后 SGLang 自动 warmup 请求（一次 prefill）触发 `RuntimeError: SelfAttentionOperation CreateOperation failed!`。`CreateOperation` 是 ATB 按 SoC 能力表构图/校验的步骤，A5 不支持该算子（decode 默认 `_npu_paged_attention` 同源）；A2/A3(910B/C) 支持故不受影响——**崩溃是 A5 特有**。修复：`export ASCEND_USE_FIA=1`，让 prefill+decode 都走 A5 原生 FIA 算子 `npu_fused_infer_attention_score`（`forward_extend` 走 `use_fia` 分支，永不到达 `_npu_flash_attention_qlens`）。`llm/` 全部 serve 脚本已默认带此 env。**别**用注释掉 `qk_head_dim<=128 and ...` 条件强制 `False`→native SDPA 的脏改（慢、无融合 kernel、且只补 prefill）。与上条 decode ATB 段错误同根。
- **在线 W4A8 FP4 dtype 来源**：fp4 dtype 参数必须来自 `torch_npu.float4_e2m1fn_x2`（int 296），**不能**用 `torch.float4_e2m1fn_x2`（torch dtype 对象会被 op 拒绝）。`_get_float4_e2m1fn_x2_dtype()` 在 NPU 时优先 `getattr(torch_npu, ...)`。
- **sglang 文档迁到 `docs_new/docs/`**：改 legacy `docs/` 会被 CI lint 拒，新文档一律写 `docs_new/docs/`（`.mdx`）。
- **MoE `npu_grouped_matmul` 需显式 `x_dtype` + `weight_dtype`**：缺少则 kernel 走错 dequant 路径 → 乱码。dense `npu_quant_matmul` 不需要（有 `group_sizes`）。
- **MoE gmm1 用 fused `npu_grouped_matmul_swiglu_quant_v2`**：勿拆三步；`group_list` 需 count→cumulative 转换。
- **MoE weight+scale `.transpose(1,2)` 不要 `.contiguous()`**：真实机制是 gmm1 的 `CheckMXTranspose` 断言——**weight 与 weight_scale 的 transpose 标志必须一致**，只动一边直接报错（非乱码）。两边都 contiguous 数值正确但更慢（128-expert decode −6.2%）。dense 路径 contig 则 OK。**注意小 expert 数 micro-bench 结论相反**（E=4 时 contiguous 快 58%），定 layout 必须用真实 expert 数。
- **MoE MXFP8 转 FRACTAL_NZ 必须 cast 在 transpose 之前**：`npu_format_cast` 产出非 transposed 张量，先 transpose 再 cast 会触发上条断言。正确顺序同 dense W4A8（`linear_method_npu.py:443`/`:578`），**不能**照抄同文件 int8 MoE 的 `npu_format_cast(w.transpose(1,2))`（int8 无 MX scale 要同步）。A5 实测 decode +1.4% / prefill +3.8%，探针 `llm/probe_mxfp8_moe_nz.py`。
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
- **离线 MXFP8 MoE 必须注册空 `weight_offset`**：`modelslim.py::ModelSlimMoEMethod.apply` 照 int8 scheme 无条件读 `layer.{w13,w2}_weight_offset` 来建 `AscendQuantInfo`。MXFP8 是纯 scale 格式（e8m0 block 指数即全部，无零点），scheme 不会创建它 → `AttributeError: 'FusedMoE' object has no attribute 'w13_weight_offset'`。修复：`layer.register_parameter(f"{prefix}_weight_offset", None)`（`AscendQuantInfo` 该字段本就是 `Optional`；None 参数不出现在 `named_parameters()`，无 loader 会找它）。
- **上游大 merge 会静默重写 NPU MoE 层**：2026-07-16 那次 merge 删掉 `fused_moe_method_npu.py`（-1581 行）、新建 `quantization/moe_methods.py`（+994）/ `moe_runner/ascend.py`（+338）/ `npu/moe/*.py`，本仓库的 MXFP8 MoE 实现是**在 merge commit 内部被重新移植的**。上面三条回归全部出自这次移植。**教训：大 merge 后必须先单独跑一遍 A5 e2e 校准 baseline，再往上叠任何改动**，否则后续每一步的验证都被污染、无法二分。

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
1. **sglang 代码改动**：进入对应 worktree（见上「SGLang worktree 目录规则」）提交，并更新该 worktree 内的 agent 指导文档；推送到 fork（https://github.com/TallMessiWu/sglang）。`sglang/qwen3_dense_w4a4/` 是主 clone + 主仓子模块，**主仓跟踪其 commit 指针**——在该 worktree 提交推送后，回到主仓 `git add sglang/qwen3_dense_w4a4` 更新指针快照即可。`qwen3_moe_w8a8/` 和 `qwen3.5_dense_w8a8/` 被 gitignore、非子模块，主仓不跟踪。
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
