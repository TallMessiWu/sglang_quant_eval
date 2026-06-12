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
| `junlin_qwen3_dense_w8a8`   | LLM 侧，Qwen3 / 3.5 dense 模型 MXFP8 量化适配                                                                                      |
| `junlin_qwen3_dense_w4a8` | LLM 侧，Dense W4A8 在线量化（MXFP4 双级，`--quantization mxfp4_w4a8_npu`）；离线 W4A8 已实现（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`）。**已 merge 同步至 `junlin_qwen3_dense_w8a8` 的 upstream 基线 + 移植 MXFP8 layout/bias-cache 性能优化（2026-06-12）** |
| `junlin_qwen3_dense_w4a4` | LLM 侧，在 w4a8 基础上新增 W4A4 在线量化（单级 MXFP4，`--quantization mxfp4w4a4_npu`）+ 离线 W4A4（INT4 ModelSlim） |
| `junlin_qwen3_moe_w8a8`   | **当前工作分支**，LLM 侧，Qwen3/3.5 MoE W8A8 MXFP8 在线量化（FusedMoE/TP，`--quantization mxfp8`）；EPMoE 待实现 |

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

> **本地 vs fork 远端命名**：本地分支名已全部对齐 `junlin_<文件夹>`，fork（`TallMessiWu/sglang`）默认分支已改为 `junlin_diffusion_w8a8`。唯一例外：`junlin_qwen3_dense_w8a8` 的 **fork 远端仍叫 `junlin_qwen3_dense`**（它的 upstream tracking 也指 `origin/junlin_qwen3_dense`）——因为它对应**未合并** PR [#22352](https://github.com/sgl-project/sglang/pull/22352)，而跨 fork 重命名 head 分支会**关闭** PR（已踩坑验证：rename API 不会重定向跨仓 PR）。待 #22352 合并后再把远端同步改名。已合并 PR 的分支（原 `junlin`、`junlin_mxfp4`）改名安全。

## 在线/离线量化模式

- **在线量化（Online）**：加载 FP16/BF16 权重，`process_weights_after_loading` 中实时量化，用 `--quantization mxfp8/mxfp4` 触发
- **离线量化（Offline ModelSlim）**：加载 msmodelslim 预量化权重，用 `--quantization modelslim` 触发，scheme 由 `quant_model_description.json` 自动检测

## 实现进度

| 功能                                   | 分支                   | 在线实现状态            | 离线实现状态            |
| -------------------------------------- | ---------------------- | ----------------------- | ----------------------- |
| Diffusion MXFP8                        | `junlin_diffusion_w8a8`             | ✅                      | ✅                      |
| Diffusion MXFP4                        | `junlin_diffusion_w4a4`       | ✅                      | ✅                      |
| LLM (Qwen3 & 3.5) Dense W8A8 (MXFP8)   | `junlin_qwen3_dense_w8a8` | ✅ (已对齐 vllm-ascend) | ✅ (已对齐 vllm-ascend) |
| LLM (Qwen3 & 3.5) Dense W4A8 (MXFP4/8) | `junlin_qwen3_dense_w4a8` | ✅ 在线已实现（`mxfp4_w4a8_npu`，双级） | ✅ 离线已实现（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`，权重格式同 MXFP8：`float8_e4m3fn`） |
| LLM (Qwen3 & 3.5) Dense W4A4 (MXFP4)   | `junlin_qwen3_dense_w4a4` | ✅ 在线已实现（`mxfp4w4a4_npu`，单级 MXFP4） | ✅ 离线已实现（`W4A4_DYNAMIC` → `ModelSlimW4A4Int4` + `NPU_W4A4DynamicLinearMethod`，**INT4 非 MXFP4**） |
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
- `--quantization mxfp4_w4a8_npu` → `layers/quantization/npu_mxfp4.py` → `NPUMxfp4Config` → `NPUMXFP4W4A8LinearMethod`（双级 W4A8）
- `--quantization mxfp4w4a4_npu` → `layers/quantization/npu_mxfp4_w4a4.py` → `NPUMxfp4W4A4Config` → `NPUSingleLevelMXFP4LinearMethod`（单级 W4A4）

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
  from sglang.srt.platforms import current_platform
  _is_npu = current_platform.is_npu()
  if _is_npu:
      import torch_npu
  ```
  模块级用到 `torch_npu` 属性的常量（如 `_FLOAT8_E8M0FNU_DTYPE`）也要用 `if _is_npu else` 三元守卫；函数体内的 `torch_npu.xxx` 调用只在 NPU 运行时执行，无需改。

- **`process_weights_after_loading` 中 transpose 不加 `.contiguous()`**：`npu_grouped_matmul` 通过 strides 感知内存布局，`.contiguous()` 会物理重排数据破坏 block-scale 映射 → 乱码。用 `Parameter(qw.transpose(1, 2), requires_grad=False)` 直接包装非连续 view，不要在 transpose 后加 `.contiguous()`。

- **vllm-ascend MoE MXFP8/MXFP4 均为 offline**：`AscendW8A8MXFP8DynamicFusedMoEMethod`（`w8a8_mxfp8.py:178`）和 `AscendW4A4MXFP4DynamicFusedMoEMethod`（`w4a4_mxfp4.py:119`）的 `process_weights_after_loading` 只做 layout transform，没有 BF16→FP 在线转换。**MoE W4A8_MXFP 在 vllm-ascend 没有 MoE scheme**（`quant_parser.py` 注册了字符串但无对应类）。MoE 在线量化需自实现：online quant 参考 dense `NPUMXFP8LinearMethod`，routing pipeline 参考 `npu_fused_experts_w4a4`。

- **FusedMoE vs EPMoE quant method 共享，但 dispatch_output 类型不同**：`Fp8Config.get_quant_method` 对 FusedMoE 及其所有 EP 子类（`DeepEPMoE`/`NpuFuseEPMoE`/`MoriEPMoE`）返回同一个 method 实例。EPMoE 传入 `apply` 的不是 `StandardDispatchOutput`，需单独处理。当前 `NPUMXFP8FusedMoEMethod.apply` 仅支持 `StandardDispatchOutput`（TP-only），其他类型抛 `NotImplementedError`。

- **W4A8_MXFP checkpoint 权重格式**：`qwen3-8b-dense-w4a8` 检查点的权重存储为 `float8_e4m3fn`（非 packed FP4 uint8），shape 为 `[out, in]`，与 MXFP8 完全相同。这是 msmodelslim 旧版本导出格式（新版 `ascendv1.py` 会 `pack_fp4_to_uint8` → `uint8` shape `[out, in//2]`）。因此 `ModelSlimMXFP4W4A8Scheme` 的 `create_weights` 与 `ModelSlimMXFP8Scheme` 实现一致。

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

- **W4A8 在线 dual-level 去 `.contiguous()` 待 NPU 验证**（`junlin_qwen3_dense_w4a8`，2026-06-12）：把 MXFP8 dense 的 layout 优化移植到 `NPUMXFP4W4A8LinearMethod` 时，`process_weights_after_loading` 里 `w_dual_scale.squeeze(-1).transpose(0, 1)` 去掉了 `.contiguous()`（commit `33dfc0b9b`，已标 `TODO(NPU-validate)`）。但这是 **`npu_dual_level_quant_matmul`**（W4A8 dual-level kernel），**与 MXFP8 dense 的 `npu_quant_matmul` 不是同一个 kernel**，strided-view 容忍度未在 Ascend 950/A3 上验证。**首次在 NPU 跑在线 W4A8（`--quantization mxfp4_w4a8_npu`）时必须确认输出无乱码**；若回归，只 revert `linear_method_npu.py` 这一行（恢复 `.contiguous()`），bias 缓存优化不受影响。bias 缓存（`layer.bias_fp32`）对在线/离线两条 W4A8 路径都安全、已落地。

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
