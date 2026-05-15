# CLAUDE.md

本仓库用于研究和实现 SGLang **Diffusion 侧**在华为 Ascend NPU 上的 MXFP8/MXFP4 量化适配（Wan2.2 等 Diffusion 模型）。
**如果涉及 LLM serving (`srt`) 侧的功能开发（如 MXFP8/MXFP4），请务必参考 `vllm-ascend` 的实现模式。**
*注：Qwen3 和 Qwen3.5 模型在 SGLang 内部共用底层 Linear/MoE 算子，因此量化实现代码完全一致。*

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion), [sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLM Qwen3)
- **Fork**: https://github.com/TallMessiWu/sglang

## 分支规则

| 分支                     | 说明                                                                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `junlin`               | 主线，**不动**                                                                                                                         |
| `junlin_mxfp4`         | Diffusion MXFP8 + MXFP4 在线量化                                                                                                             |
| `junlin_mxfp4_offline` | Diffusion 在 `junlin_mxfp4` 基础上增加 MXFP4 离线加载                                                                                      |
| `junlin_qwen3_dense`   | LLM 侧，Qwen3 / 3.5 dense 模型 MXFP8 量化适配                                                                                      |
| `junlin_qwen3_dense_w4a8` | LLM 侧，Dense W4A8 在线量化（MXFP4 双级，`--quantization mxfp4_npu`）；离线 W4A8 已实现（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`） |
| `junlin_qwen3_dense_w4a4` | LLM 侧，在 w4a8 基础上新增 W4A4 在线量化（单级 MXFP4，`--quantization mxfp4w4a4_npu`）+ 离线 W4A4（INT4 ModelSlim） |
| `junlin_qwen3_moe_w8a8`   | **当前工作分支**，LLM 侧，Qwen3/3.5 MoE W8A8 MXFP8 在线量化（FusedMoE/TP，`--quantization mxfp8`）；EPMoE 待实现 |

## 在线/离线量化模式

- **在线量化（Online）**：加载 FP16/BF16 权重，`process_weights_after_loading` 中实时量化，用 `--quantization mxfp8/mxfp4` 触发
- **离线量化（Offline ModelSlim）**：加载 msmodelslim 预量化权重，用 `--quantization modelslim` 触发，scheme 由 `quant_model_description.json` 自动检测

## 实现进度

| 功能                                   | 分支                   | 在线实现状态            | 离线实现状态            |
| -------------------------------------- | ---------------------- | ----------------------- | ----------------------- |
| Diffusion MXFP8                        | `junlin`             | ✅                      | ✅                      |
| Diffusion MXFP4                        | `junlin_mxfp4`       | ✅                      | ✅                      |
| LLM (Qwen3 & 3.5) Dense W8A8 (MXFP8)   | `junlin_qwen3_dense` | ✅ (已对齐 vllm-ascend) | ✅ (已对齐 vllm-ascend) |
| LLM (Qwen3 & 3.5) Dense W4A8 (MXFP4/8) | `junlin_qwen3_dense_w4a8` | ✅ 在线已实现（`mxfp4_npu`，双级） | ✅ 离线已实现（`W4A8_MXFP` → `ModelSlimMXFP4W4A8Scheme`，权重格式同 MXFP8：`float8_e4m3fn`） |
| LLM (Qwen3 & 3.5) Dense W4A4 (MXFP4)   | `junlin_qwen3_dense_w4a4` | ✅ 在线已实现（`mxfp4w4a4_npu`，单级 MXFP4） | ✅ 离线已实现（`W4A4_DYNAMIC` → `ModelSlimW4A4Int4` + `NPU_W4A4DynamicLinearMethod`，**INT4 非 MXFP4**） |
| LLM (Qwen3 & 3.5) MoE W8A8 (MXFP8)     | `junlin_qwen3_moe_w8a8` | ✅ 在线已实现（`mxfp8`，`NPUMXFP8FusedMoEMethod`，仅 FusedMoE/TP；e2e 待 NPU 服务器验证） | ❌ 待实现               |
| LLM (Qwen3 & 3.5) MoE W4A8 (MXFP4/8)   | 待定                   | ❌ 待实现               | ❌ 待实现               |
| LLM (Qwen3 & 3.5) MoE W4A4 (MXFP4)     | 待定                   | ❌ 待实现               | ❌ 待实现               |

## 关键代码路径

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
- `--quantization mxfp4_npu` → `layers/quantization/npu_mxfp4.py` → `NPUMxfp4Config` → `NPUMXFP4W4A8LinearMethod`（双级 W4A8）
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

> 详细 API 参考和实现模式见 `/mxfp4-impl-ref` skill。

## 开发工具

- **pre-commit**：`sglang/` 是独立 git 仓库，pre-commit 必须在 `sglang/` 目录内运行：
  ```bash
  pre-commit run --all-files  # 在 sglang/ 目录下执行
  ```
  Windows 上 CI 脚本已修复编码和路径分隔符兼容性问题（`check_workflow_job_names.py`、`check_registered_tests.py`）。

## 调试 CI 失败

**GitHub Actions 日志需要登录才能查看。** 遇到 CI 失败链接时，如果通过powershell gh访问获取不到信息，必须第一时间告知用户情况。不得自行猜测或绕弯子抓取。

## 代码提交
代码提交时必须使用gitmoji-commit这个skill。每次提交代码后，调用claude-md-improver这个skill来更新CLAUDE.md。

### 子模块提交流程
1. 在子模块（`sglang/`）内完成代码修改后，提交并更新子模块内的 CLAUDE.md
2. 回到主仓，更新主仓 CLAUDE.md（记录子模块变更摘要）
3. 最后提交主仓（含 CLAUDE.md 更新 + 子模块指针更新）

## Agent Team 协作模式（本分支：**禁用**）

本分支 **不** 启用 agent team 模式。新会话默认 **单 agent**，**不要** 主动调用 `TeamCreate` / 不要 spawn 命名子 agent / 不要走 planner-generator-evaluator 三角色流水线。即使用户说「开 team」之类的关键词，也请先确认是否要切到 `agent-team` 分支再启用。

如需 agent team 协作（planner → generator → evaluator，带「重生成」与「重规划」回环），请：
1. `git checkout agent-team`
2. 参见该分支 CLAUDE.md 末尾「Agent Team 协作模式（`pge-team`）」节

> 分支即模式，互不打扰。
