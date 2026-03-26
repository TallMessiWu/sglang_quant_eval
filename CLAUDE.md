# CLAUDE.md

## Project Overview

本仓库 (`sglang_quant_eval`) 用于研究和实现 SGLang **Diffusion 侧**在华为 Ascend NPU 上的 MXFP8/MXFP4 量化适配工作（Wan2.2 等 Diffusion 模型）。**不涉及 LLM serving (`srt`) 侧。**

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424)
- **目标**: 在 Ascend NPU 上为 SGLang Diffusion (`multimodal_gen`) 适配 MXFP8/MXFP4 量化
- **Fork**: https://github.com/TallMessiWu/sglang

## 仓库结构

```
sglang_quant_eval/
├── sglang/                          # SGLang 源码 (submodule, fork of sgl-project/sglang)
├── MindIE-SD/                       # MindIE-SD 源码 (submodule, 参考实现)
├── msmodelslim/                     # msmodelslim 源码 (submodule)
├── run_wan22_ti2v_mxfp8_online.py   # Wan2.2 MXFP8 在线量化推理脚本
├── sglang_mxfp8_ascend_research.md  # 详细研究报告 (英文)
├── sglang_mxfp8_ascend_research_zh.md  # 详细研究报告 (中文)
└── CLAUDE.md
```

## sglang 子模块分支说明

| 分支 | 角色 | 说明 |
|------|------|------|
| `junlin` | **主线（不动）** | 干净的基础分支，不在此提交特性 |
| `junlin_mxfp4` | **当前工作分支** | 从 `junlin` 克隆，所有 Diffusion 量化实现放这里 |

> **规则**：所有代码改动必须提交到 `junlin_mxfp4`，`junlin` 仅作为 upstream 基线使用。

## 量化策略

### 策略 A — 在线量化（Online）
加载原始 FP16/BF16 模型权重，在 `process_weights_after_loading` 中通过 `npu_dynamic_mx_quant` 实时量化为 MXFP8/MXFP4，无需预量化权重文件。使用 `--quantization mxfp8` / `--quantization mxfp4` 触发。

### 策略 B — 离线预量化加载（ModelSlim）
加载 msmodelslim 导出的预量化权重（FP8/FP4 packed + uint8 scale），通过 `ModelSlim` scheme 机制加载。使用 `--quantization modelslim` + 指定量化模型路径触发，scheme 由 `quant_model_description.json` 自动检测。

## 实现进度

### ✅ 已完成：Diffusion MXFP8 在线量化（`junlin_mxfp4` 分支，策略 A）

| 文件 | 变更 |
|------|------|
| `sglang/.../multimodal_gen/runtime/layers/quantization/mxfp8_npu.py` | **新增** `MXFP8Config` + `NPUMXFP8DiffusionLinearMethod` |
| `sglang/.../multimodal_gen/runtime/layers/quantization/__init__.py` | **修改** 注册 `"mxfp8"` |
| `sglang/.../multimodal_gen/runtime/server_args.py` | **修改** 新增 `quantization` 字段 |
| `sglang/.../multimodal_gen/runtime/loader/.../transformer_loader.py` | **修改** 显式 quantization 优先于自动检测 |

### ✅ 已完成：Diffusion MXFP8 离线预量化加载（`junlin_mxfp4` 分支，策略 B）

| 文件 | 变更 |
|------|------|
| `sglang/.../multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py` | **新增** `ModelSlimMXFP8Scheme`（加载 msmodelslim 预量化 MXFP8 权重）|

### ❌ 待实现：Diffusion MXFP4 在线量化（`junlin_mxfp4` 分支，策略 A）

加载原始 FP16/BF16 权重，推理时在线量化到 MXFP4（W4A4 MXFP4）。

| 文件 | 操作 | 说明 |
|------|------|------|
| `sglang/.../multimodal_gen/runtime/layers/quantization/mxfp4_npu.py` | **新增** | `MXFP4Config` + `NPUMXFP4DiffusionLinearMethod` |
| `sglang/.../multimodal_gen/runtime/layers/quantization/__init__.py` | **修改** | 注册 `"mxfp4"` |

### ❌ 待实现：Diffusion MXFP4 离线预量化加载（`junlin_mxfp4` 分支，策略 B）

加载 msmodelslim 导出的 MXFP4 权重（FP4 packed + uint8 scale）。

| 文件 | 操作 | 说明 |
|------|------|------|
| `sglang/.../multimodal_gen/runtime/layers/quantization/modelslim_mxfp4_scheme.py` | **新增** | `ModelSlimMXFP4Scheme` |
| `sglang/.../multimodal_gen/runtime/layers/quantization/modelslim.py` | **修改** | `_get_scheme_from_parts()` 添加 `W4A4_MXFP4` 分支 |

### 📋 参考：LLM 侧 MXFP4/MXFP8 实现（`junlin_llm` 分支，已完成，暂不合并）

LLM Serving (`srt`) 侧的适配，供未来参考或合并：

| 文件 | 变更 |
|------|------|
| `sglang/.../hardware_backend/npu/quantization/mxfp8_method_npu.py` | **新增** `NPUMXFP8LinearMethod`（在线 MXFP8）|
| `sglang/.../layers/quantization/fp8.py` | `get_quant_method()` 添加 NPU 分支 |

## 关键代码路径

### SGLang Diffusion 量化系统 (`multimodal_gen`)
- 量化层: `sglang/python/sglang/multimodal_gen/runtime/layers/quantization/`
  - 注册表: `__init__.py` → `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` dict
  - 基类: `configs/base_config.py` → `QuantizationConfig`, `QuantizeMethodBase`
  - Linear 基类: `../linear.py` → `LinearBase`, `LinearMethodBase`
  - **MXFP8 在线 (✅)**: `mxfp8_npu.py` → `MXFP8Config`, `NPUMXFP8DiffusionLinearMethod`
  - **ModelSlim MXFP8 离线 (✅)**: `modelslim_mxfp8_scheme.py` → `ModelSlimMXFP8Scheme`
  - ModelSlim 分发: `modelslim.py` → `ModelSlimConfig`, `_get_scheme_from_parts()`
  - Scheme 基类: `modelslim_scheme.py` → `ModelSlimLinearScheme`
  - ModelOpt FP4: `modelopt_quant.py`
- 参数类型: `sglang/python/sglang/multimodal_gen/runtime/models/parameter.py` → `ModelWeightParameter`
- 服务配置: `sglang/python/sglang/multimodal_gen/runtime/server_args.py` → `ServerArgs`
- Transformer 加载: `sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py`
- Wan2.2 Diffusion: `sglang/python/sglang/multimodal_gen/`

### MindIE-SD 参考实现
- 量化层: `MindIE-SD/mindiesd/quantization/layer.py`
  - **MXFP8**: `W8A8MXFP8QuantLinear` — 使用 `npu_dynamic_mx_quant` + `npu_quant_matmul(group_sizes=[1,1,32])`
  - **MXFP4 (DualScale)**: `W4A4MXFP4DualQuantLinear` — 使用双级量化 API（见下方）
- 量化入口: `MindIE-SD/mindiesd/quantization/quantize.py`
- 量化算法枚举: `MindIE-SD/mindiesd/quantization/mode.py` → `QuantAlgorithm.W4A4_MXFP4_DUALSCALE`

#### MXFP4 关键 API（DualScale 模式，来自 `W4A4MXFP4DualQuantLinear`）

MindIE-SD 的 MXFP4 使用**双级量化**（与 MXFP8 的单级不同）：

```python
# 激活量化：返回量化值 + 两级 scale
x1, l0_scale, l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=None)

# 权重 dtype cast（离线预量化权重加载时）
weight = torch_npu.npu_dtype_cast(weight.npu(), torch_npu.float4_e2m1fn_x2)

# 矩阵乘法（双级 scale）
output = torch_npu.npu_dual_level_quant_matmul(
    x1, weight, l0_scale, weight_dual_scale, l1_scale, weight_scale,
    bias=bias, output_dtype=dtype
)
```

**与 MXFP8 的关键差异：**
| | MXFP8 | MXFP4 (DualScale) |
|--|-------|-------------------|
| 激活量化 API | `npu_dynamic_mx_quant(..., dst_type=float8_e4m3fn)` | `npu_dynamic_dual_level_mx_quant(x)` |
| 矩阵乘法 API | `npu_quant_matmul(..., group_sizes=[1,1,32])` | `npu_dual_level_quant_matmul(...)` |
| Scale 数量 | 1 级（per-block e8m0） | 2 级（l0 + l1） |
| 权重 dtype | `float8_e4m3fn` | `float4_e2m1fn_x2`（2个FP4打包）|

> **在线量化注意**：MindIE-SD 的 MXFP4 目前只有**离线**（预量化权重加载）路径。在线量化（从 FP16 直接量化）需要确认 `npu_dynamic_dual_level_mx_quant` 是否也能用于权重量化，或者是否有独立的权重在线量化 API。

### msmodelslim MXFP4 参考实现
- MXFP4 量化 IR: `msmodelslim/msmodelslim/ir/w4a4_mx_dynamic.py` → `W4A4MXDynamicPerBlockFakeQuantLinear`
- Ascend 权重保存: `msmodelslim/msmodelslim/core/quant_service/modelslim_v1/save/ascendv1.py` → `on_w4a4_mx_dynamic_per_block()`

## 核心 Ascend NPU API (torch_npu)

### 已知可用（MXFP8）
| API | 用途 |
|-----|------|
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)` | MXFP8 动态量化，返回 `(qx, input_scale)` |
| `torch_npu.npu_quant_matmul(..., group_sizes=[1,1,32])` | MXFP8 量化矩阵乘法（block_size=32）|
| `torch_npu.npu_dtype_cast(x, torch_npu.float8_e4m3fn)` | int8 → float8_e4m3fn 转换 |
| `torch_npu.npu_dynamic_quant(x)` | 动态 per-token INT8 量化 |
| `torch_npu.npu_quantize(x, scale, offset)` | 静态 INT8 量化 |
| `torch_npu.npu_format_cast(tensor, 29)` | NZ 格式转换（INT8 性能优化）|
| `torch_npu.float8_e4m3fn` / `torch_npu.float8_e8m0fnu` | FP8 数据类型 / scale factor 类型 |

### MXFP4 NPU API（来自 MindIE-SD 参考实现确认）
| API | 用途 | 状态 |
|-----|------|------|
| `torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=None)` | MXFP4 双级动态激活量化，返回 `(x1, l0_scale, l1_scale)` | ✅ 已在 MindIE-SD 中确认 |
| `torch_npu.npu_dual_level_quant_matmul(x1, w, l0_scale, w_dual_scale, l1_scale, w_scale, ...)` | MXFP4 双级量化矩阵乘法 | ✅ 已在 MindIE-SD 中确认 |
| `torch_npu.npu_dtype_cast(w, torch_npu.float4_e2m1fn_x2)` | 权重 cast 为 FP4 打包格式（2个E2M1打包为1字节） | ✅ 已在 MindIE-SD 中确认 |
| `torch_npu.float4_e2m1fn_x2` | FP4 E2M1 打包数据类型（2x） | ✅ 已确认类型名称 |
| MXFP4 权重在线量化 API | 从 FP16/BF16 权重直接量化到 MXFP4 | ❓ 待确认（MindIE-SD 只有离线加载路径）|

## MXFP4 权重格式（msmodelslim 导出）

基于 `ascendv1.py` 的 `on_w4a4_mx_dynamic_per_block()` 分析：

| 字段 | dtype | shape | 说明 |
|------|-------|-------|------|
| `weight` | `float8_e4m3fn`（用作 FP4 打包容器） | `[out, in/2]` | 2 个 FP4 值打包为 1 字节，推理时需 dtype cast 到 FP4 |
| `weight_scale` | `uint8` | `[out, in/32]` | e8m0 scale 偏移 +127（即 `scale_e8m0 + 127`），推理时需还原 |
| `bias`（可选） | `float32` | `[out]` | |
| `block_size` | — | — | 32 |

> **注意**：weight_scale 偏移原因：e8m0 范围 [-127, 128] 偏移后正好覆盖 uint8 [0, 255]。还原公式：`actual_scale = weight_scale.to(int16) - 127`

## MXFP4 在线量化实现模式（参考 MXFP8）

```python
# process_weights_after_loading（在线量化，加载原始权重后执行）
weight_fp = layer.weight.data.to(torch.bfloat16)
if not weight_fp.is_npu:
    weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")
qw, w_scale = torch_npu.npu_dynamic_mx_quant(weight_fp, dst_type=<FP4_DTYPE>)
layer.weight = Parameter(qw, requires_grad=False)
layer.weight_scale_inv = Parameter(w_scale, requires_grad=False)

# apply（推理）
qx, input_scale = torch_npu.npu_dynamic_mx_quant(x_2d, dst_type=<FP4_DTYPE>)
output = torch_npu.npu_quant_matmul(
    qx, layer.weight.T, layer.weight_scale_inv.T,
    scale_dtype=<E8M0_DTYPE>,
    pertoken_scale=input_scale,
    pertoken_scale_dtype=<E8M0_DTYPE>,
    bias=bias.to(torch.float32) if bias is not None else None,
    output_dtype=original_dtype,
    group_sizes=[1, 1, 32],
)
```

## ModelSlim Scheme 开发模式（策略 B 离线加载）

添加新 ModelSlim Scheme 的标准流程：

1. **新建 Scheme 文件**: `multimodal_gen/runtime/layers/quantization/modelslim_<name>.py`，继承 `ModelSlimLinearScheme`，实现：
   - `create_weights()` — 注册权重和 scale 参数
   - `process_weights_after_loading()` — 权重格式转换（dtype cast、reshape、NZ format）
   - `apply_weights()` — 前向推理（量化激活 + 量化矩阵乘）
2. **分发**: 在 `modelslim.py` 的 `_get_scheme_from_parts()` 添加 `elif quant_type == "..."` 分支

## 注意事项

- **分支规则**：所有代码改动只提交到 `junlin_mxfp4`，不动 `junlin`
- **CANN 版本**: ≥ 8.0.RC3 才支持 `npu_dynamic_mx_quant`（MXFP8）；MXFP4 支持的最低 CANN 版本待确认
- **硬件**: 开发测试需要 Atlas 800I A2/A3
- **bias 精度**: `npu_quant_matmul` 要求 bias 为 `float32`
- **tensor reshape**：diffusion 输入可能是 3D `[batch, seq, hidden]`，`npu_dynamic_mx_quant` 需要 2D 输入，apply 中需先 reshape 再 restore
- **CPU offload**：`dit_cpu_offload` 默认 True，`process_weights_after_loading` 中需手动将权重移到 NPU 后再调用 `npu_dynamic_mx_quant`
- **MXFP4 FP4 打包**：2 个 E2M1 FP4 值打包为 1 个字节，传给 `npu_quant_matmul` 时需确认 Ascend 内核期望的打包格式
- 与社区 YChange01 协调 MXFP8/MXFP4 工作分工（已在 Issue #14424 认领）
