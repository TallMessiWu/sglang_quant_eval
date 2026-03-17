# CLAUDE.md

## Project Overview

本仓库 (`sglang_quant_eval`) 用于研究和实现 SGLang 在华为 Ascend NPU 上的 MXFP8 量化适配工作。

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424)
- **目标**: 在 Ascend NPU 上为 SGLang 适配 MXFP8/MXFP4 量化，支持 LLM 和 Wan2.2 Diffusion 模型
- **Fork**: https://github.com/TallMessiWu/sglang (main 分支)

## 仓库结构

```
sglang_quant_eval/
├── sglang/                          # SGLang 源码 (submodule, fork of sgl-project/sglang)
├── MindIE-SD/                       # MindIE-SD 源码 (submodule, 参考实现)
├── msmodelslim/                     # msmodelslim 源码 (submodule)
├── sglang_mxfp8_ascend_research.md  # 详细研究报告 (英文)
├── sglang_mxfp8_ascend_research_zh.md  # 详细研究报告 (中文)
└── CLAUDE.md
```

## 实现进度

### ✅ 已完成：Path A — msmodelslim MXFP8 权重支持

| 文件 | 变更 |
|------|------|
| `sglang/.../modelslim/schemes/modelslim_mxfp8.py` | **新增** `ModelSlimMXFP8` Scheme |
| `sglang/.../modelslim/schemes/__init__.py` | 注册 `ModelSlimMXFP8` |
| `sglang/.../modelslim/modelslim.py` | `_get_scheme_from_parts()` 添加 `W8A8_MXFP8` 分支 |

**使用方式**（msmodelslim 导出 MXFP8 权重后）:
```bash
python3 -m sglang.launch_server \
    --model-path /path/to/mxfp8-quantized-model \
    --quantization modelslim
```

### ❌ 待实现：Path B — Online MXFP8（`--quantization mxfp8`）

需新增 `sglang/.../hardware_backend/npu/quantization/mxfp8_method_npu.py` 并修改 `fp8.py` 的 `get_quant_method()` 添加 NPU 分支。

## 关键代码路径

### SGLang 量化系统
- 量化核心: `sglang/python/sglang/srt/layers/quantization/`
  - 注册表: `__init__.py` → `QUANTIZATION_METHODS` dict
  - 基类: `base_config.py` → `QuantizationConfig`, `LinearMethodBase`, `FusedMoEMethodBase`
  - FP8/MXFP8 (CUDA): `fp8.py`, `fp8_kernel.py`, `fp8_utils.py`
  - INT8: `w8a8_int8.py`, `blockwise_int8.py`
- **ModelSlim (NPU)**: `sglang/python/sglang/srt/layers/quantization/modelslim/`
  - 配置类: `modelslim.py` → `ModelSlimConfig`, `_get_scheme_from_parts()`
  - Scheme 基类: `schemes/modelslim_scheme.py`
  - **MXFP8 Scheme**: `schemes/modelslim_mxfp8.py` → `ModelSlimMXFP8` ✅
  - W8A8 Scheme: `schemes/modelslim_w8a8_int8.py` → `ModelSlimW8A8Int8`
  - W4A4 Scheme: `schemes/modelslim_w4a4_int4.py` → `ModelSlimW4A4Int4`
- NPU 后端: `sglang/python/sglang/srt/hardware_backend/npu/`
  - NPU 量化内核: `quantization/linear_method_npu.py`（W8A8/W4A4 INT 系列）
  - NPU MoE: `quantization/fused_moe_method_npu.py`
  - NPU GraphRunner: `graph_runner/npu_graph_runner.py`
- 模型加载: `sglang/python/sglang/srt/model_loader/loader.py`
- Wan2.2 Diffusion: `sglang/python/sglang/multimodal_gen/`

### MindIE-SD 参考实现 (MXFP8)
- MXFP8 层实现: `MindIE-SD/mindiesd/quantization/layer.py` → `W8A8MXFP8QuantLinear`
- 量化入口: `MindIE-SD/mindiesd/quantization/quantize.py`
- 量化模式枚举: `MindIE-SD/mindiesd/quantization/mode.py` → `QuantAlgorithm`
- 量化配置: `MindIE-SD/mindiesd/quantization/config.py`

## 核心 Ascend NPU API (torch_npu)

| API | 用途 |
|-----|------|
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)` | MXFP8 动态量化，返回 `(qx, input_scale)` |
| `torch_npu.npu_quant_matmul(..., group_sizes=[1,1,32])` | MXFP8 量化矩阵乘法（block_size=32）|
| `torch_npu.npu_dtype_cast(x, torch_npu.float8_e4m3fn)` | int8 → float8_e4m3fn 转换 |
| `torch_npu.npu_dynamic_quant(x)` | 动态 per-token INT8 量化 |
| `torch_npu.npu_quantize(x, scale, offset)` | 静态 INT8 量化 |
| `torch_npu.npu_format_cast(tensor, 29)` | NZ 格式转换（INT8 性能优化）|
| `torch_npu.float8_e4m3fn` / `torch_npu.float8_e8m0fnu` | FP8 数据类型 / scale factor 类型 |

## ModelSlim Scheme 开发模式

添加新 ModelSlim Scheme 的标准流程（Path A 模式）：

1. **新建 Scheme 文件**: `schemes/modelslim_<name>.py`，继承 `ModelSlimLinearScheme`，实现：
   - `create_weights()` — 注册权重和 scale 参数
   - `process_weights_after_loading()` — 权重格式转换（dtype cast、reshape、NZ format）
   - `apply_weights()` — 前向推理（量化激活 + 量化矩阵乘）
2. **注册**: 在 `schemes/__init__.py` 中 import 并加入 `__all__`
3. **分发**: 在 `modelslim.py` 的 `_get_scheme_from_parts()` 添加 `elif quant_type == "..."` 分支
4. **测试**: 在 `sglang/test/srt/ascend/` 下添加测试

## 注意事项

- **CANN 版本**: ≥ 8.0.RC3 才支持 `npu_dynamic_mx_quant`
- **硬件**: 开发测试需要 Atlas 800I A2/A3
- **weight_scale 格式**: msmodelslim 导出为 `uint8` (存储 `float8_e8m0fnu`)，shape `[out, in/32 * 2]`，加载后需 reshape 为 `[out, in/32, 2]`
- **weight 格式**: msmodelslim 导出为 `int8`，推理时需 `npu_dtype_cast` 转为 `float8_e4m3fn`
- **bias 精度**: `npu_quant_matmul` 要求 bias 为 `float32`
- **SGLang Diffusion 子系统** (`multimodal_gen`) 与 LLM Serving (`srt`) 是独立的两套代码
- 与社区 YChange01 协调 MXFP8/MXFP4 工作分工（已在 Issue #14424 认领）
