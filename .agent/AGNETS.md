# CLAUDE.md

## Project Overview

本仓库 (`sglang_quant_eval`) 用于研究和实现 SGLang 在华为 Ascend NPU 上的 MXFP8 量化适配工作。

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424)
- **目标**: 在 Ascend NPU 上为 SGLang 适配 MXFP8/MXFP4 量化，支持 LLM 和 Wan2.2 Diffusion 模型

## 仓库结构

```
sglang_quant_eval/
├── sglang/                          # SGLang 源码 (git clone)
├── MindIE-SD/                       # MindIE-SD 源码 (git clone, 参考实现)
├── sglang_mxfp8_ascend_research.md  # 详细研究报告
└── CLAUDE.md
```

## 关键代码路径

### SGLang 量化系统
- 量化核心: `sglang/python/sglang/srt/layers/quantization/`
  - 注册表: `__init__.py` → `QUANTIZATION_METHODS` dict
  - 基类: `base_config.py` → `QuantizationConfig`, `LinearMethodBase`, `FusedMoEMethodBase`
  - FP8/MXFP8 (CUDA): `fp8.py`, `fp8_kernel.py`, `fp8_utils.py`
  - INT8: `w8a8_int8.py`, `blockwise_int8.py`
- NPU 后端: `sglang/python/sglang/srt/hardware_backend/npu/`
  - NPU 量化: `quantization/linear_method_npu.py` (W8A8 INT8)
  - NPU MoE: `quantization/fused_moe_method_npu.py` (W4A4)
  - NPU GraphRunner: `graph_runner/npu_graph_runner.py`
- 模型加载: `sglang/python/sglang/srt/model_loader/loader.py`
- 服务参数: `sglang/python/sglang/srt/server_args.py`
- Wan2.2 Diffusion: `sglang/python/sglang/multimodal_gen/`

### MindIE-SD 参考实现 (MXFP8)
- MXFP8 层实现: `MindIE-SD/mindiesd/quantization/layer.py` → `W8A8MXFP8QuantLinear`
- 量化入口: `MindIE-SD/mindiesd/quantization/quantize.py`
- 量化模式枚举: `MindIE-SD/mindiesd/quantization/mode.py` → `QuantAlgorithm`
- 量化配置: `MindIE-SD/mindiesd/quantization/config.py`

## 核心 Ascend NPU API (torch_npu)

| API | 用途 |
|-----|------|
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)` | MXFP8 动态量化 |
| `torch_npu.npu_quant_matmul(..., group_sizes=[1,1,32])` | 量化矩阵乘法 |
| `torch_npu.npu_quantize(x, scale, offset)` | 静态 INT8 量化 |
| `torch_npu.npu_dynamic_quant(x)` | 动态 per-token INT8 量化 |
| `torch_npu.float8_e4m3fn` / `torch_npu.float8_e8m0fnu` | FP8 数据类型 |

## 添加新量化方法的标准流程

1. **创建量化 Config 类**: 继承 `QuantizationConfig`, 实现 `get_quant_method()` 等
2. **创建量化 Method 类**: 继承 `LinearMethodBase`, 实现 `create_weights()`, `apply()`, `process_weights_after_loading()`
3. **注册**: 在 `__init__.py` 的 `QUANTIZATION_METHODS` 中添加
4. **NPU 特化** (可选): 在 `hardware_backend/npu/quantization/` 下创建 NPU 专用实现
5. **测试**: 在 `test/srt/ascend/` 下添加测试

## 注意事项

- 与社区 YChange01 协调 MXFP8/MXFP4 工作分工 (已在 Issue #14424 认领)
- CANN 版本需 ≥ 8.0.RC3 以支持 `npu_dynamic_mx_quant`
- 需要 Atlas 800I A2/A3 硬件进行开发测试
- SGLang 的 Diffusion 子系统 (`multimodal_gen`) 与 LLM Serving (`srt`) 是独立的两套代码

## 交互规范 (Interaction)
-   **语言**: 所有输出（思考、回复、文档、提交信息）**必须使用中文**。