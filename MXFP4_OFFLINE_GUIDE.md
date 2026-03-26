# SGLang Diffusion MXFP4 离线预量化加载 (策略 B)

## 概述

本指南说明如何使用 SGLang Diffusion 在 Ascend NPU 上加载 msmodelslim 预量化的 MXFP4 权重进行推理。

### 关键特性
- **策略 B（离线）**: 从 msmodelslim 导出的预量化 MXFP4 权重直接加载
- **无需在线量化**: 权重加载时已是 FP4 打包格式，省去实时量化开销
- **双级量化推理**: 使用 `npu_dynamic_dual_level_mx_quant` + `npu_dual_level_quant_matmul` API
- **性能优化**: 相比在线量化（策略 A），跳过权重量化步骤

## 前置条件

### 硬件
- Ascend NPU (Atlas 800I A2/A3)

### 软件
- CANN >= 8.0.RC3 (支持 `npu_dynamic_dual_level_mx_quant`, `npu_dual_level_quant_matmul`)
- torch_npu 已安装
- SGLang 从 `junlin_mxfp4_offline` 分支编译

### 预量化模型
使用 msmodelslim 导出的预量化权重，需要包含：
```
<quant_model_dir>/
├── quant_model_description.json        # 量化描述文件（自动检测 W4A4_MXFP4）
├── quant_model_weights.safetensors     # 或多个分片权重文件
└── config.json                         # 模型配置
```

## 工作流程

### 第 1 步: 生成预量化权重（使用 msmodelslim）

使用 msmodelslim 将原始 Wan2.2 模型量化为 MXFP4：

```bash
python -m msmodelslim.cli.quantize \
    --model_dir /path/to/Wan2.2-TI2V-5B-Diffusers \
    --quantize_algorithm W4A4_MXFP4 \
    --output_dir /path/to/Wan2.2-TI2V-5B-Diffusers-MXFP4
```

验证导出的量化文件:
```bash
ls -la /path/to/Wan2.2-TI2V-5B-Diffusers-MXFP4/
# 应该包含 quant_model_description.json 和 quant_model_weights.safetensors
```

### 第 2 步: 运行推理脚本

使用 `run_wan22_ti2v_mxfp4_offline.py` 进行推理：

```bash
python run_wan22_ti2v_mxfp4_offline.py \
    --model-path /path/to/Wan2.2-TI2V-5B-Diffusers-MXFP4 \
    --image-path input.jpg \
    --num-frames 81 \
    --num-inference-steps 50 \
    --output-dir ./outputs_mxfp4_offline
```

#### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | - | 预量化 MXFP4 模型路径（必须包含 `quant_model_description.json`） |
| `--image-path` | `gyro.jpg` | 输入图片路径 |
| `--prompt` | (见脚本) | 文本提示词 |
| `--num-frames` | 81 | 生成帧数（需满足 `(n-1) % 4 == 0`） |
| `--num-inference-steps` | 50 | 推理迭代步数 |
| `--height` | 704 | 视频高度 |
| `--width` | 1280 | 视频宽度 |
| `--fps` | 24 | 输出视频帧率 |
| `--seed` | 42 | 随机种子 |
| `--num-gpus` | 1 | 使用的 NPU 数量 |
| `--output-dir` | `./outputs_mxfp4_offline` | 输出目录 |

## 实现细节

### ModelSlimMXFP4Scheme

位置: `sglang/python/sglang/multimodal_gen/runtime/layers/quantization/modelslim_mxfp4_scheme.py`

#### 权重加载流程

1. **权重读取**
   - `weight`: float8_e4m3fn (FP4 打包容器), shape [out, in]
   - `weight_scale`: uint8, shape [out, in/32]（+127 偏移）

2. **权重处理** (`process_weights_after_loading`)
   - 将 `weight` dtype 转换为 `float4_e2m1fn_x2` (FP4 打包格式)
   - 将 `weight_scale` reshape 为 [out, in/64, 2] (双级 scale 结构)

3. **推理** (`apply_weights`)
   - 激活量化: `npu_dynamic_dual_level_mx_quant(x)` → (x1, l0_scale, l1_scale)
   - 双级量化矩阵乘: `npu_dual_level_quant_matmul(...)`
   - 3D 输入自动 reshape 处理

### 量化流程自动检测

在 `transformer_loader.py` 中，加载顺序如下：

1. 检查 `ServerArgs.quantization` 字段（如显式设置 `--quantization modelslim`）
2. 自动从 `quant_model_description.json` 检测量化类型
3. 若检测到 `W4A4_MXFP4`，自动选择 `ModelSlimMXFP4Scheme`

## 性能特性

### 优势
- ✅ **无在线量化开销**: 权重已预量化，加载时直接可用
- ✅ **双级量化精度**: 相比单级 MXFP8，MXFP4 更精细的量化控制
- ✅ **内存节省**: FP4 权重占用 MXFP8 的 1/2

### 权衡
- 需要预先生成量化权重文件（msmodelslim 量化过程）
- 模型文件集中存储，不如在线量化灵活

## 故障排查

### 错误: "No modelslim compatible scheme was found"
**原因**: `quant_model_description.json` 中的量化类型不被识别
**解决**: 检查 `quant_model_description.json` 中权重的 `quant_type` 字段是否为 `W4A4_MXFP4`

### 错误: "npu_dynamic_dual_level_mx_quant not found"
**原因**: CANN 版本过低
**解决**: 升级 CANN >= 8.0.RC3

### 权重加载失败
**原因**: msmodelslim 导出的权重格式与预期不符
**解决**:
1. 验证权重文件: `quant_model_weights.safetensors` 存在且格式正确
2. 检查 `quant_model_description.json` 中的权重 key 与模型 hook 名称匹配

## 对比: 在线 vs 离线

| 指标 | 在线量化 (MXFP8) | 离线量化 (MXFP4) |
|------|-----------------|-----------------|
| 权重加载 | FP16/BF16 → 实时量化 | 预量化 FP4 → 直接加载 |
| 量化精度 | 单级 (e8m0) | 双级 (l0, l1) |
| 内存使用 | 原始 FP16 + 量化缓存 | FP4 (1/4 FP32) |
| 灵活性 | 高（任意模型） | 中（需预量化） |
| 性能 | 有量化开销 | 无量化开销 |

## 技术参考

### NPU API

- `torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=None)`
  - 返回: (x1, l0_scale, l1_scale)

- `torch_npu.npu_dual_level_quant_matmul(x1, w, l0_scale, w_dual_scale, l1_scale, w_scale, ...)`
  - 执行双级量化矩阵乘

- `torch_npu.npu_dtype_cast(x, torch_npu.float4_e2m1fn_x2)`
  - 权重 dtype 转换

### 权重格式 (msmodelslim 导出)

```python
# 模型权重字段
weight          (dtype: float8_e4m3fn, shape: [out, in])           # FP4 打包容器
weight_scale    (dtype: uint8, shape: [out, in/32])                # e8m0 scale (+127 偏移)
bias            (dtype: float32, shape: [out]) [optional]          # 偏置项
```

## 参考资源

- **SGLang Diffusion**: https://github.com/sgl-project/sglang
- **msmodelslim**: https://gitcode.com/Ascend/msmodelslim
- **CANN 文档**: https://www.hiascend.com/en/software/cann
- **Wan2.2 模型**: https://huggingface.co/Tencent-Hunyuan/Wan2.2-TI2V-5B-Diffusers

## 许可

本项目遵循原始 SGLang 和 msmodelslim 的许可条款。

---

**最后更新**: 2026-03-26
**分支**: `junlin_mxfp4_offline`
