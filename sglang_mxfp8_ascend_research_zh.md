# SGLang MXFP8 量化在华为 Ascend NPU 上的适配研究报告

> 日期: 2026-03-16
> 关联 Issue: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424)
> 目标: 在华为 Ascend NPU 上为 SGLang 适配 MXFP8 量化，支持 Wan2.2 等模型

---

## 一、SGLang 简介

### 1.1 什么是 SGLang

SGLang 是由 LMSYS 团队开发的高性能大模型推理服务框架，全球已部署超过 40 万 GPU。核心特性包括:

- **RadixAttention** 前缀缓存
- 零开销 CPU 调度器
- Prefill-Decode 解耦
- 投机解码、连续批处理、分页注意力
- 张量/流水线/专家/数据并行
- 结构化输出生成
- OpenAI 兼容 API

### 1.2 基本使用方式

```bash
# 安装
pip install sglang

# 启动服务
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 30000 --host 0.0.0.0

# Python API
import sglang as sgl
llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
outputs = llm.generate(["Hello!"], {"max_new_tokens": 100})
```

### 1.3 硬件支持

| 硬件 | 状态 |
|------|------|
| NVIDIA GPU (CUDA) | 完整支持 |
| AMD GPU (ROCm) | 支持 MI300/MI350 |
| Intel Xeon CPU | 支持 |
| Google TPU | 支持 |
| **华为 Ascend NPU** | **已有基础支持，量化持续完善中** |

### 1.4 代码架构

SGLang Serving Runtime (`sglang/python/sglang/srt/`) 主要模块:

```
srt/
├── entrypoints/        # HTTP/gRPC 服务入口 (FastAPI, OpenAI兼容)
├── model_executor/     # 模型执行引擎 (CUDA/CPU/NPU GraphRunner)
├── model_loader/       # 权重加载 (safetensors, GGUF, pt等)
├── models/             # 100+ 模型实现 (Llama, Qwen, DeepSeek, GLM等)
├── layers/
│   ├── quantization/   # ★ 量化核心目录 (20+种量化方法)
│   ├── attention/      # 注意力层
│   └── moe/            # MoE 层
├── hardware_backend/
│   └── npu/            # ★ Ascend NPU 后端
├── managers/           # 内存和调度管理
├── distributed/        # TP, EP, DP 并行
└── speculative/        # 投机解码
```

视频/图像生成独立子系统:
```
multimodal_gen/         # ★ Wan2.2 等 Diffusion 模型在此
├── configs/            # 模型配置、量化配置、Pipeline配置
├── runtime/
│   ├── models/         # DiT, VAE 模型实现
│   └── pipelines/      # 推理 Pipeline (Wan, HunyuanVideo等)
└── registry.py         # 模型注册表
```

---

## 二、SGLang 现有量化支持

### 2.1 INT8 量化 (已支持)

SGLang 完整支持 INT8 量化，有三种方式:

#### a) W8A8 INT8 (per-channel 权重, per-token 激活)
```bash
python3 -m sglang.launch_server \
    --model-path <int8-quantized-model> \
    --quantization w8a8_int8
```
- 实现: `srt/layers/quantization/w8a8_int8.py`
- 使用 `sgl_kernel.int8_scaled_mm` (CUDA)
- 支持 NVIDIA、AMD、Intel

#### b) Block-wise INT8
```bash
python3 -m sglang.launch_server \
    --model-path <blockwise-int8-model> \
    --quantization blockwise_int8
```
- 实现: `srt/layers/quantization/blockwise_int8.py`
- 可配置 block size

#### c) TorchAO INT8
```bash
# Weight-only INT8
python3 -m sglang.launch_server \
    --model-path <model> --torchao-config int8wo

# Dynamic INT8
python3 -m sglang.launch_server \
    --model-path <model> --torchao-config int8dq --disable-cuda-graph
```

### 2.2 FP8/MXFP8 量化 (CUDA 侧已支持)

| 方法 | CLI 参数 | 描述 |
|------|----------|------|
| Online FP8 | `--quantization fp8` | 运行时动态缩放，无需预量化 |
| **Online MXFP8** | `--quantization mxfp8` | **微缩放 FP8, block-size-32** |
| W8A8 FP8 | `--quantization w8a8_fp8` | 预量化权重 + 动态激活 |
| ModelOpt FP8 | `--quantization modelopt_fp8` | NVIDIA ModelOpt 预量化 |
| FBGEMM FP8 | `--quantization fbgemm_fp8` | FBGEMM 后端 |
| MXFP4 | `--quantization mxfp4` | 微缩放 FP4 |

**关键点**: MXFP8 在 CUDA 侧已通过 PR #17449 合并支持，但**在 Ascend NPU 侧尚未适配**。这正是 Issue #14424 要求的工作。

### 2.3 完整量化方法注册表

文件: `srt/layers/quantization/__init__.py`

```python
QUANTIZATION_METHODS = {
    "fp8": Fp8Config,
    "mxfp8": Fp8Config,          # use_mxfp8=True
    "mxfp4": Mxfp4Config,
    "blockwise_int8": BlockInt8Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "modelopt_fp8": ModelOptFp8Config,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    "modelslim": ModelSlimConfig,  # NPU 专用
    # ... 20+ 种
}
```

---

## 三、Ascend NPU 后端现状

### 3.1 已有 NPU 代码结构

```
srt/hardware_backend/npu/
├── quantization/
│   ├── linear_method_npu.py      # W8A8 INT8 NPU Linear
│   └── fused_moe_method_npu.py   # W4A4 MoE NPU
├── graph_runner/
│   ├── npu_graph_runner.py       # NPU 图执行器
│   ├── eagle_draft_npu_graph_runner.py
│   └── vit_npu_graph_runner.py
├── modules/
│   ├── deepseek_v2_attention_mla_npu.py
│   └── ...
├── moe/topk.py
└── utils.py
```

### 3.2 NPU 已支持的量化方法

| 方法 | 状态 | 实现位置 |
|------|------|----------|
| W8A8 INT8 (静态) | ✅ 已完成 | `linear_method_npu.py` → `NPUW8A8Int8LinearMethod` |
| W8A8 INT8 (动态) | ✅ 已完成 | `linear_method_npu.py` → `NPUW8A8Int8DynamicLinearMethod` |
| W4A4 MoE | ✅ 已完成 | `fused_moe_method_npu.py` |
| W4A8 (带激活裁剪) | ✅ 已完成 | 通过 ModelSlim |
| **MXFP8** | ❌ **未实现** | **需要新增** |
| **MXFP4** | ❌ 未实现 | 需要新增 |
| KV Cache 量化 | 🔄 进行中 | -- |

### 3.3 NPU INT8 实现示例 (参考模式)

`NPUW8A8Int8LinearMethod` 的核心流程:
```python
# 1. 创建权重参数
def create_weights(self, layer, ...):
    layer.weight = Parameter(torch.empty(..., dtype=torch.int8))
    layer.weight_scale = Parameter(torch.empty(..., dtype=torch.float32))
    layer.input_scale = Parameter(torch.empty(..., dtype=torch.float32))

# 2. 加载后处理 (NPU格式转换)
def process_weights_after_loading(self, layer):
    layer.weight = npu_format_cast(layer.weight)

# 3. 推理
def apply(self, layer, x, bias=None):
    qx = torch.ops.npu.npu_quantize(x, layer.input_scale, ...)
    return torch.ops.npu.npu_quant_matmul(qx, layer.weight, layer.deq_scale, ...)
```

---

## 四、MindIE-SD 参考分析

### 4.1 MindIE-SD 简介

MindIE-SD 是华为的 Stable Diffusion 推理加速引擎，专为 Ascend NPU 设计。已经适配了多种量化方法，包括 **MXFP8**。

### 4.2 MindIE-SD 已支持的量化算法

| 类别 | 方法 |
|------|------|
| Weight-only | W8A16, W4A16, W4A16_AWQ, W8A16_GPTQ, W4A16_GPTQ |
| W8A8 INT8 | 静态/动态/逐通道/逐张量/逐token/timestep感知 |
| **MXFP8** | **W8A8_MXFP8** ✅ |
| FP8 Attention | FP8_DYNAMIC (带旋转矩阵) |

### 4.3 MindIE-SD 的 MXFP8 实现 (核心参考)

文件: `MindIE-SD/mindiesd/quantization/layer.py` → `W8A8MXFP8QuantLinear`

```python
class W8A8MXFP8QuantLinear(nn.Module):
    def forward(self, x):
        # 1. 动态 MXFP8 量化激活
        qx, x_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float8_e4m3fn
        )

        # 2. MXFP8 矩阵乘法
        output = torch_npu.npu_quant_matmul(
            qx,
            self.weight,           # float8_e4m3fn
            self.weight_scale,     # float8_e8m0fnu
            pertoken_scale=x_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32]  # MXFP8 block size = 32
        )
        return output
```

### 4.4 关键 Ascend NPU API (torch_npu)

| API | 用途 |
|-----|------|
| `torch_npu.npu_dynamic_mx_quant(x, dst_type)` | **MXFP8 动态量化** |
| `torch_npu.npu_quant_matmul(...)` | 量化矩阵乘法 (INT8/FP8) |
| `torch_npu.npu_quantize(x, scale, offset)` | 静态 INT8 量化 |
| `torch_npu.npu_dynamic_quant(x)` | 动态 per-token INT8 量化 |
| `torch_npu.npu_format_cast(tensor, 29)` | NZ 格式转换 |
| `torch_npu.npu_weight_quant_batchmatmul(...)` | Weight-only 量化矩阵乘法 |
| `torch_npu.float8_e4m3fn` | FP8 E4M3 数据类型 |
| `torch_npu.float8_e8m0fnu` | FP8 E8M0 缩放因子类型 |

### 4.5 SGLang vs MindIE-SD 相似性分析

| 维度 | SGLang | MindIE-SD | 相似度 |
|------|--------|-----------|--------|
| 量化框架 | 注册表 + Config + Method 模式 | 遍历模型 + 层替换模式 | 中等 |
| 权重加载 | 与推理内核分离 (正在重构) | JSON 描述文件 + safetensors | 设计理念相似 |
| NPU INT8 | `npu_quantize` + `npu_quant_matmul` | 相同 API | **高度相似** |
| Linear 量化 | `QuantizeMethodBase.apply()` | `QuantLinear.forward()` | 模式相同 |
| MoE 量化 | `FusedMoEMethodBase` | 无 (Diffusion 无 MoE) | 不适用 |
| MXFP8 (CUDA) | `mxfp8_group_quantize` + Triton/FlashInfer | N/A | -- |
| **MXFP8 (NPU)** | **❌ 未实现** | **✅ `npu_dynamic_mx_quant`** | **可直接借鉴** |

**核心结论**: MindIE-SD 的 MXFP8 NPU 实现 (`W8A8MXFP8QuantLinear`) 可以直接作为 SGLang NPU MXFP8 的参考模板。底层 API (`torch_npu.npu_dynamic_mx_quant`, `torch_npu.npu_quant_matmul`) 完全相同。

---

## 五、Wan2.2 模型在 SGLang 中的支持现状

### 5.1 Wan2.2 模型矩阵

SGLang 已通过 `multimodal_gen` 子系统支持 Wan2.2:

| 模型 | HuggingFace ID | 分辨率 |
|------|----------------|--------|
| Wan2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 720p |
| Wan2.2 T2V A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 480p, 720p |
| Wan2.2 I2V A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 480p, 720p |

### 5.2 Wan2.2 架构特点

- **双 Transformer 架构**: `transformer` 和 `transformer_2` 分别处理高噪声和低噪声步骤
- **DiT (Diffusion Transformer)**: 基于 Transformer 的扩散模型
- **Pipeline**: TextEncode → Denoise (多步去噪) → VAE Decode
- 支持 TP (张量并行) 和 SP (序列并行)

### 5.3 现有 Diffusion 量化

SGLang Diffusion 目前仅支持 **SVDQuant/Nunchaku** (W4A4):
```
multimodal_gen/configs/quantization.py → NunchakuSVDQuantArgs
```
- 仅限 NVIDIA GPU
- 不支持 FP8/MXFP8
- 不支持 Ascend NPU

### 5.4 使用方式

```bash
# 启动 Wan2.2 服务
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers --num-gpus 4

# Python API
from sglang import DiffGenerator
gen = DiffGenerator.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
```

---

## 六、msmodelslim 权重与 SGLang 兼容性分析

### 6.1 msmodelslim 简介

[msmodelslim](https://gitcode.com/Ascend/msmodelslim) 是华为昇腾的模型压缩工具包，支持 PTQ (训练后量化)、QAT (量化感知训练)、低秩分解、知识蒸馏、稀疏训练等。可导出 PyTorch、MindSpore、ONNX、AutoAWQ、AutoGPTQ 格式。

### 6.2 msmodelslim 输出格式

msmodelslim 量化后输出**两个文件**:

1. **JSON 描述文件**: `quant_model_description.json`
   - 扁平字典，key 为层权重名 (如 `model.layers.0.self_attn.q_proj.weight`)，value 为量化类型字符串 (如 `"W8A8_DYNAMIC"`, `"W8A8_MXFP8"`, `"FLOAT"`)
   - 多卡场景为 `quant_model_description_{rank}.json`

2. **Safetensors 权重文件**: `quant_model_weight_{algo}.safetensors`
   - 包含量化后的 weight、weight_scale、weight_offset、input_scale、input_offset、deq_scale、quant_bias 等
   - 多卡场景为 `quant_model_weight_{algo}_{rank}.safetensors`

### 6.3 SGLang 已支持的 msmodelslim 格式

**结论: SGLang 已有 modelslim 集成模块，但不支持 MXFP8。**

SGLang 的 modelslim 支持位于 `sglang/python/sglang/srt/layers/quantization/modelslim/`，采用 Scheme 模式:

| msmodelslim 量化类型 | SGLang Scheme 类 | 状态 |
|---------------------|------------------|------|
| `W8A8` (静态) | `ModelSlimW8A8Int8` → `NPUW8A8Int8LinearMethod` | ✅ 已支持 |
| `W8A8_DYNAMIC` | `ModelSlimW8A8Int8` → `NPUW8A8Int8DynamicLinearMethod` | ✅ 已支持 |
| `W4A4_DYNAMIC` | `ModelSlimW4A4Int4` | ✅ 已支持 |
| `W4A4_DYNAMIC` (MoE) | `ModelSlimW4A4Int4MoE` | ✅ 已支持 |
| `W4A8_DYNAMIC` (MoE) | `ModelSlimW4A8Int8MoE` | ✅ 已支持 |
| `W8A8_DYNAMIC` (MoE) | `ModelSlimW8A8Int8MoE` | ✅ 已支持 |
| **`W8A8_MXFP8`** | **无** | **❌ 不支持** |
| `W8A16` / `W4A16` | 无 | ❌ 不支持 |

**不支持的关键原因**: `modelslim.py` 第 179-193 行的 `_get_scheme_from_parts()` 方法仅识别 `"W8A8_DYNAMIC"`, `"W8A8"`, `"W4A4_DYNAMIC"` 三种类型，遇到 `"W8A8_MXFP8"` 会抛出 `NotImplementedError`。

```python
# modelslim.py 第 179-193 行 (当前代码)
def _get_scheme_from_parts(self, layer_name):
    quant_type = self.quant_description.get(layer_name + ".weight", "")
    if quant_type == "W8A8_DYNAMIC" or quant_type == "W8A8":
        return ModelSlimW8A8Int8(...)
    elif quant_type == "W4A4_DYNAMIC":
        return ModelSlimW4A4Int4(...)
    raise NotImplementedError("No modelslim compatible scheme was found.")
    # ↑ W8A8_MXFP8 会命中这里
```

### 6.4 适配方案: 使 SGLang 支持 msmodelslim MXFP8 权重

适配工作量较小，因为 SGLang 的 modelslim 框架已经很完善，只需在现有 Scheme 模式下新增一个 MXFP8 Scheme。

#### 需要新增的文件

**文件 1**: `sglang/python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py`

参考 `modelslim_w8a8_int8.py` 的模式，结合 MindIE-SD 的 `W8A8MXFP8QuantLinear`:

```python
import torch
import torch_npu

from sglang.srt.layers.parameter import ModelWeightParameter, ChannelQuantScaleParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme


class ModelSlimMXFP8(ModelSlimLinearScheme):
    """msmodelslim W8A8_MXFP8 量化 Scheme"""

    def __init__(self, quant_config, prefix):
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # 权重: int8 存储, 运行时 cast 为 float8_e4m3fn
        weight = ModelWeightParameter(
            data=torch.empty(
                (output_size_per_partition, input_size_per_partition),
                dtype=torch.int8),
            input_dim=1, output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # 权重缩放因子: block-size-32, 每32个元素一个 scale
        # msmodelslim 导出格式: [out_features, in_features/32 * 2]
        # reshape 后为 [out_features, in_features/32, 2]
        scale_cols = (input_size_per_partition + 31) // 32 * 2
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, scale_cols),
                dtype=torch.uint8),  # float8_e8m0fnu 存储为 uint8
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer):
        # Cast weight 为 float8_e4m3fn
        weight = layer.weight.data
        layer.weight = torch.nn.Parameter(
            torch_npu.npu_dtype_cast(weight, torch_npu.float8_e4m3fn),
            requires_grad=False)

        # Reshape weight_scale: [out, cols] → [out, cols/2, 2]
        ws = layer.weight_scale.data
        layer.weight_scale = torch.nn.Parameter(
            ws.reshape(ws.shape[0], -1, 2),
            requires_grad=False)

    def apply_weights(self, layer, x, bias=None):
        if x.dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)

        # 动态 MXFP8 量化激活
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float8_e4m3fn)

        # MXFP8 矩阵乘法
        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight.transpose(0, 1),
            layer.weight_scale.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=x.dtype,
            group_sizes=[1, 1, 32],
        )
        return output
```

#### 需要修改的文件

**修改 1**: `modelslim/schemes/__init__.py` — 注册新 Scheme

```python
# 新增导入
from .modelslim_mxfp8 import ModelSlimMXFP8
```

**修改 2**: `modelslim/modelslim.py` 第 179-193 行 — 添加 MXFP8 分发

```python
def _get_scheme_from_parts(self, layer_name):
    quant_type = self.quant_description.get(layer_name + ".weight", "")
    if quant_type == "W8A8_DYNAMIC" or quant_type == "W8A8":
        return ModelSlimW8A8Int8(
            quant_config=self.quant_description, prefix=layer_name)
    elif quant_type == "W4A4_DYNAMIC":
        return ModelSlimW4A4Int4(
            quant_config=self.quant_description, prefix=layer_name)
    elif quant_type == "W8A8_MXFP8":                        # ← 新增
        return ModelSlimMXFP8(                               # ← 新增
            quant_config=self.quant_description, prefix=layer_name)
    raise NotImplementedError("No modelslim compatible scheme was found.")
```

**修改 3**: `modelslim/modelslim.py` 顶部 import — 新增 `ModelSlimMXFP8`

```python
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimMXFP8,          # ← 新增
    ModelSlimW4A4Int4,
    ...
)
```

#### 使用方式

msmodelslim 生成 MXFP8 权重后，直接使用 SGLang 加载:

```bash
# msmodelslim 量化 (离线)
# 产出: quant_model_description.json + quant_model_weight_w8a8_mxfp8.safetensors

# SGLang 加载 (自动检测 quant_model_description.json)
python3 -m sglang.launch_server \
    --model-path /path/to/mxfp8-quantized-model \
    --quantization modelslim
```

SGLang 会自动检测模型目录中的 `quant_model_description.json`，读取每层的量化类型，对标记为 `"W8A8_MXFP8"` 的层使用 `ModelSlimMXFP8` Scheme。

### 6.5 适配工作量评估

| 任务 | 工时 | 说明 |
|------|------|------|
| 新增 `modelslim_mxfp8.py` | 1-2 天 | 核心实现，参考 MindIE-SD + 现有 W8A8 Scheme |
| 修改 `modelslim.py` 分发逻辑 | 0.5 天 | 仅需添加一个 elif 分支 |
| 权重参数对齐调试 | 1-2 天 | weight_scale 的 reshape/dtype 需要与 msmodelslim 输出对齐 |
| 单元测试 + 端到端验证 | 1-2 天 | 在 Ascend 设备上跑通完整推理 |
| **总计** | **3-6 天** | 比从零实现 MXFP8 简单很多 |

### 6.6 注意事项

1. **weight_scale 格式**: msmodelslim 导出的 weight_scale 是 `float8_e8m0fnu` 类型但以 `uint8` 存储在 safetensors 中，需要在 `process_weights_after_loading` 中正确 reshape 为 `[out, in/32, 2]`
2. **权重 dtype**: 权重以 `int8` 存储，运行时需 `npu_dtype_cast` 为 `float8_e4m3fn`
3. **bias dtype**: `npu_quant_matmul` 要求 bias 为 `float32`
4. **MoE 支持**: 如果需要 MXFP8 MoE，还需新增 `ModelSlimMXFP8MoE` scheme (参考 `modelslim_w8a8_int8_moe.py`)
5. **ChannelQuantScaleParameter**: weight_scale 的 `input_dim` 需要根据实际 reshape 逻辑确定是否需要自定义 Parameter 类型

---

## 七、MXFP8 完整适配方案设计 (含 Online + Offline 两条路径)

说明: 适配有两条路径:
- **路径 A (Offline/msmodelslim)**: 加载 msmodelslim 预量化的 MXFP8 权重 → 通过 modelslim Scheme 框架 (见第六章)
- **路径 B (Online)**: 从 FP16/BF16 权重在线量化为 MXFP8 → 通过 `--quantization mxfp8` (独立于 modelslim)

两条路径共享底层 NPU MXFP8 内核 (`npu_dynamic_mx_quant` + `npu_quant_matmul`)，但权重加载和注册方式不同。

### 7.1 需要修改的文件清单

#### 路径 A: msmodelslim MXFP8 权重支持 (推荐优先)

| 文件路径 | 描述 | 优先级 |
|----------|------|--------|
| `srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py` | **新增 MXFP8 Scheme** | P0 |
| `srt/layers/quantization/modelslim/schemes/__init__.py` | 注册新 Scheme | P0 |
| `srt/layers/quantization/modelslim/modelslim.py` | `_get_scheme_from_parts` 添加 MXFP8 分支 | P0 |
| `test/srt/ascend/test_ascend_mxfp8_quantization.py` | 测试用例 | P0 |

#### 路径 B: Online MXFP8 支持

| 文件路径 | 描述 | 优先级 |
|----------|------|--------|
| `srt/hardware_backend/npu/quantization/mxfp8_method_npu.py` | **NPU MXFP8 Linear 实现** | P0 |
| `srt/hardware_backend/npu/quantization/mxfp8_moe_method_npu.py` | NPU MXFP8 MoE 实现 (如需要) | P1 |
| `test/srt/ascend/test_ascend_mxfp8_quantization.py` | 测试用例 | P0 |

#### 需要修改的现有文件

| 文件路径 | 修改内容 | 优先级 |
|----------|----------|--------|
| `srt/layers/quantization/__init__.py` | 注册 NPU MXFP8 到量化方法表 | P0 |
| `srt/layers/quantization/fp8.py` | 在 `Fp8Config.get_quant_method()` 中增加 NPU 分支 | P0 |
| `srt/server_args.py` | 确认 `mxfp8` 在 `QUANTIZATION_CHOICES` 中 (已有) | P0 |
| `srt/hardware_backend/npu/quantization/linear_method_npu.py` | 可能需要添加 MXFP8 基类方法 | P1 |
| `srt/model_loader/loader.py` | 确认 NPU MXFP8 权重加载路径 | P1 |

#### Wan2.2 Diffusion 模型相关 (如需在 Diffusion 侧支持)

| 文件路径 | 修改内容 | 优先级 |
|----------|----------|--------|
| `multimodal_gen/configs/quantization.py` | 新增 MXFP8 量化配置 (参考 MindIE-SD) | P2 |
| `multimodal_gen/runtime/models/dits/wanvideo.py` | DiT 模型中注入量化层 | P2 |
| `multimodal_gen/configs/pipeline_configs/wan.py` | Pipeline 中集成量化选项 | P2 |

### 7.2 实现步骤 (路径 B: Online MXFP8)

#### Step 1: 实现 NPU MXFP8 Linear Method

参考 MindIE-SD 的 `W8A8MXFP8QuantLinear` 和 SGLang 现有的 `NPUW8A8Int8LinearMethod`:

```python
# srt/hardware_backend/npu/quantization/mxfp8_method_npu.py

class NPUMXFP8LinearMethod(LinearMethodBase):
    """Ascend NPU 上的 MXFP8 量化 Linear 方法"""

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, ...):
        # 权重: float8_e4m3fn
        layer.weight = Parameter(
            torch.empty(sum(output_partition_sizes), input_size_per_partition,
                       dtype=torch_npu.float8_e4m3fn))
        # 缩放因子: float8_e8m0fnu, block_size=32
        scale_size = (input_size_per_partition + 31) // 32
        layer.weight_scale = Parameter(
            torch.empty(sum(output_partition_sizes), scale_size,
                       dtype=torch_npu.float8_e8m0fnu))

    def apply(self, layer, x, bias=None):
        # 动态 MXFP8 量化激活
        qx, x_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float8_e4m3fn)

        # MXFP8 矩阵乘法
        output = torch_npu.npu_quant_matmul(
            qx, layer.weight, layer.weight_scale,
            pertoken_scale=x_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
            bias=bias)
        return output

    def process_weights_after_loading(self, layer):
        # NPU 格式转换 (如需要)
        layer.weight = Parameter(
            npu_format_cast(layer.weight.data), requires_grad=False)
```

#### Step 2: 注册到量化框架

在 `srt/layers/quantization/fp8.py` 的 `Fp8Config.get_quant_method()` 中:

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, LinearBase):
        if is_npu():
            if self.use_mxfp8:
                from ...hardware_backend.npu.quantization.mxfp8_method_npu import (
                    NPUMXFP8LinearMethod
                )
                return NPUMXFP8LinearMethod(self)
            else:
                # 普通 FP8 on NPU...
                pass
        return Fp8LinearMethod(self)  # CUDA 路径
```

#### Step 3: 权重加载适配

确保 `model_loader/loader.py` 中 NPU 路径能正确加载 MXFP8 权重:
- 支持 `float8_e4m3fn` 权重和 `float8_e8m0fnu` 缩放因子
- 支持 online quantization (从 FP16/BF16 权重动态量化)
- 支持 offline quantization (加载预量化的 MXFP8 checkpoint)

#### Step 4: 测试验证

```python
# test/srt/ascend/test_ascend_mxfp8_quantization.py

def test_mxfp8_linear():
    """测试 MXFP8 线性层在 NPU 上的正确性"""
    pass

def test_mxfp8_model_e2e():
    """端到端测试: 加载模型 + MXFP8 量化 + 推理"""
    pass

def test_mxfp8_accuracy():
    """精度测试: 对比 FP16 baseline"""
    pass
```

#### Step 5 (可选): Wan2.2 Diffusion 模型 MXFP8 集成

这需要在 `multimodal_gen` 侧做额外工作，类似 MindIE-SD 的 `quantize()` 函数:
- 为 DiT 模型的 Linear 层注入 MXFP8 量化
- 可参考 MindIE-SD 的 timestep-aware 量化策略

---

## 八、工作量评估

### 8.1 任务分解与工时估计

#### 阶段零: msmodelslim MXFP8 权重适配 (最快路径)

| 任务 | 详细描述 | 预估工时 | 难度 | 依赖 |
|------|----------|----------|------|------|
| 0.1 环境搭建 | Ascend NPU + CANN + torch_npu + SGLang | 1-2 天 | 低 | 需要硬件 |
| 0.2 验证 torch_npu MXFP8 API | 单独脚本测试 `npu_dynamic_mx_quant` | 0.5 天 | 低 | 0.1 |
| 0.3 实现 `ModelSlimMXFP8` Scheme | 新增 scheme 文件 + 修改分发逻辑 | 1-2 天 | 低-中 | 0.2 |
| 0.4 权重参数对齐 | weight_scale reshape/dtype 与 msmodelslim 对齐 | 1-2 天 | 中 | 0.3 |
| 0.5 端到端测试 | 用 msmodelslim 量化模型跑通推理 | 1-2 天 | 中 | 0.4 |
| **小计** | | **4-8 天** | | |

#### 阶段一: 基础 MXFP8 Linear 支持 (Online 量化, LLM Serving 侧)

| 任务 | 详细描述 | 预估工时 | 难度 | 依赖 |
|------|----------|----------|------|------|
| 1.1 环境搭建 | 搭建 Ascend NPU 开发环境、安装 torch_npu、验证基础 API | 1-2 天 | 低 | 需要 Ascend 硬件 |
| 1.2 验证 torch_npu MXFP8 API | 单独测试 `npu_dynamic_mx_quant` 和 `npu_quant_matmul` 的 MXFP8 模式 | 1 天 | 低 | 1.1 |
| 1.3 实现 `NPUMXFP8LinearMethod` | 参考 MindIE-SD 实现 MXFP8 Linear 方法 | 2-3 天 | 中 | 1.2 |
| 1.4 注册到量化框架 | 修改 `fp8.py`, `__init__.py` 等注册文件 | 0.5 天 | 低 | 1.3 |
| 1.5 权重加载适配 | 支持 MXFP8 online quantization 权重加载 | 1-2 天 | 中 | 1.3 |
| 1.6 单元测试 | Linear 层正确性、精度回归测试 | 1-2 天 | 低 | 1.3 |
| 1.7 端到端测试 | 选择 LLM 模型 (如 Llama-8B) 进行完整推理验证 | 1-2 天 | 中 | 1.5 |
| 1.8 性能调优 | profiling、NZ 格式优化、吞吐量测试 | 2-3 天 | 中高 | 1.7 |
| **小计** | | **9.5-15.5 天** | | |

#### 阶段二: MXFP8 MoE 支持

| 任务 | 详细描述 | 预估工时 | 难度 | 依赖 |
|------|----------|----------|------|------|
| 2.1 分析 MoE 量化架构 | 研究现有 `fused_moe_method_npu.py` (W4A4) | 1 天 | 中 | -- |
| 2.2 实现 MXFP8 MoE 方法 | 在 NPU 上实现 MXFP8 的 FusedMoE | 3-5 天 | 高 | 2.1, 阶段一 |
| 2.3 DeepSeek-V2 等 MoE 模型测试 | 端到端 MoE 模型验证 | 2-3 天 | 中 | 2.2 |
| **小计** | | **6-9 天** | | |

#### 阶段三: Wan2.2 Diffusion 模型 MXFP8 支持

| 任务 | 详细描述 | 预估工时 | 难度 | 依赖 |
|------|----------|----------|------|------|
| 3.1 分析 SGLang Diffusion 架构 | 理解 `multimodal_gen` 子系统、Wan2.2 Pipeline | 1-2 天 | 中 | -- |
| 3.2 设计 Diffusion MXFP8 量化方案 | 参考 MindIE-SD 的层替换模式，设计 SGLang Diffusion 量化接口 | 1-2 天 | 中 | 3.1 |
| 3.3 实现 DiT Linear 层 MXFP8 | 为 Wan2.2 DiT 的 Linear 层注入 MXFP8 | 2-3 天 | 中 | 3.2, 阶段一 |
| 3.4 (可选) Timestep-aware 量化 | 参考 MindIE-SD 的 `TimestepPolicyConfig`，不同去噪步使用不同策略 | 2-3 天 | 高 | 3.3 |
| 3.5 (可选) FP8 Attention 量化 | 参考 MindIE-SD 的 `FP8RotateQuantFA` | 3-5 天 | 高 | 3.3 |
| 3.6 端到端 Wan2.2 测试 | 生成视频质量验证、性能 benchmark | 2-3 天 | 中 | 3.3 |
| **小计** | | **11-18 天** (含可选项) | | |

#### 阶段四: 代码提交与上游合并

| 任务 | 详细描述 | 预估工时 | 难度 | 依赖 |
|------|----------|----------|------|------|
| 4.1 代码规范对齐 | 符合 SGLang 代码风格、lint、类型标注 | 1 天 | 低 | -- |
| 4.2 文档编写 | 更新 docs、README、使用说明 | 1 天 | 低 | -- |
| 4.3 PR 创建与 Review | 创建 PR、回应 Review 意见、CI 调试 | 3-5 天 | 中 | 全部 |
| **小计** | | **5-7 天** | | |

### 8.2 总工作量汇总

| 阶段 | 范围 | 工时 | 累计 |
|------|------|------|------|
| **阶段零** ⭐ | **msmodelslim MXFP8 权重适配** | **4-8 天** | **4-8 天** |
| **阶段一** | Online MXFP8 Linear (LLM) | 9.5-15.5 天 | 13.5-23.5 天 |
| **阶段二** | MXFP8 MoE | 6-9 天 | 19.5-32.5 天 |
| **阶段三** | Wan2.2 Diffusion MXFP8 | 11-18 天 | 30.5-50.5 天 |
| **阶段四** | 提交合并 | 5-7 天 | 35.5-57.5 天 |
| **总计** | | **约 1.5-3 个月** (1人全职) | |

> ⭐ **推荐优先做阶段零**: 如果团队已有 msmodelslim 量化好的 MXFP8 权重，走 modelslim Scheme 路径最快，约 1-2 周即可跑通。

### 8.3 最小可行方案 (MVP)

**MVP 方案: 阶段零 (msmodelslim MXFP8 权重适配)**

- **工时**: 约 **1-2 周**
- **产出**: 用 msmodelslim 量化的 MXFP8 模型可在 SGLang + Ascend NPU 上推理
- **使用方式**: `--quantization modelslim` (自动检测 `quant_model_description.json`)
- **风险**: 很低，框架已有、API 已有、仅需新增一个 Scheme 类
- **优势**: 与团队现有 msmodelslim 工具链完全兼容

如果还需要 Online MXFP8 (无需预量化权重)，再叠加**阶段一**:
- **额外工时**: 约 2-3 周
- **产出**: `--quantization mxfp8` 从 FP16 模型在线量化推理

### 8.4 风险与依赖

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **torch_npu API 版本兼容** | `npu_dynamic_mx_quant` 可能仅在特定 CANN 版本支持 | 提前确认 CANN 版本要求 (≥8.0.RC3) |
| **Ascend 硬件获取** | 开发和测试需要 Atlas 800I A2/A3 | 确认团队有可用设备 |
| **上游 Review 周期** | SGLang 社区 PR review 可能需要较长时间 | 提前与 @ping1jing2 / @OrangeRedeng 沟通 |
| **MXFP8 精度问题** | block-size-32 在某些模型可能精度不足 | 准备精度 benchmark 对比数据 |
| **Wan2.2 Diffusion NPU 支持** | `multimodal_gen` 可能尚未适配 NPU | 可能需要先解决 Diffusion 的 NPU 基础支持 |
| **Issue 中 YChange01 已认领** | MXFP8/MXFP4 任务已被 YChange01 认领 | 需要与其协调分工或确认是否需要协作 |

---

## 九、建议的推进策略

### 9.1 短期 (第1-2周) — msmodelslim MXFP8 适配

1. **与社区对齐**: 在 Issue #14424 中评论，说明我们要做 MXFP8 NPU 适配，与 YChange01 协调
2. **环境搭建**: 准备 Ascend NPU 开发环境 + CANN + torch_npu
3. **API 验证**: 用独立脚本验证 `torch_npu.npu_dynamic_mx_quant` 可用性
4. **实现 ModelSlimMXFP8 Scheme**: 新增 scheme + 修改分发逻辑
5. **用 msmodelslim MXFP8 权重跑通推理**: 端到端验证

### 9.2 中期 (第3-5周) — Online MXFP8 + MoE

6. **实现 Online MXFP8 Linear**: 完成 `NPUMXFP8LinearMethod` 核心代码
7. **集成测试**: 在 LLM 模型上跑通 Online MXFP8 推理
8. **MoE 支持**: 如有需要，扩展到 MoE 模型
9. **提交 PR**: 阶段零 + 阶段一代码提交

### 9.3 长期 (第6-10周) — Diffusion + 优化

10. **Diffusion 支持**: Wan2.2 MXFP8 量化集成
11. **性能优化**: 吞吐量、延迟对比 benchmark
12. **社区 Review 与合并**

---

## 十、参考资源

### 代码仓库
- SGLang: `./sglang/` (本地已 clone)
- MindIE-SD: `./MindIE-SD/` (本地已 clone)

### 关键文档
- [SGLang 官方文档](https://docs.sglang.io/index.html)
- [SGLang 量化文档](https://docs.sglang.io/advanced_features/quantization.html)
- [Issue #14424 - Ascend NPU 量化路线图](https://github.com/sgl-project/sglang/issues/14424)
- [Issue #17093 - MXFP8 GPU 侧支持](https://github.com/sgl-project/sglang/issues/17093)
- [PR #17449 - MXFP8 CUDA 实现](https://github.com/sgl-project/sglang/pull/17449)
- [Issue #18258 - ModelOpt MXFP8 Loader](https://github.com/sgl-project/sglang/issues/18258)
- [msmodelslim - Ascend 模型压缩工具](https://gitcode.com/Ascend/msmodelslim)

### 关键代码路径
- SGLang 量化核心: `sglang/python/sglang/srt/layers/quantization/`
- SGLang NPU 后端: `sglang/python/sglang/srt/hardware_backend/npu/`
- SGLang FP8 实现: `sglang/python/sglang/srt/layers/quantization/fp8.py`
- SGLang Wan2.2: `sglang/python/sglang/multimodal_gen/configs/pipeline_configs/wan.py`
- MindIE-SD MXFP8: `MindIE-SD/mindiesd/quantization/layer.py` → `W8A8MXFP8QuantLinear`
- MindIE-SD 量化入口: `MindIE-SD/mindiesd/quantization/quantize.py`
- SGLang ModelSlim 模块: `sglang/python/sglang/srt/layers/quantization/modelslim/`
- SGLang ModelSlim Scheme: `sglang/python/sglang/srt/layers/quantization/modelslim/schemes/`
