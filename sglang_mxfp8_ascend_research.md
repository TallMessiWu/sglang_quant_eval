![English](https://img.shields.io/badge/Lang-English-blue) [![中文](https://img.shields.io/badge/语言-中文-red)](./sglang_mxfp8_ascend_research_zh.md)
# SGLang MXFP8 Quantization Adaptation Research on Huawei Ascend NPU

> Date: 2026-03-16
> Related Issue: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424)
> Goal: Adapt MXFP8 quantization for SGLang on Huawei Ascend NPU, supporting Wan2.2 and other models

---

## 1. SGLang Overview

### 1.1 What is SGLang

SGLang is a high-performance large model inference serving framework developed by the LMSYS team, deployed on over 400,000 GPUs worldwide. Core features include:

- **RadixAttention** prefix caching
- Zero-overhead CPU scheduler
- Prefill-Decode disaggregation
- Speculative decoding, continuous batching, paged attention
- Tensor/pipeline/expert/data parallelism
- Structured output generation
- OpenAI-compatible API

### 1.2 Basic Usage

```bash
# Installation
pip install sglang

# Start server
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 30000 --host 0.0.0.0

# Python API
import sglang as sgl
llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
outputs = llm.generate(["Hello!"], {"max_new_tokens": 100})
```

### 1.3 Hardware Support

| Hardware | Status |
|----------|--------|
| NVIDIA GPU (CUDA) | Full support |
| AMD GPU (ROCm) | Supports MI300/MI350 |
| Intel Xeon CPU | Supported |
| Google TPU | Supported |
| **Huawei Ascend NPU** | **Basic support available, quantization continuously improving** |

### 1.4 Code Architecture

SGLang Serving Runtime (`sglang/python/sglang/srt/`) main modules:

```
srt/
├── entrypoints/        # HTTP/gRPC service entry (FastAPI, OpenAI-compatible)
├── model_executor/     # Model execution engine (CUDA/CPU/NPU GraphRunner)
├── model_loader/       # Weight loading (safetensors, GGUF, pt, etc.)
├── models/             # 100+ model implementations (Llama, Qwen, DeepSeek, GLM, etc.)
├── layers/
│   ├── quantization/   # ★ Quantization core directory (20+ quantization methods)
│   ├── attention/      # Attention layers
│   └── moe/            # MoE layers
├── hardware_backend/
│   └── npu/            # ★ Ascend NPU backend
├── managers/           # Memory and scheduling management
├── distributed/        # TP, EP, DP parallelism
└── speculative/        # Speculative decoding
```

Video/image generation independent subsystem:
```
multimodal_gen/         # ★ Wan2.2 and other Diffusion models are here
├── configs/            # Model configs, quantization configs, Pipeline configs
├── runtime/
│   ├── models/         # DiT, VAE model implementations
│   └── pipelines/      # Inference Pipelines (Wan, HunyuanVideo, etc.)
└── registry.py         # Model registry
```

---

## 2. SGLang Existing Quantization Support

### 2.1 INT8 Quantization (Supported)

SGLang fully supports INT8 quantization in three modes:

#### a) W8A8 INT8 (per-channel weights, per-token activations)
```bash
python3 -m sglang.launch_server \
    --model-path <int8-quantized-model> \
    --quantization w8a8_int8
```
- Implementation: `srt/layers/quantization/w8a8_int8.py`
- Uses `sgl_kernel.int8_scaled_mm` (CUDA)
- Supports NVIDIA, AMD, Intel

#### b) Block-wise INT8
```bash
python3 -m sglang.launch_server \
    --model-path <blockwise-int8-model> \
    --quantization blockwise_int8
```
- Implementation: `srt/layers/quantization/blockwise_int8.py`
- Configurable block size

#### c) TorchAO INT8
```bash
# Weight-only INT8
python3 -m sglang.launch_server \
    --model-path <model> --torchao-config int8wo

# Dynamic INT8
python3 -m sglang.launch_server \
    --model-path <model> --torchao-config int8dq --disable-cuda-graph
```

### 2.2 FP8/MXFP8 Quantization (Supported on CUDA)

| Method | CLI Argument | Description |
|--------|-------------|-------------|
| Online FP8 | `--quantization fp8` | Dynamic scaling at runtime, no pre-quantization needed |
| **Online MXFP8** | `--quantization mxfp8` | **Microscaling FP8, block-size-32** |
| W8A8 FP8 | `--quantization w8a8_fp8` | Pre-quantized weights + dynamic activations |
| ModelOpt FP8 | `--quantization modelopt_fp8` | NVIDIA ModelOpt pre-quantization |
| FBGEMM FP8 | `--quantization fbgemm_fp8` | FBGEMM backend |
| MXFP4 | `--quantization mxfp4` | Microscaling FP4 |

**Key point**: MXFP8 on CUDA has been merged via PR #17449, but **has not yet been adapted for Ascend NPU**. This is precisely the work requested by Issue #14424.

### 2.3 Complete Quantization Method Registry

File: `srt/layers/quantization/__init__.py`

```python
# BASE_QUANTIZATION_METHODS (always registered)
BASE_QUANTIZATION_METHODS = {
    "fp8": Fp8Config,
    "mxfp8": Fp8Config,          # use_mxfp8=True
    "blockwise_int8": BlockInt8Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "modelopt_fp8": ModelOptFp8Config,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    "modelslim": ModelSlimConfig,  # NPU-specific
    # ... 20+ methods
}

# mxfp4 is conditionally registered only on CUDA or MXFP-supported HIP platforms
if is_cuda() or (_is_mxfp_supported and is_hip()):
    BASE_QUANTIZATION_METHODS["mxfp4"] = Mxfp4Config
```

---

## 3. Ascend NPU Backend Current State

### 3.1 Existing NPU Code Structure

```
srt/hardware_backend/npu/
├── allocator_npu.py              # NPU memory allocator
├── cmo.py                        # CMO related
├── memory_pool_npu.py            # NPU memory pool
├── utils.py
├── attention/
│   ├── ascend_backend.py         # Ascend attention backend
│   ├── ascend_torch_native_backend.py
│   └── mla_preprocess.py         # MLA preprocessing
├── quantization/
│   ├── linear_method_npu.py      # W8A8/W4A4 Linear NPU
│   └── fused_moe_method_npu.py   # W4A4/W4A8/W4A16/W8A8 MoE NPU
├── graph_runner/
│   ├── npu_graph_runner.py       # NPU graph executor
│   ├── eagle_draft_npu_graph_runner.py
│   ├── eagle_draft_extend_npu_graph_runner.py
│   └── vit_npu_graph_runner.py
├── modules/
│   ├── deepseek_v2_attention_mla_npu.py
│   └── qwen_vl_processor.py
└── moe/topk.py
```

### 3.2 NPU-Supported Quantization Methods

| Method | Status | Implementation Location |
|--------|--------|------------------------|
| W8A8 INT8 (static, Linear) | ✅ Done | `linear_method_npu.py` → `NPUW8A8Int8LinearMethod`; used via ModelSlim `ModelSlimW8A8Int8` |
| W8A8 INT8 (dynamic, Linear) | ✅ Done | `linear_method_npu.py` → `NPUW8A8Int8DynamicLinearMethod`; used via ModelSlim `ModelSlimW8A8Int8` |
| W4A4 INT4 (dynamic, Linear) | ✅ Done | `linear_method_npu.py` → `NPU_W4A4DynamicLinearMethod`; used via ModelSlim `ModelSlimW4A4Int4` |
| W8A8 INT8 (dynamic, MoE) | ✅ Done | `fused_moe_method_npu.py` → `NPUW8A8Int8DynamicMoEMethod`; used via ModelSlim `ModelSlimW8A8Int8MoE` |
| W4A4 INT4 (dynamic, MoE) | ✅ Done | `fused_moe_method_npu.py` → `NPUW4A4Int4DynamicMoEMethod`; used via ModelSlim `ModelSlimW4A4Int4MoE` |
| W4A8 INT8 (dynamic, MoE) | ✅ Done | `fused_moe_method_npu.py` → `NPUW4A8Int8DynamicMoEMethod`; used via ModelSlim `ModelSlimW4A8Int8MoE` |
| W4A16 INT4 (dynamic, MoE) | ✅ Done | `fused_moe_method_npu.py` → `NPUW4A16Int4DynamicMoEMethod`; used via `compressed_tensors` |
| **MXFP8** | ❌ **Not implemented** | **Needs to be added** |
| **MXFP4** | ❌ Not implemented | Needs to be added |
| KV Cache quantization | 🔄 In progress | -- |

### 3.3 NPU INT8 Implementation Example (Reference Pattern)

Core flow of `NPUW8A8Int8LinearMethod`:
```python
# 1. Create weight parameters
def create_weights(self, layer, ...):
    layer.weight = Parameter(torch.empty(..., dtype=torch.int8))
    layer.weight_scale = Parameter(torch.empty(..., dtype=torch.float32))
    layer.input_scale = Parameter(torch.empty(..., dtype=torch.float32))

# 2. Post-loading processing (NPU format conversion)
def process_weights_after_loading(self, layer):
    layer.weight = npu_format_cast(layer.weight)

# 3. Inference
def apply(self, layer, x, bias=None):
    qx = torch.ops.npu.npu_quantize(x, layer.input_scale, ...)
    return torch.ops.npu.npu_quant_matmul(qx, layer.weight, layer.deq_scale, ...)
```

---

## 4. MindIE-SD Reference Analysis

### 4.1 MindIE-SD Overview

MindIE-SD is Huawei's Stable Diffusion inference acceleration engine, designed specifically for Ascend NPU. It has already adapted multiple quantization methods, including **MXFP8**.

### 4.2 Quantization Algorithms Supported by MindIE-SD

| Category | Methods |
|----------|---------|
| Weight-only | W8A16, W4A16, W4A16_AWQ, W8A16_GPTQ, W4A16_GPTQ |
| W8A8 INT8 | Static/Dynamic/Per-channel/Per-tensor/Per-token/Timestep-aware |
| **MXFP8** | **W8A8_MXFP8** ✅ |
| FP8 Attention | FP8_DYNAMIC (with rotation matrix) |

### 4.3 MindIE-SD MXFP8 Implementation (Core Reference)

File: `MindIE-SD/mindiesd/quantization/layer.py` → `W8A8MXFP8QuantLinear` (inherits from `W8A8QuantBaseLinear`)

```python
class W8A8MXFP8QuantLinear(W8A8QuantBaseLinear):
    # forward() inherited from W8A8QuantBaseLinear, flattens 3D+ inputs first
    # Core computation logic is in quant_matmul():
    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        # 1. Optional mul_scale pre-scaling (smooth quant scenario)
        if self.mul_scale is not None:
            x1, input_scale = torch_npu.npu_dynamic_mx_quant(
                x * self.mul_scale, dst_type=torch_npu.float8_e4m3fn)
        else:
            x1, input_scale = torch_npu.npu_dynamic_mx_quant(
                x, dst_type=torch_npu.float8_e4m3fn)

        if self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32)

        # 2. Weight dtype conversion + transpose
        x2 = self.weight
        if x2.dtype != torch.float8_e4m3fn:
            x2 = torch_npu.npu_dtype_cast(x2, torch_npu.float8_e4m3fn)
        x2 = x2.transpose(0, 1)

        # 3. MXFP8 matrix multiplication
        output = torch_npu.npu_quant_matmul(
            x1,
            x2,
            self.weight_scale.transpose(0, 1),  # weight_scale also needs transpose
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=self.bias,                      # must be float32
            output_dtype=self.dtype,             # output restored to original precision
            group_sizes=[1, 1, 32],              # MXFP8 block size = 32
        )
        return output
```

### 4.4 Key Ascend NPU APIs (torch_npu)

| API | Purpose |
|-----|---------|
| `torch_npu.npu_dynamic_mx_quant(x, dst_type)` | **MXFP8 dynamic quantization** |
| `torch_npu.npu_quant_matmul(...)` | Quantized matrix multiplication (INT8/FP8) |
| `torch_npu.npu_quantize(x, scale, offset)` | Static INT8 quantization |
| `torch_npu.npu_dynamic_quant(x)` | Dynamic per-token INT8 quantization |
| `torch_npu.npu_format_cast(tensor, 29)` | NZ format conversion |
| `torch_npu.npu_weight_quant_batchmatmul(...)` | Weight-only quantized batch matmul |
| `torch_npu.float8_e4m3fn` | FP8 E4M3 data type |
| `torch_npu.float8_e8m0fnu` | FP8 E8M0 scale factor type |

### 4.5 SGLang vs MindIE-SD Similarity Analysis

| Dimension | SGLang | MindIE-SD | Similarity |
|-----------|--------|-----------|------------|
| Quantization framework | Registry + Config + Method pattern | Model traversal + layer replacement | Medium |
| Weight loading | Separated from inference kernel (refactoring) | JSON description file + safetensors | Design philosophy similar |
| NPU INT8 | `npu_quantize` + `npu_quant_matmul` | Same APIs | **Highly similar** |
| Linear quantization | `QuantizeMethodBase.apply()` | `QuantLinear.forward()` | Same pattern |
| MoE quantization | `FusedMoEMethodBase` | None (Diffusion has no MoE) | N/A |
| MXFP8 (CUDA) | `mxfp8_group_quantize` + Triton/FlashInfer | N/A | -- |
| **MXFP8 (NPU)** | **❌ Not implemented** | **✅ `npu_dynamic_mx_quant`** | **Directly reusable** |

**Core conclusion**: MindIE-SD's MXFP8 NPU implementation (`W8A8MXFP8QuantLinear`) can be directly used as a reference template for SGLang NPU MXFP8. The underlying APIs (`torch_npu.npu_dynamic_mx_quant`, `torch_npu.npu_quant_matmul`) are identical.

---

## 5. Wan2.2 Model Support Status in SGLang

### 5.1 Wan2.2 Model Matrix

SGLang already supports Wan2.2 through the `multimodal_gen` subsystem:

| Model | HuggingFace ID | Resolution |
|-------|----------------|------------|
| Wan2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 720p |
| Wan2.2 T2V A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 480p, 720p |
| Wan2.2 I2V A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 480p, 720p |

### 5.2 Wan2.2 Architecture Characteristics

- **DiT (Diffusion Transformer)**: Transformer-based diffusion model with multiple Transformer blocks
- **Pipeline**: TextEncode → Denoise (multi-step denoising) → VAE Decode
- Supports TP (tensor parallelism) and SP (sequence parallelism)

### 5.3 Existing Diffusion Quantization

SGLang Diffusion currently supports the following quantization methods:

| Method | Configuration Location | Platform |
|--------|----------------------|----------|
| **SVDQuant/Nunchaku** (W4A4) | `multimodal_gen/configs/quantization.py` → `NunchakuSVDQuantArgs` | NVIDIA GPU |
| **FP8** | `multimodal_gen/runtime/layers/quantization/fp8.py` | NVIDIA GPU |
| **ModelSlim** | `multimodal_gen/runtime/layers/quantization/modelslim.py` | Ascend NPU |

- MXFP8 is not yet supported
- Nunchaku is NVIDIA GPU only

### 5.4 Usage

```bash
# Start Wan2.2 server
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers --num-gpus 4

# Python API
from sglang.multimodal_gen import DiffGenerator
gen = DiffGenerator.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
```

---

## 6. msmodelslim Weight and SGLang Compatibility Analysis

### 6.1 msmodelslim Overview

[msmodelslim](https://gitcode.com/Ascend/msmodelslim) is Huawei Ascend's model compression toolkit, supporting PTQ (post-training quantization), QAT (quantization-aware training), low-rank decomposition, knowledge distillation, sparse training, etc. Can export to PyTorch, MindSpore, ONNX, AutoAWQ, AutoGPTQ formats.

### 6.2 msmodelslim Output Format

msmodelslim quantization outputs **two files**:

1. **JSON description file**: `quant_model_description.json`
   - Flat dictionary, keys are layer weight names (e.g., `model.layers.0.self_attn.q_proj.weight`), values are quantization type strings (e.g., `"W8A8_DYNAMIC"`, `"W8A8_MXFP8"`, `"FLOAT"`)
   - Multi-GPU scenario: `quant_model_description_{rank}.json`

2. **Safetensors weight file**: `quant_model_weight_{algo}.safetensors`
   - Contains quantized weight, weight_scale, weight_offset, input_scale, input_offset, deq_scale, quant_bias, etc.
   - Multi-GPU scenario: `quant_model_weight_{algo}_{rank}.safetensors`

### 6.3 msmodelslim Formats Already Supported by SGLang

**Conclusion: SGLang already has a modelslim integration module, but does not support MXFP8.**

SGLang's modelslim support is located at `sglang/python/sglang/srt/layers/quantization/modelslim/`, using a Scheme pattern:

| msmodelslim Quantization Type | SGLang Scheme Class | Status |
|------------------------------|---------------------|--------|
| `W8A8` (static) | `ModelSlimW8A8Int8` → `NPUW8A8Int8LinearMethod` | ✅ Supported |
| `W8A8_DYNAMIC` | `ModelSlimW8A8Int8` → `NPUW8A8Int8DynamicLinearMethod` | ✅ Supported |
| `W4A4_DYNAMIC` | `ModelSlimW4A4Int4` | ✅ Supported |
| `W4A4_DYNAMIC` (MoE) | `ModelSlimW4A4Int4MoE` | ✅ Supported |
| `W4A8_DYNAMIC` (MoE) | `ModelSlimW4A8Int8MoE` | ✅ Supported |
| `W8A8_DYNAMIC` (MoE) | `ModelSlimW8A8Int8MoE` | ✅ Supported |
| **`W8A8_MXFP8`** | **None** | **❌ Not supported** |
| `W8A16` / `W4A16` | None | ❌ Not supported |

**Root cause of missing support**: The `_get_scheme_from_parts()` method at lines 179-193 of `modelslim.py` only recognizes `"W8A8_DYNAMIC"`, `"W8A8"`, `"W4A4_DYNAMIC"` three types; encountering `"W8A8_MXFP8"` will raise `NotImplementedError`.

```python
# modelslim.py lines 179-193 (current code)
def _get_scheme_from_parts(self, layer_name):
    quant_type = self.quant_description.get(layer_name + ".weight", "")
    if quant_type == "W8A8_DYNAMIC" or quant_type == "W8A8":
        return ModelSlimW8A8Int8(...)
    elif quant_type == "W4A4_DYNAMIC":
        return ModelSlimW4A4Int4(...)
    raise NotImplementedError("No modelslim compatible scheme was found.")
    # ↑ W8A8_MXFP8 will hit this
```

### 6.4 Adaptation Plan: Making SGLang Support msmodelslim MXFP8 Weights

The adaptation effort is small, as SGLang's modelslim framework is already mature — only a new MXFP8 Scheme needs to be added under the existing Scheme pattern.

#### Files to Add

**File 1**: `sglang/python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py`

Reference the pattern of `modelslim_w8a8_int8.py`, combined with MindIE-SD's `W8A8MXFP8QuantLinear`:

```python
import torch
import torch_npu

from sglang.srt.layers.parameter import ModelWeightParameter, ChannelQuantScaleParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme


class ModelSlimMXFP8(ModelSlimLinearScheme):
    """msmodelslim W8A8_MXFP8 quantization Scheme"""

    def __init__(self, quant_config, prefix):
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # Weight: stored as int8, cast to float8_e4m3fn at runtime
        weight = ModelWeightParameter(
            data=torch.empty(
                (output_size_per_partition, input_size_per_partition),
                dtype=torch.int8),
            input_dim=1, output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Weight scale: block-size-32, one scale per 32 elements
        # msmodelslim export format: [out_features, in_features/32 * 2]
        # after reshape: [out_features, in_features/32, 2]
        scale_cols = (input_size_per_partition + 31) // 32 * 2
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, scale_cols),
                dtype=torch.uint8),  # float8_e8m0fnu stored as uint8
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer):
        # Cast weight to float8_e4m3fn
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

        # Dynamic MXFP8 activation quantization
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float8_e4m3fn)

        # MXFP8 matrix multiplication
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

#### Files to Modify

**Modification 1**: `modelslim/schemes/__init__.py` — Register new Scheme

```python
# Add import
from .modelslim_mxfp8 import ModelSlimMXFP8
```

**Modification 2**: `modelslim/modelslim.py` lines 179-193 — Add MXFP8 dispatch

```python
def _get_scheme_from_parts(self, layer_name):
    quant_type = self.quant_description.get(layer_name + ".weight", "")
    if quant_type == "W8A8_DYNAMIC" or quant_type == "W8A8":
        return ModelSlimW8A8Int8(
            quant_config=self.quant_description, prefix=layer_name)
    elif quant_type == "W4A4_DYNAMIC":
        return ModelSlimW4A4Int4(
            quant_config=self.quant_description, prefix=layer_name)
    elif quant_type == "W8A8_MXFP8":                        # ← new
        return ModelSlimMXFP8(                               # ← new
            quant_config=self.quant_description, prefix=layer_name)
    raise NotImplementedError("No modelslim compatible scheme was found.")
```

**Modification 3**: Top-level imports in `modelslim/modelslim.py` — Add `ModelSlimMXFP8`

```python
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimMXFP8,          # ← new
    ModelSlimW4A4Int4,
    ...
)
```

#### Usage

After msmodelslim generates MXFP8 weights, load directly with SGLang:

```bash
# msmodelslim quantization (offline)
# Output: quant_model_description.json + quant_model_weight_w8a8_mxfp8.safetensors

# SGLang loading (auto-detects quant_model_description.json)
python3 -m sglang.launch_server \
    --model-path /path/to/mxfp8-quantized-model \
    --quantization modelslim
```

SGLang will automatically detect `quant_model_description.json` in the model directory, read the quantization type for each layer, and use the `ModelSlimMXFP8` Scheme for layers marked as `"W8A8_MXFP8"`.

### 6.5 Adaptation Effort Estimate

| Task | Effort | Notes |
|------|--------|-------|
| Add `modelslim_mxfp8.py` | 1-2 days | Core implementation, reference MindIE-SD + existing W8A8 Scheme |
| Modify `modelslim.py` dispatch logic | 0.5 days | Only need to add one elif branch |
| Weight parameter alignment debugging | 1-2 days | weight_scale reshape/dtype must align with msmodelslim output |
| Unit tests + end-to-end validation | 1-2 days | Run complete inference on Ascend device |
| **Total** | **3-6 days** | Much simpler than implementing MXFP8 from scratch |

### 6.6 Notes

1. **weight_scale format**: msmodelslim exports weight_scale as `float8_e8m0fnu` type but stored as `uint8` in safetensors; needs correct reshape to `[out, in/32, 2]` in `process_weights_after_loading`
2. **Weight dtype**: Weights stored as `int8`, need `npu_dtype_cast` to `float8_e4m3fn` at runtime
3. **Bias dtype**: `npu_quant_matmul` requires bias to be `float32`
4. **MoE support**: If MXFP8 MoE is needed, also need to add `ModelSlimMXFP8MoE` scheme (reference `modelslim_w8a8_int8_moe.py`)
5. **ChannelQuantScaleParameter**: The `input_dim` of weight_scale needs to be determined based on actual reshape logic, and may require a custom Parameter type

---

## 7. Complete MXFP8 Adaptation Plan Design (Online + Offline Paths)

Note: There are two adaptation paths:
- **Path A (Offline/msmodelslim)**: Load msmodelslim pre-quantized MXFP8 weights → via modelslim Scheme framework (see Chapter 6)
- **Path B (Online)**: Quantize FP16/BF16 weights to MXFP8 online → via `--quantization mxfp8` (independent of modelslim)

Both paths share the underlying NPU MXFP8 kernels (`npu_dynamic_mx_quant` + `npu_quant_matmul`), but differ in weight loading and registration approach.

### 7.1 Files to Modify

#### Path A: msmodelslim MXFP8 Weight Support (Recommended Priority)

| File Path | Description | Priority |
|-----------|-------------|----------|
| `srt/layers/quantization/modelslim/schemes/modelslim_mxfp8.py` | **Add MXFP8 Scheme** | P0 |
| `srt/layers/quantization/modelslim/schemes/__init__.py` | Register new Scheme | P0 |
| `srt/layers/quantization/modelslim/modelslim.py` | Add MXFP8 branch to `_get_scheme_from_parts` | P0 |
| `test/srt/ascend/test_ascend_mxfp8_quantization.py` | Test cases | P0 |

#### Path B: Online MXFP8 Support

| File Path | Description | Priority |
|-----------|-------------|----------|
| `srt/hardware_backend/npu/quantization/mxfp8_method_npu.py` | **NPU MXFP8 Linear implementation** | P0 |
| `srt/hardware_backend/npu/quantization/mxfp8_moe_method_npu.py` | NPU MXFP8 MoE implementation (if needed) | P1 |
| `test/srt/ascend/test_ascend_mxfp8_quantization.py` | Test cases | P0 |

#### Existing Files to Modify

| File Path | Change | Priority |
|-----------|--------|----------|
| `srt/layers/quantization/__init__.py` | Register NPU MXFP8 in quantization method table | P0 |
| `srt/layers/quantization/fp8.py` | Add NPU branch in `Fp8Config.get_quant_method()` | P0 |
| `srt/server_args.py` | Confirm `mxfp8` is in `QUANTIZATION_CHOICES` (already present) | P0 |
| `srt/hardware_backend/npu/quantization/linear_method_npu.py` | May need to add MXFP8 base class methods | P1 |
| `srt/model_loader/loader.py` | Confirm NPU MXFP8 weight loading path | P1 |

#### Wan2.2 Diffusion Model Related (if Diffusion-side support is needed)

| File Path | Change | Priority |
|-----------|--------|----------|
| `multimodal_gen/configs/quantization.py` | Add MXFP8 quantization config (reference MindIE-SD) | P2 |
| `multimodal_gen/runtime/models/dits/wanvideo.py` | Inject quantization layers into DiT model | P2 |
| `multimodal_gen/configs/pipeline_configs/wan.py` | Integrate quantization options into Pipeline | P2 |

### 7.2 Implementation Steps (Path B: Online MXFP8)

#### Step 1: Implement NPU MXFP8 Linear Method

Reference MindIE-SD's `W8A8MXFP8QuantLinear` and SGLang's existing `NPUW8A8Int8LinearMethod`:

```python
# srt/hardware_backend/npu/quantization/mxfp8_method_npu.py

class NPUMXFP8LinearMethod(LinearMethodBase):
    """MXFP8 quantized Linear method on Ascend NPU"""

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, ...):
        # Weight: float8_e4m3fn
        layer.weight = Parameter(
            torch.empty(sum(output_partition_sizes), input_size_per_partition,
                       dtype=torch_npu.float8_e4m3fn))
        # Scale factor: float8_e8m0fnu, block_size=32
        scale_size = (input_size_per_partition + 31) // 32
        layer.weight_scale = Parameter(
            torch.empty(sum(output_partition_sizes), scale_size,
                       dtype=torch_npu.float8_e8m0fnu))

    def apply(self, layer, x, bias=None):
        # Dynamic MXFP8 activation quantization
        qx, x_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float8_e4m3fn)

        # MXFP8 matrix multiplication
        output = torch_npu.npu_quant_matmul(
            qx, layer.weight, layer.weight_scale,
            pertoken_scale=x_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
            bias=bias)
        return output

    def process_weights_after_loading(self, layer):
        # NPU format conversion (if needed)
        layer.weight = Parameter(
            npu_format_cast(layer.weight.data), requires_grad=False)
```

#### Step 2: Register to Quantization Framework

In `Fp8Config.get_quant_method()` in `srt/layers/quantization/fp8.py`:

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
                # Regular FP8 on NPU...
                pass
        return Fp8LinearMethod(self)  # CUDA path
```

#### Step 3: Weight Loading Adaptation

Ensure `model_loader/loader.py` in the NPU path can correctly load MXFP8 weights:
- Support `float8_e4m3fn` weights and `float8_e8m0fnu` scale factors
- Support online quantization (dynamic quantization from FP16/BF16 weights)
- Support offline quantization (loading pre-quantized MXFP8 checkpoints)

#### Step 4: Test Validation

```python
# test/srt/ascend/test_ascend_mxfp8_quantization.py

def test_mxfp8_linear():
    """Test MXFP8 linear layer correctness on NPU"""
    pass

def test_mxfp8_model_e2e():
    """End-to-end test: load model + MXFP8 quantization + inference"""
    pass

def test_mxfp8_accuracy():
    """Accuracy test: compare against FP16 baseline"""
    pass
```

#### Step 5 (Optional): Wan2.2 Diffusion Model MXFP8 Integration

This requires additional work on the `multimodal_gen` side, similar to MindIE-SD's `quantize()` function:
- Inject MXFP8 quantization into Linear layers of the DiT model
- Can reference MindIE-SD's timestep-aware quantization strategy

---

## 8. Effort Estimate

### 8.1 Task Breakdown and Time Estimate

#### Phase 0: msmodelslim MXFP8 Weight Adaptation (Fastest Path)

| Task | Detailed Description | Estimated Effort | Difficulty | Dependencies |
|------|---------------------|-----------------|------------|--------------|
| 0.1 Environment setup | Ascend NPU + CANN + torch_npu + SGLang | 1-2 days | Low | Hardware required |
| 0.2 Validate torch_npu MXFP8 API | Standalone script to test `npu_dynamic_mx_quant` | 0.5 days | Low | 0.1 |
| 0.3 Implement `ModelSlimMXFP8` Scheme | Add scheme file + modify dispatch logic | 1-2 days | Low-Medium | 0.2 |
| 0.4 Weight parameter alignment | Align weight_scale reshape/dtype with msmodelslim | 1-2 days | Medium | 0.3 |
| 0.5 End-to-end testing | Run inference with msmodelslim quantized model | 1-2 days | Medium | 0.4 |
| **Subtotal** | | **4-8 days** | | |

#### Phase 1: Basic MXFP8 Linear Support (Online Quantization, LLM Serving Side)

| Task | Detailed Description | Estimated Effort | Difficulty | Dependencies |
|------|---------------------|-----------------|------------|--------------|
| 1.1 Environment setup | Set up Ascend NPU dev env, install torch_npu, validate base APIs | 1-2 days | Low | Ascend hardware required |
| 1.2 Validate torch_npu MXFP8 API | Standalone test of `npu_dynamic_mx_quant` and `npu_quant_matmul` MXFP8 mode | 1 day | Low | 1.1 |
| 1.3 Implement `NPUMXFP8LinearMethod` | Implement MXFP8 Linear method referencing MindIE-SD | 2-3 days | Medium | 1.2 |
| 1.4 Register to quantization framework | Modify `fp8.py`, `__init__.py` and other registration files | 0.5 days | Low | 1.3 |
| 1.5 Weight loading adaptation | Support MXFP8 online quantization weight loading | 1-2 days | Medium | 1.3 |
| 1.6 Unit tests | Linear layer correctness, accuracy regression tests | 1-2 days | Low | 1.3 |
| 1.7 End-to-end testing | Complete inference validation with an LLM (e.g., Llama-8B) | 1-2 days | Medium | 1.5 |
| 1.8 Performance tuning | Profiling, NZ format optimization, throughput testing | 2-3 days | Medium-High | 1.7 |
| **Subtotal** | | **9.5-15.5 days** | | |

#### Phase 2: MXFP8 MoE Support

| Task | Detailed Description | Estimated Effort | Difficulty | Dependencies |
|------|---------------------|-----------------|------------|--------------|
| 2.1 Analyze MoE quantization architecture | Study existing `fused_moe_method_npu.py` (W4A4) | 1 day | Medium | -- |
| 2.2 Implement MXFP8 MoE method | Implement MXFP8 FusedMoE on NPU | 3-5 days | High | 2.1, Phase 1 |
| 2.3 Test DeepSeek-V2 and other MoE models | End-to-end MoE model validation | 2-3 days | Medium | 2.2 |
| **Subtotal** | | **6-9 days** | | |

#### Phase 3: Wan2.2 Diffusion Model MXFP8 Support

| Task | Detailed Description | Estimated Effort | Difficulty | Dependencies |
|------|---------------------|-----------------|------------|--------------|
| 3.1 Analyze SGLang Diffusion architecture | Understand `multimodal_gen` subsystem, Wan2.2 Pipeline | 1-2 days | Medium | -- |
| 3.2 Design Diffusion MXFP8 quantization plan | Reference MindIE-SD's layer replacement pattern, design SGLang Diffusion quantization interface | 1-2 days | Medium | 3.1 |
| 3.3 Implement DiT Linear layer MXFP8 | Inject MXFP8 into Linear layers of Wan2.2 DiT | 2-3 days | Medium | 3.2, Phase 1 |
| 3.4 (Optional) Timestep-aware quantization | Reference MindIE-SD's `TimestepPolicyConfig`, use different strategies at different denoising steps | 2-3 days | High | 3.3 |
| 3.5 (Optional) FP8 Attention quantization | Reference MindIE-SD's `FP8RotateQuantFA` | 3-5 days | High | 3.3 |
| 3.6 End-to-end Wan2.2 testing | Generated video quality validation, performance benchmarking | 2-3 days | Medium | 3.3 |
| **Subtotal** | | **11-18 days** (including optional items) | | |

#### Phase 4: Code Submission and Upstream Merge

| Task | Detailed Description | Estimated Effort | Difficulty | Dependencies |
|------|---------------------|-----------------|------------|--------------|
| 4.1 Code style alignment | Conform to SGLang code style, lint, type annotations | 1 day | Low | -- |
| 4.2 Documentation | Update docs, README, usage instructions | 1 day | Low | -- |
| 4.3 PR creation and review | Create PR, respond to review comments, CI debugging | 3-5 days | Medium | All |
| **Subtotal** | | **5-7 days** | | |

### 8.2 Total Effort Summary

| Phase | Scope | Effort | Cumulative |
|-------|-------|--------|------------|
| **Phase 0** ⭐ | **msmodelslim MXFP8 weight adaptation** | **4-8 days** | **4-8 days** |
| **Phase 1** | Online MXFP8 Linear (LLM) | 9.5-15.5 days | 13.5-23.5 days |
| **Phase 2** | MXFP8 MoE | 6-9 days | 19.5-32.5 days |
| **Phase 3** | Wan2.2 Diffusion MXFP8 | 11-18 days | 30.5-50.5 days |
| **Phase 4** | Submission and merge | 5-7 days | 35.5-57.5 days |
| **Total** | | **~1.5-3 months** (1 person full-time) | |

> ⭐ **Recommended to prioritize Phase 0**: If the team already has msmodelslim-quantized MXFP8 weights, the modelslim Scheme path is fastest — about 1-2 weeks to get running.

### 8.3 Minimum Viable Product (MVP)

**MVP Plan: Phase 0 (msmodelslim MXFP8 Weight Adaptation)**

- **Effort**: ~**1-2 weeks**
- **Output**: msmodelslim-quantized MXFP8 models can run inference on SGLang + Ascend NPU
- **Usage**: `--quantization modelslim` (auto-detects `quant_model_description.json`)
- **Risk**: Very low — framework exists, APIs exist, only a new Scheme class needs to be added
- **Advantage**: Fully compatible with the team's existing msmodelslim toolchain

If Online MXFP8 is also needed (without pre-quantized weights), add **Phase 1** on top:
- **Additional effort**: ~2-3 weeks
- **Output**: `--quantization mxfp8` for online quantization inference from FP16 models

### 8.4 Risks and Dependencies

| Risk | Impact | Mitigation |
|------|--------|------------|
| **torch_npu API version compatibility** | `npu_dynamic_mx_quant` may only be supported on specific CANN versions | Confirm CANN version requirements in advance (≥8.0.RC3) |
| **Ascend hardware availability** | Development and testing require Atlas 800I A2/A3 | Confirm team has available devices |
| **Upstream review cycle** | SGLang community PR review may take a long time | Communicate early with @ping1jing2 / @OrangeRedeng |
| **MXFP8 accuracy issues** | block-size-32 may have insufficient accuracy on some models | Prepare accuracy benchmark comparison data |
| **Wan2.2 Diffusion NPU support** | `multimodal_gen` may not yet be adapted for NPU | May need to resolve Diffusion NPU baseline support first |
| **YChange01 already claimed in Issue** | MXFP8/MXFP4 task has been claimed by YChange01 | Need to coordinate division of work or confirm whether collaboration is needed |

---

## 9. Recommended Advancement Strategy

### 9.1 Short-term (Weeks 1-2) — msmodelslim MXFP8 Adaptation

1. **Align with community**: Comment in Issue #14424, explain that we are doing MXFP8 NPU adaptation, coordinate with YChange01
2. **Environment setup**: Prepare Ascend NPU development environment + CANN + torch_npu
3. **API validation**: Use a standalone script to verify `torch_npu.npu_dynamic_mx_quant` availability
4. **Implement ModelSlimMXFP8 Scheme**: Add scheme + modify dispatch logic
5. **Run inference with msmodelslim MXFP8 weights**: End-to-end validation

### 9.2 Medium-term (Weeks 3-5) — Online MXFP8 + MoE

6. **Implement Online MXFP8 Linear**: Complete `NPUMXFP8LinearMethod` core code
7. **Integration testing**: Run Online MXFP8 inference on LLM models
8. **MoE support**: Extend to MoE models if needed
9. **Submit PR**: Phase 0 + Phase 1 code submission

### 9.3 Long-term (Weeks 6-10) — Diffusion + Optimization

10. **Diffusion support**: Wan2.2 MXFP8 quantization integration
11. **Performance optimization**: Throughput, latency comparison benchmarks
12. **Community review and merge**

---

## 10. Reference Resources

### Code Repositories
- SGLang: `./sglang/` (cloned locally)
- MindIE-SD: `./MindIE-SD/` (cloned locally)

### Key Documentation
- [SGLang Official Docs](https://docs.sglang.io/index.html)
- [SGLang Quantization Docs](https://docs.sglang.io/advanced_features/quantization.html)
- [Issue #14424 - Ascend NPU Quantization Roadmap](https://github.com/sgl-project/sglang/issues/14424)
- [Issue #17093 - MXFP8 GPU-side Support](https://github.com/sgl-project/sglang/issues/17093)
- [PR #17449 - MXFP8 CUDA Implementation](https://github.com/sgl-project/sglang/pull/17449)
- [Issue #18258 - ModelOpt MXFP8 Loader](https://github.com/sgl-project/sglang/issues/18258)
- [msmodelslim - Ascend Model Compression Tool](https://gitcode.com/Ascend/msmodelslim)

### Key Code Paths
- SGLang quantization core: `sglang/python/sglang/srt/layers/quantization/`
- SGLang NPU backend: `sglang/python/sglang/srt/hardware_backend/npu/`
- SGLang FP8 implementation: `sglang/python/sglang/srt/layers/quantization/fp8.py`
- SGLang Wan2.2: `sglang/python/sglang/multimodal_gen/configs/pipeline_configs/wan.py`
- MindIE-SD MXFP8: `MindIE-SD/mindiesd/quantization/layer.py` → `W8A8MXFP8QuantLinear`
- MindIE-SD quantization entry: `MindIE-SD/mindiesd/quantization/quantize.py`
- SGLang ModelSlim module: `sglang/python/sglang/srt/layers/quantization/modelslim/`
- SGLang ModelSlim Scheme: `sglang/python/sglang/srt/layers/quantization/modelslim/schemes/`
