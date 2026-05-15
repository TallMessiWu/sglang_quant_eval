---
name: trace-quant-path
description: Trace the full code path for a quantization method in SGLang - from CLI args through model loading to inference
---

# Trace Quantization Code Path

When the user wants to understand the full execution flow of a quantization method:

Trace these stages in order, reading the actual source code:

## Stage 1: CLI → Config
- `sglang/python/sglang/srt/server_args.py`: `--quantization` arg → `QUANTIZATION_CHOICES`
- `sglang/python/sglang/srt/configs/model_config.py`: `ModelConfig.__init__()` resolves quantization config

## Stage 2: Config → Method
- `sglang/python/sglang/srt/layers/quantization/__init__.py`: `get_quantization_config()` factory
- The specific config class's `get_quant_method(layer, prefix)` returns a `QuantizeMethodBase`

## Stage 3: Model Loading
- `sglang/python/sglang/srt/model_loader/loader.py`: `_get_quantization_config()` with NPU-specific handling
- Weight iteration and loading through quantization-aware loaders
- `QuantizeMethodBase.process_weights_after_loading()` post-processing

## Stage 4: Inference
- Model's forward pass calls quantized layers
- `LinearBase` delegates to `QuantizeMethodBase.apply(layer, x, bias)`
- Quantized GEMM execution (CUDA kernels or NPU operators)

## Stage 5: NPU Specifics (if applicable)
- Check `sglang/python/sglang/srt/hardware_backend/npu/` for hardware-specific overrides
- NPU graph runner: `npu_graph_runner.py`
- NPU quantization methods: `quantization/linear_method_npu.py`

For each stage, show the relevant code snippet and file:line_number.
