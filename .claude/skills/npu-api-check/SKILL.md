---
name: npu-api-check
description: Check torch_npu API usage patterns in the codebase - find how specific NPU operators are called and what parameters they expect
---

# NPU API Usage Check

When the user asks about a specific `torch_npu` or `torch.ops.npu` API:

1. **Search both codebases** for the API name:
   - `sglang/python/sglang/srt/hardware_backend/npu/` (SGLang NPU backend)
   - `MindIE-SD/mindiesd/` (MindIE-SD reference)

2. **For each usage found**, extract:
   - Full function signature with all parameters
   - Data types of inputs/outputs
   - Context (which quantization method, which layer type)
   - Any preprocessing/postprocessing around the call

3. **Cross-reference** with known APIs:
   - `npu_dynamic_mx_quant` → MXFP8 dynamic quantization, returns `(qx, scale)`
   - `npu_quant_matmul` → quantized matmul (INT8/FP8, supports group_sizes for MXFP8)
   - `npu_dynamic_dual_level_mx_quant` → MXFP4 dual-level quant, returns `(y, l0_scale, l1_scale)`
   - `npu_dual_level_quant_matmul` → MXFP4 dual-level matmul; **x2 must be FRACTAL_NZ (format=29)**; x2_level0_scale must be `[in/512, out]` (transposed vs what the quant op returns)
   - `npu_quantize` → static INT8 quantization
   - `npu_dynamic_quant` → dynamic per-token INT8 quantization
   - `npu_weight_quant_batchmatmul` → weight-only quantized matmul
   - `npu_format_cast(tensor, acl_format)` → NPU internal format conversion (NZ=29); only 2 positional args, `customize_dtype` is keyword-only
   - `npu_fused_infer_attention_score_v2` → fused attention with quantization

4. If the API is not found in either codebase, search the web for `torch_npu` documentation.

Output: API signature, all usage examples found, and parameter explanations.
