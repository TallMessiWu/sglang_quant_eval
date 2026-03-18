---
name: sglang-quant-lookup
description: Look up SGLang quantization implementation details - find how a specific quantization method is implemented, registered, and used
---

# SGLang Quantization Lookup

When the user asks about a specific quantization method in SGLang, search the following locations in order:

1. **Registration**: `sglang/python/sglang/srt/layers/quantization/__init__.py` - find the method name in `QUANTIZATION_METHODS` dict
2. **Config class**: The mapped config file (e.g., `fp8.py`, `w8a8_int8.py`)
3. **Method implementation**: Look for `LinearMethodBase` and `FusedMoEMethodBase` subclasses in that file
4. **Kernel**: Check `fp8_kernel.py`, `int8_kernel.py`, or `sglang/python/sglang/srt/jit_kernel/` for compute kernels
5. **NPU specialization**: Check `sglang/python/sglang/srt/hardware_backend/npu/quantization/` for NPU-specific overrides
6. **Server args**: Check `sglang/python/sglang/srt/server_args.py` for CLI flag registration
7. **Tests**: Check `sglang/test/srt/ascend/` for NPU tests, `sglang/test/` for general tests

Present findings in a structured format: registration → config → weights → forward pass → NPU support status.
