# Target-specific wheel contract

Read this reference for build-target, provider-staging, Gemma RMSNorm, wheel-content, or paired SGLang work.

## Stable APIs

Every target wheel exposes:

```python
from sgl_kernel_npu.norm.gemma_rmsnorm import npu_gemma_rms_norm
from sgl_kernel_npu.norm.add_rmsnorm_bias import add_gemma_rms_norm
```

- `npu_gemma_rms_norm(input, weight, eps) -> (output, rstd)`
- `add_gemma_rms_norm(input, residual, weight, eps) -> (norm_output, residual_sum)`
- Gemma checkpoint weight is an offset; logical gamma is `1 + weight`.

## Provider mapping

| Hardware | Public selector | Canonical wheel target | Installed ordinary provider |
| --- | --- | --- | --- |
| A2 | `910B`, `Ascend910B1` | `Ascend910` | native `npu_gemma_rms_norm` |
| A3 | `910`, `910C`, `Ascend910_9382` | `Ascend910` | native `npu_gemma_rms_norm` |
| A5 | `950`, `Ascend950`, `Ascend950PR_*`, `Ascend950DT_*` | `Ascend950` | `npu_rms_norm(input, 1 + weight, eps)` |

Residual Gemma RMSNorm uses `npu_add_rms_norm(input, residual, 1 + weight, eps)` on all targets.

## Build-time staging

The source tree may contain the public 910 implementation and a private 950 template. During setuptools `build_py`:

1. read `SGL_KERNEL_NPU_BUILD_TARGET`;
2. stage exactly one implementation as `build/lib/sgl_kernel_npu/norm/gemma_rmsnorm.py`;
3. remove the unused private template from staging;
4. package the specialized staging tree.

Reject import-time branches, runtime SoC queries, module-presence probes, and wheels containing both providers.

## Logical target versus compiler target

`Ascend950` identifies the Python wheel provider. The main C++ bundle may still compile with `Ascend910_9382` as a compatibility target while other kernels are not A5 compiler-compatible. This does not invalidate the 950 Python provider.

## Commands

```bash
bash build.sh -h
bash build.sh -a kernels 910
bash build.sh -a kernels 950

python -m pytest \
  tests/python/sgl_kernel_npu/test_build_targets.py \
  tests/python/sgl_kernel_npu/test_gemma_rmsnorm_provider.py -q

pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
```

## Acceptance checklist

- `build/lib`, wheel, and installed `site-packages` contain the selected provider only.
- 950 ordinary and residual correctness tests pass.
- 950 Qwen3.5 load, warmup, and deterministic request pass.
- 910B/910C import and native-provider smoke pass.
- Performance claims identify hardware, shapes, dtype, synchronization, warmup, and iterations.

Current paired work is sgl-kernel-npu PR #638 and SGLang PR #32745. Re-query GitHub before reporting state or head SHA.
