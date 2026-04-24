# CLAUDE.md

本仓库用于研究和实现 SGLang **Diffusion 侧**在华为 Ascend NPU 上的 MXFP8/MXFP4 量化适配（Wan2.2 等 Diffusion 模型）。
**如果涉及 LLM serving (`srt`) 侧的功能开发（如 MXFP8/MXFP4），请务必参考 `vllm-ascend` 的实现模式。**
*注：Qwen3 和 Qwen3.5 模型在 SGLang 内部共用底层 Linear/MoE 算子，因此量化实现代码完全一致。*

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion), [sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLM Qwen3)
- **Fork**: https://github.com/TallMessiWu/sglang

## 分支规则

| 分支                     | 说明                                                                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `junlin`               | 主线，**不动**                                                                                                                         |
| `junlin_mxfp4`         | Diffusion MXFP8 + MXFP4 在线量化                                                                                                             |
| `junlin_mxfp4_offline` | Diffusion 在 `junlin_mxfp4` 基础上增加 MXFP4 离线加载                                                                                      |
| `junlin_qwen3_dense`   | LLM 侧，Qwen3 / 3.5 dense 模型 MXFP8 量化适配                                                                                      |
| `junlin_qwen3_dense_w4a8` | LLM 侧，Dense W4A8 在线量化（MXFP4 双级，`--quantization mxfp4_npu`）；离线 W4A8 占位符存在但待修复 |
| `junlin_qwen3_dense_w4a4` | **当前工作分支**，LLM 侧，在 w4a8 基础上新增 W4A4 在线量化（单级 MXFP4，`--quantization mxfp4w4a4_npu`）+ 离线 W4A4（INT4 ModelSlim） |

## 在线/离线量化模式

- **在线量化（Online）**：加载 FP16/BF16 权重，`process_weights_after_loading` 中实时量化，用 `--quantization mxfp8/mxfp4` 触发
- **离线量化（Offline ModelSlim）**：加载 msmodelslim 预量化权重，用 `--quantization modelslim` 触发，scheme 由 `quant_model_description.json` 自动检测

## 实现进度

| 功能                                   | 分支                   | 在线实现状态            | 离线实现状态            |
| -------------------------------------- | ---------------------- | ----------------------- | ----------------------- |
| Diffusion MXFP8                        | `junlin`             | ✅                      | ✅                      |
| Diffusion MXFP4                        | `junlin_mxfp4`       | ✅                      | ✅                      |
| LLM (Qwen3 & 3.5) Dense W8A8 (MXFP8)   | `junlin_qwen3_dense` | ✅ (已对齐 vllm-ascend) | ✅ (已对齐 vllm-ascend) |
| LLM (Qwen3 & 3.5) Dense W4A8 (MXFP4/8) | `junlin_qwen3_dense_w4a8` | ✅ 在线已实现（`mxfp4_npu`，双级） | ❌ 待修复（`W4A8_MXFP` 占位符误用 `ModelSlimMXFP8Scheme`） |
| LLM (Qwen3 & 3.5) Dense W4A4 (MXFP4)   | `junlin_qwen3_dense_w4a4` | ✅ 在线已实现（`mxfp4w4a4_npu`，单级 MXFP4） | ✅ 离线已实现（`W4A4_DYNAMIC` → `ModelSlimW4A4Int4` + `NPU_W4A4DynamicLinearMethod`，**INT4 非 MXFP4**） |
| LLM (Qwen3 & 3.5) MoE W8A8 (MXFP8)     | 待定                   | ❌ 待实现               | ❌ 待实现               |
| LLM (Qwen3 & 3.5) MoE W4A8 (MXFP4/8)   | 待定                   | ❌ 待实现               | ❌ 待实现               |
| LLM (Qwen3 & 3.5) MoE W4A4 (MXFP4)     | 待定                   | ❌ 待实现               | ❌ 待实现               |

## 关键代码路径

### Diffusion 侧（multimodal_gen）

量化层目录：`sglang/python/sglang/multimodal_gen/runtime/layers/quantization/`

| 文件                          | 作用                                          |
| ----------------------------- | --------------------------------------------- |
| `__init__.py`               | 注册表 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` |
| `mxfp8_npu.py`              | MXFP8 在线量化                                |
| `mxfp4_npu.py`              | MXFP4 在线量化（双级）                        |
| `modelslim.py`              | ModelSlim 分发 `_get_scheme_from_parts()`   |
| `modelslim_mxfp8_scheme.py` | ModelSlim MXFP8 离线                          |
| `modelslim_mxfp4_scheme.py` | ModelSlim MXFP4 离线（双级）                  |

### LLM 侧（srt）

量化层目录：`sglang/python/sglang/srt/layers/quantization/modelslim/`

| 文件                                  | 作用                                                   |
| ------------------------------------- | ------------------------------------------------------ |
| `modelslim.py`                      | `ModelSlimConfig`：`get_quant_method` 分发、注册       |
| `schemes/modelslim_mxfp8.py`        | ModelSlim MXFP8 离线 scheme（W8A8）                    |
| `schemes/modelslim_w8a8_int8.py`    | ModelSlim W8A8 Int8 离线 scheme                        |

在线量化：
- `--quantization mxfp8` → `linear_method_npu.py` → `NPUMXFP8LinearMethod`
- `--quantization mxfp4_npu` → `layers/quantization/npu_mxfp4.py` → `NPUMxfp4Config` → `NPUMXFP4W4A8LinearMethod`（双级 W4A8）
- `--quantization mxfp4w4a4_npu` → `layers/quantization/npu_mxfp4_w4a4.py` → `NPUMxfp4W4A4Config` → `NPUSingleLevelMXFP4LinearMethod`（单级 W4A4）

其他关键文件：

- `srt/models/qwen3.py` — Qwen3 / 3.5 模型定义，`EntryClass = Qwen3ForCausalLM`
- `srt/models/registry.py` — `ModelRegistry`，扫描 `sglang.srt.models` 注册所有 `EntryClass`
- `srt/layers/rotary_embedding/base.py` — RoPE 实现，NPU 路径 import `sgl_kernel_npu`
- `srt/model_loader/loader.py` — `DefaultModelLoader`：`_get_quantization_config` → `_initialize_model`
- `MindIE-SD/mindiesd/quantization/layer.py` — NPU 量化参考实现 (Diffusion)
- `vllm-ascend/vllm_ascend/quantization/methods/w8a8_mxfp8.py` — NPU 量化参考实现 (LLM)
- `msmodelslim/.../save/ascendv1.py` — MXFP4 权重导出格式

## 注意事项

- **CANN 版本**: MXFP8 需 ≥ 8.0.RC3；MXFP4 最低版本待确认
- **硬件**: Atlas 800I A2/A3（`DualLevelQuantBatchMatmul` 仅支持 Ascend 950，A2/A3 不支持）
- **CPU offload**：`dit_cpu_offload` 默认 True，`process_weights_after_loading` 中需手动 `.to("npu:X")` 后再调用量化 API
- **bias 精度**：量化 matmul 要求 bias 为 `float32`
- **tensor reshape**：diffusion 输入可能是 3D `[batch, seq, hidden]`，NPU 量化 API 需 2D，apply 中先 reshape 后 restore
- 与社区 YChange01 协调 MXFP8/MXFP4 工作分工（已在 Issue #14424 认领）

## 已知陷阱

- **量化不生效/乱码输出**：先验证模型是否注册成功。若 `sgl_kernel_npu` 某 kernel 不存在会导致模型模块 import 失败，`ModelRegistry` 静默跳过，fallback 到 HF Transformers（无量化感知），FP8 权重被当 BF16 解读 → 乱码。
  ```bash
  python3 -c "from sglang.srt.models.registry import ModelRegistry; print(list(ModelRegistry.models.keys()))"
  python3 -c "from sglang.srt.models.qwen3 import Qwen3ForCausalLM; print('OK')"
  ```
  修复：`sgl_kernel_npu` 非核心 kernel 的 import 改为 try/except + `None` fallback（见 `rotary_embedding/base.py`）。

- **`process_weights_after_loading` 中 transpose 不加 `.contiguous()`**：`npu_quant_matmul` 通过 strides 感知内存布局，`.contiguous()` 会物理重排数据破坏 block-scale 映射 → 乱码。用 `.data` 原地赋值保留 non-contiguous view（与 vllm-ascend 一致）。

> 详细 API 参考和实现模式见 `/mxfp4-impl-ref` skill。

## 代码提交
代码提交时必须使用gitmoji-commit这个skill。