# CLAUDE.md

## Project Overview

本仓库用于研究和实现 SGLang **Diffusion 侧**在华为 Ascend NPU 上的 MXFP8/MXFP4 量化适配（Wan2.2 等 Diffusion 模型）。
**如果涉及 LLM serving (`srt`) 侧的功能开发（如 MXFP8/MXFP4），请务必参考 `vllm-ascend` 的实现模式。**

- **关联 Issue**: [sgl-project/sglang#14424](https://github.com/sgl-project/sglang/issues/14424) (Diffusion), [sgl-project/sglang#21584](https://github.com/sgl-project/sglang/issues/21584) (LLM Qwen3)
- **Fork**: https://github.com/TallMessiWu/sglang

## 分支规则

| 分支 | 说明 |
|------|------|
| `junlin` | 主线，**不动** |
| `junlin_mxfp4` | Diffusion MXFP8 + MXFP4 在线量化 |
| `junlin_mxfp4_offline` | Diffusion 在 `junlin_mxfp4` 基础上增加 MXFP4 离线加载 |
| `junlin_qwen3_dense` | **当前工作分支**，LLM 侧，Qwen3 dense 模型 MXFP8 适配<br>👉 **当前紧要任务：实现 Dense W4A8 和 Dense W4A4 (MXFP4)** |

## 在线/离线量化模式

- **在线量化（Online）**：加载 FP16/BF16 权重，`process_weights_after_loading` 中实时量化，用 `--quantization mxfp8/mxfp4` 触发
- **离线量化（Offline ModelSlim）**：加载 msmodelslim 预量化权重，用 `--quantization modelslim` 触发，scheme 由 `quant_model_description.json` 自动检测

## 实现进度

| 功能 | 分支 | 在线实现状态 | 离线实现状态 |
|------|------|--------------|--------------|
| Diffusion MXFP8 | `junlin` | ✅ | ✅ |
| Diffusion MXFP4 | `junlin_mxfp4` | ✅ | ✅ |
| LLM (Qwen3) Dense W8A8 (MXFP8) | `junlin_qwen3_dense` | ✅ (已对齐 vllm-ascend) | ✅ (已对齐 vllm-ascend) |
| LLM (Qwen3) Dense W4A8 (MXFP4/8) | 待定 | ❌ 待实现 | ❌ 待实现 |
| LLM (Qwen3) Dense W4A4 (MXFP4) | 待定 | ❌ 待实现 | ❌ 待实现 |
| LLM (Qwen3) MoE W8A8 (MXFP8) | 待定 | ❌ 待实现 | ❌ 待实现 |
| LLM (Qwen3) MoE W4A8 (MXFP4/8) | 待定 | ❌ 待实现 | ❌ 待实现 |
| LLM (Qwen3) MoE W4A4 (MXFP4) | 待定 | ❌ 待实现 | ❌ 待实现 |

## 关键代码路径

量化层目录：`sglang/python/sglang/multimodal_gen/runtime/layers/quantization/`

| 文件 | 作用 |
|------|------|
| `__init__.py` | 注册表 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` |
| `mxfp8_npu.py` | MXFP8 在线量化 |
| `mxfp4_npu.py` | MXFP4 在线量化（双级） |
| `modelslim.py` | ModelSlim 分发 `_get_scheme_from_parts()` |
| `modelslim_mxfp8_scheme.py` | ModelSlim MXFP8 离线 |
| `modelslim_mxfp4_scheme.py` | ModelSlim MXFP4 离线（双级） |

其他关键文件：
- `server_args.py` — `quantization` 字段
- `loader/.../transformer_loader.py` — 显式 quantization 优先于自动检测
- `MindIE-SD/mindiesd/quantization/layer.py` — NPU 量化参考实现 (Diffusion)
- `vllm-ascend/vllm_ascend/quantization/` — NPU 量化参考实现 (LLM, 如 `w8a8_mxfp8.py` 的加载预转置模式)
- `msmodelslim/.../save/ascendv1.py` — MXFP4 权重导出格式

## 注意事项

- **CANN 版本**: MXFP8 需 ≥ 8.0.RC3；MXFP4 最低版本待确认
- **硬件**: Atlas 800I A2/A3（`DualLevelQuantBatchMatmul` 仅支持 Ascend 950，A2/A3 不支持）
- **CPU offload**：`dit_cpu_offload` 默认 True，`process_weights_after_loading` 中需手动 `.to("npu:X")` 后再调用量化 API
- **bias 精度**：量化 matmul 要求 bias 为 `float32`
- **tensor reshape**：diffusion 输入可能是 3D `[batch, seq, hidden]`，NPU 量化 API 需 2D，apply 中先 reshape 后 restore
- 与社区 YChange01 协调 MXFP8/MXFP4 工作分工（已在 Issue #14424 认领）

> 详细 API 参考和实现模式见 `/mxfp4-impl-ref` skill。
