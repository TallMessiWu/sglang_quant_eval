# Ascend MXFP 量化概览

本文保存相对稳定的能力矩阵、实现入口和跨仓参考关系。实时 PR、分支、HEAD 与 worktree 不在这里维护，统一见 [branches.md](branches.md)。

## 参考优先级

- SGLang LLM serving (`srt`)：优先对齐 `vllm-ascend` 的实现语义、tensor layout 和 `torch_npu` API 契约。
- SGLang Diffusion (`multimodal_gen`)：优先参考 `MindIE-SD`。
- 离线权重格式：以当前 `msmodelslim` 导出实现和 checkpoint 内 `quant_model_description.json` 为准。
- Qwen3 与 Qwen3.5 共享底层 Linear/MoE 量化能力；模型专属代码只负责结构映射、视觉塔和加载入口等适配。

## 两种权重来源

| 模式 | 输入权重 | 选择方式 | 主要处理位置 |
| --- | --- | --- | --- |
| Online | BF16/FP16 | `--quantization mxfp8` / `mxfp_w4a8` / `mxfp4` | `process_weights_after_loading` 实时量化 |
| Offline ModelSlim | 预量化 checkpoint | `quant_model_description.json` 自动识别 | ModelSlim scheme 注册物理 shape/dtype，post-load 只做布局整理 |

Offline 启动脚本通常不显式传 `--quantization modelslim`；以 checkpoint 描述自动检测为准。

## 能力状态

| 能力 | SGLang 状态 | 在线 | 离线 |
| --- | --- | --- | --- |
| Diffusion MXFP8 | 已合并 [#20922](https://github.com/sgl-project/sglang/pull/20922)，文档 [#24918](https://github.com/sgl-project/sglang/pull/24918) | 是 | 是 |
| Diffusion MXFP4 | 已合并 [#22338](https://github.com/sgl-project/sglang/pull/22338)，文档 [#25904](https://github.com/sgl-project/sglang/pull/25904) | 是 | 是 |
| Dense W8A8 MXFP8 | 已合并 [#22352](https://github.com/sgl-project/sglang/pull/22352)、[#28505](https://github.com/sgl-project/sglang/pull/28505) | 是 | 是 |
| Dense W4A8 MXFP | 已合并 [#23650](https://github.com/sgl-project/sglang/pull/23650) | 是 | 是 |
| Dense W4A4 MXFP4 | 已合并 [#23795](https://github.com/sgl-project/sglang/pull/23795)；packed loader 修复 [#32013](https://github.com/sgl-project/sglang/pull/32013) 已合并 | 双级 MXFP4 | 单级 packed MXFP4 |
| MoE W8A8 MXFP8 | 已合并 [#30768](https://github.com/sgl-project/sglang/pull/30768) | 是 | 是 |
| Qwen3.5 ModelSlim / 视觉塔 W8A8 适配 | Open Draft [#32266](https://github.com/sgl-project/sglang/pull/32266) | 复用已合并能力 | GDN mapping、视觉塔和 partial scale 适配 |
| MoE W4A8 MXFP | Open Draft [#32601](https://github.com/sgl-project/sglang/pull/32601) | 已实现，待当前分支验证闭环 | 已实现，待当前分支验证闭环 |
| MoE W4A4 MXFP4 | Open Draft [#32602](https://github.com/sgl-project/sglang/pull/32602) | 已实现，待当前分支验证闭环 | 已实现，待当前分支验证闭环 |
| Qwen3.5 Gemma RMSNorm on 950 | SGLang [#32745](https://github.com/sgl-project/sglang/pull/32745) + kernel [#638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | 构建期 wheel provider | 同一稳定 API |

“已实现”不等于当前分支已完成硬件验收；以 PR 正文的最新 Validation/TODO 和真实 A5/A2/A3 日志为准。

## SGLang 关键路径

以下路径都相对于具体 `sglang/<worktree>/python/sglang/`。

### Diffusion

- `multimodal_gen/runtime/layers/quantization/`
  - `mxfp8_npu.py`、`mxfp4_npu.py`：在线量化
  - `modelslim.py`：离线 scheme 分发
  - `modelslim_mxfp8_scheme.py`、`modelslim_mxfp4_scheme.py`：离线实现

### LLM Dense / ModelSlim

- `srt/layers/quantization/linear_method_npu.py`：NPU Dense quantized linear 方法
- `srt/layers/quantization/npu_mxfp4.py`：在线 W4A8 配置与 Linear 分发
- `srt/layers/quantization/npu_mxfp4_w4a4.py`：在线 W4A4 配置与 NPU 分发
- `srt/layers/quantization/modelslim/modelslim.py`：ModelSlim Linear/MoE scheme 分发
- `srt/layers/quantization/modelslim/schemes/`：MXFP8、W4A8、W4A4 离线 scheme

### LLM MoE

- `srt/hardware_backend/npu/quantization/moe_methods.py`：per-GMM NPU kernel 方法
- `srt/hardware_backend/npu/quantization/online_moe_methods.py`：在线 FusedMoE 入口
- `srt/hardware_backend/npu/moe/`：matmul、quant、activation wrapper
- `srt/layers/moe/moe_runner/ascend.py`：Ascend MoE 编排
- `srt/models/qwen3_moe.py`：Qwen MoE 模型与 router gate

### Qwen3.5 / 视觉塔 / kernel

- `srt/models/qwen3_vl.py`：视觉塔 quant config 透传
- `srt/layers/layernorm.py`：Gemma RMSNorm 的 SGLang 稳定 API 调用
- `sgl-kernel-npu/python/sgl_kernel_npu/norm/`：target wheel 中的 Gemma RMSNorm provider

## 稳定格式与边界

- ModelSlim W4A8/W4A4 权重为 packed FP4：`uint8 [out, in/2]`。创建参数时必须匹配物理 shape/dtype，post-load 不得二次打包。
- MXFP 是纯 scale 格式；ModelSlim 公共 MoE 路径若读取 offset，应注册空 `weight_offset`，不能伪造零点。
- 在线 W4A8 的 FP4 matmul 要求 `K % 32 == 0`；不对齐 Linear（例如视觉 FC2 的 `K=4304`）回退 BF16。
- 离线 MXFP8 要保留最后一个 partial 32 元素 block：placeholder 用 `ceil(K/32)`，奇数 scale 数在 pair reshape 前补齐。
- MoE MXFP 的 dtype、group-list 语义、transpose/NZ 顺序和 dispatcher 限制见 [known-pitfalls.md](known-pitfalls.md)。
- A5 上运行本仓 LLM 脚本时保留 `ASCEND_USE_FIA=1`；它处理的是 attention backend 兼容性，不是量化精度开关。

## 启动与验证入口

- `llm/qwen3.5_dense_*.sh`：Qwen3.5 Dense BF16 / W8A8
- `llm/qwen3.5_moe_*.sh`：Qwen3.5 MoE BF16 / W8A8 / W4A8 / W4A4
- `llm/qwen3_*`：Qwen3 对照脚本
- `diffusion/`：Wan2.2 Diffusion 量化与 PR 记录

结论必须区分：静态检查、CPU 回归、单 kernel probe、模型加载/warmup、确定性请求、精度评测和性能评测。没有对应硬件日志时，不把静态检查写成 e2e 通过。
