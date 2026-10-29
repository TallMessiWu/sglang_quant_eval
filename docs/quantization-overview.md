# Ascend MXFP 量化概览

本文维护格式、入口与源码定位；**实现存在不等于所有硬件/并行配置已验收**。当前 PR 与证据统一见 [目标计划](mxfp-refactoring-plan.md)，目标 checkout 见 [分支表](branches.md)。

## 格式与来源

| 路径 | 在线入口（原始 BF16/FP16） | 离线来源 | 关键区别 |
| --- | --- | --- | --- |
| SRT W8A8 MXFP8 | `--quantization mxfp8` | ModelSlim 描述自动选择 scheme | FP8 payload、UE8M0 block scale；partial-scale/TP 边界需处理 |
| SRT W4A8 MXFP | `--quantization mxfp_w4a8` | ModelSlim W4A8 MXFP4 | FP4 权重、动态 FP8 activation；不同于 INT4 W4A8 |
| SRT W4A4 MXFP | `--quantization mxfp4` | ModelSlim W4A4 MXFP4 | Dense 在线双级与 MoE 单级不能混为一种 scale 契约 |
| Diffusion MXFP8 | `--quantization mxfp8` | ModelSlim MXFP8 | Linear 与 MXFP8 attention 独立选择；attention 需实际日志确认 |
| Diffusion MXFP4 | `--quantization mxfp4_npu` | ModelSlim 导出描述 | 以具体 exporter 的单级/双级格式为准，不按位宽猜 scale |

- 在线权重在初始化期间准备：通常在 post-load；当前在线 W4A8 还在逻辑权重加载完成时立即量化、释放浮点 backing。**推理时动态量化的是 activation，不是重新量化 checkpoint。**
- 离线通常不必显式传 `--quantization modelslim`；由 `quant_model_description.json` 检测。packed FP4 为 `uint8 [out,in/2]`，不二次打包。
- Qwen3/Qwen3.5 复用底层 Linear/MoE 方法；专家、attention、共享 MLP、视觉塔的实际精度应按各自配置报告，不能简称“全模型 W4”。
- 具体 NPU API、CANN/torch_npu、kernel wheel/target 和 TP 后 shape 决定支持范围；不提供脱离版本的最低版本或整代际通用保证。

## 分层与源码入口

以下路径相对**选定 worktree 的 `python/sglang/`**。不同 PR 的新增文件不一定已在 main，先检查实际路径，不使用旧 `sglang/python/` 假设。

| 职责 | 入口 |
| --- | --- |
| CLI/config 与注册 | `srt/server_args.py`、`srt/layers/quantization/__init__.py` |
| 在线 W4A8 / W4A4 配置 | `srt/layers/quantization/npu_mxfp4.py`、`npu_mxfp4_w4a4.py` |
| NPU Linear/MoE 执行 | `srt/hardware_backend/npu/quantization/linear_method_npu.py`、`moe_methods.py` |
| 在线 MoE 与完成度 loader | 同目录 `online_moe_methods.py`、`online_weight_loader.py` |
| ModelSlim 检测、参数和 loader | `srt/layers/quantization/modelslim/modelslim.py`、`schemes/` |
| MoE dispatch / runner | `srt/layers/moe/token_dispatcher/ascend_ops.py`、`moe_runner/ascend.py`；具体分支可能仍有旧接口 |
| Diffusion configs / schemes | `multimodal_gen/runtime/layers/quantization/` 下 `mxfp8_online.py`、`mxfp4_npu.py`、`modelslim_mxfp*_scheme.py` |
| Diffusion attention | `multimodal_gen/runtime/layers/attention/`；解析 selector/resolver 及实际 backend |
| 代际契约 | 独立 kernel 构建期 provider；跨组件表示按需用 `srt/hardware_backend/npu/device_op.py` |

公共执行只消费 prepared tensors；在线量化与离线加载保留各自参数、metadata 和生命周期。不要为去重而合并不兼容的表示。G2 的公共执行与能力试点分别在发布 PR 中，不当作已合入 main 的功能。

## 必须核对的边界

1. MXFP8 block=32、pair=2；奇数 block 需 pad，TP scale 起点按全局 K offset 算。K=4304 的 135 个 block 不能直接 pair reshape。
2. 在线 W4A8 对齐失败的视觉 fc2 保留 BF16，不宣称全视觉 W4A8。
3. 异构 gate/up 的 Dense loader 与 MoE fused expert loader 是两种问题；不得只改 JSON 让描述匹配。
4. weight/scale transpose、ND/NZ、L0/L1、COUNT/CUMSUM 与输出 dtype 都属于 API 契约；不要从 rank 或方法名猜布局。
5. CPU importability、加载完成、真实 NPU 数值、graph replay 与 serving 性能分别验证。详见 [已知陷阱](known-pitfalls.md)。

## 参考来源

- LLM 优先对照 `vllm-ascend/vllm_ascend/quantization/`，Diffusion 优先 `MindIE-SD/mindiesd/`。
- 离线格式以当前 `msmodelslim/` exporter 和真实 checkpoint 描述为准；本仓 `*_quant_descriptions/` 只是样例，不是完整模型配置。
- [双级量化 API 摘录](npu-api/)用于理解 shape，不能替代安装版本的签名/硬件验证。
- 普通运行见 [LLM](../llm/README.md)、[Diffusion](../diffusion/README.md)；不要把旧诊断补丁作为启动前置。
