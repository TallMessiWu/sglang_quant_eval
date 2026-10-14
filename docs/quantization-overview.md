# Ascend MXFP 量化概览

本文保存相对稳定的能力矩阵、实现入口和跨仓参考关系。分支、HEAD 与 worktree 定位统一见 [branches.md](branches.md)。

PR 合入情况、重构目标、任务拆分与验收进度统一见 [MXFP 重构计划](mxfp-refactoring-plan.md)；早期架构审阅保存在 [2026-10-10 HTML 报告](reports/mxfp-architecture-roadmap-2026-10-10.html)，其中状态仅代表当时快照。

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
| Qwen3.5 ModelSlim / 视觉塔 W8A8 适配 | 历史 [#32266](https://github.com/sgl-project/sglang/pull/32266)；组合参考 [#32745](https://github.com/sgl-project/sglang/pull/32745) / [#36426](https://github.com/sgl-project/sglang/pull/36426) | 复用已合并能力 | GDN mapping、视觉塔和 partial scale 适配 |
| MoE W4A8 MXFP | 离线已合并 [#30318](https://github.com/sgl-project/sglang/pull/30318)；在线入口 [#43739](https://github.com/sgl-project/sglang/pull/43739)（原 [#32601](https://github.com/sgl-project/sglang/pull/32601)） | `--quantization mxfp_w4a8` 的 experts 分支，待 A5 验证闭环 | 是（`ModelSlimW4A8MXFP4MoE`） |
| MoE W4A4 MXFP4 | 离线已合并 [#30319](https://github.com/sgl-project/sglang/pull/30319)；在线入口 [#32602](https://github.com/sgl-project/sglang/pull/32602) | `--quantization mxfp4` 的 experts 分支，待 A5 验证闭环 | 是（`ModelSlimW4A4MXFP4MoE`） |
| Qwen3.5 Gemma RMSNorm on 950 | SGLang [#32745](https://github.com/sgl-project/sglang/pull/32745) + kernel [#638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | 构建期 wheel provider | 同一稳定 API |

“已实现”不等于当前分支已完成硬件验收；以 PR 正文的最新 Validation/TODO 和真实 A5/A2/A3 日志为准。

**功能复测与精度评测分开记录**：Dense 在线/离线请求的用户真机反馈见 [G2.1 验收清单](mxfp-refactoring-plan.md#refactoring-status)，不等于 MoE 或整模型精度评测通过。历史记录有 2026-07-28 的 MoE TP=4 GSM8K：BF16 `93.33` / 离线 W4A8 `93.71` / 在线 `mxfp_w4a8` `65.58`；旧数字不能继承为当前 head 的结果。同月「让非 expert 的全部 `LinearBase` 回退 BF16」的实验已撤销，后续精度隔离应使用 `ignored_layers`，而非修改 dispatch。当前 PR 状态只维护在 [PR 跟踪表](mxfp-refactoring-plan.md#pr-merge-status)。

MoE 的离线 MXFP4 路径已由上游 #30318 / #30319 落地，#43739（原 #32601）与 #32602 跟踪在线量化入口；相关去重与注册表陷阱见 [known-pitfalls.md](known-pitfalls.md)。

## SGLang 关键路径

以下路径都相对于具体 `sglang/<worktree>/python/sglang/`。

### Diffusion

- `multimodal_gen/runtime/layers/quantization/`
  - `mxfp8.py`：在线 MXFP8 配置，选择 SRT 公共 `NPUMXFP8LinearMethod`；`mxfp4_npu.py`：在线 MXFP4
  - `modelslim.py`：离线 scheme 分发
  - `modelslim_mxfp8_scheme.py`、`modelslim_mxfp4_scheme.py`：离线实现

G2.2 分支的 MXFP8 离线 Scheme 保留 checkpoint 参数/loader 和必要的 scale 适配，post-load/apply 委托 SRT `NPUMXFP8LinearMethod`。G2.3 的局部能力描述位于 `srt/hardware_backend/npu/quantization/mxfp_capabilities.py`，两分支独立；实现与验证进度见 [G2 跟踪](mxfp-refactoring-plan.md#refactoring-status)。

### LLM Dense / ModelSlim

- `srt/hardware_backend/npu/quantization/linear_method_npu.py`：NPU Dense quantized linear 方法
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
- 当前在线 W4A8 配置按 `K % 32 == 0` 选择量化路径；不对齐 Linear（例如视觉 FC2 的 `K=4304`）保留非量化精度。该保护来自本 fork 的历史算子报错，不据此断言所有当前 torch_npu/CANN 版本都不支持 partial block，详见 [对齐保护说明](known-pitfalls.md#w4a8-alignment-fallback)。
- Dense ModelSlim 的 gate/up 异构精度需对应 loader/结构适配，不能为了融合而修改描述 JSON 或删除一致性 guard。它与 MoE experts 的分片契约分别处理，详见 [混精加载说明](known-pitfalls.md#modelslim-mixed-gate-up)。
- 离线 MXFP8 要保留最后一个 partial 32 元素 block：placeholder 用 `ceil(K/32)`，奇数 scale 数在 pair reshape 前补齐。
- MoE MXFP 的 dtype、group-list 语义、transpose/NZ 顺序和 dispatcher 限制见 [known-pitfalls.md](known-pitfalls.md)。
- A5 上运行本仓 LLM 脚本时保留 `ASCEND_USE_FIA=1`；它处理的是 attention backend 兼容性，不是量化精度开关。

## 启动与验证入口

- `llm/qwen3.5_dense_*.sh`：Dense BF16 / W8A8 / W4A8；部分入口沿用文件名、默认加载 Qwen3.6-27B，具体路径见 [启动说明](../llm/README.md#dense-w4a8-在线--离线)
- `llm/qwen3.5_moe_*.sh`：Qwen3.5 MoE BF16 / W8A8 / W4A8 / W4A4
- `llm/qwen3_*`：Qwen3 对照脚本
- [`diffusion/i2v_offline_mxfp8.sh`](../diffusion/i2v_offline_mxfp8.sh)：Wan2.2 I2V-A14B 离线 MXFP8，直接调用当前环境的 `sglang generate`；使用已有 repacked 权重与 `gyro.jpg`，输出留在当前目录的 `outputs_i2v_mxfp8/`。分支与下一步见 [G2.3 接手入口](mxfp-refactoring-plan.md#g23-handoff)。
- `diffusion/` 其余脚本：Wan2.2 TI2V/T2V 在线、离线量化入口与 PR 记录

结论必须区分：静态检查、CPU 回归、单 kernel probe、模型加载/warmup、确定性请求、精度评测和性能评测。没有对应硬件日志时，不把静态检查写成 e2e 通过。
