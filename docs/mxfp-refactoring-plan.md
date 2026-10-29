# Ascend MXFP 目标与验收

更新：2026-10-29。本文件只维护目标、PR 状态、关键证据与未测范围；分支/HEAD 在 [branches.md](branches.md)，格式契约在 [量化概览](quantization-overview.md)，故障机制在 [已知陷阱](known-pitfalls.md)。历史过程由 Git 追溯，不另存流水表。

范围：SGLang Ascend 的 SRT Dense/MoE 与 Diffusion，在线 BF16/FP16 量化及离线 ModelSlim。相同执行契约复用 kernel；加载、量化准备与推理分开。Dense 双级 FP4 与 MoE/ModelSlim 单级格式不强行合并，模型名不成为复制量化实现的理由。

<a id="refactoring-status"></a>

## 目标

| 目标 | 当前结论 | 剩余工作 |
| --- | --- | --- |
| G1：格式与边界正确性 | MXFP8 partial-scale / TP 对齐已随 #32745 合入；在线 MoE W4A8 在 #43739，W4A4 在独立 #32602 | Diffusion partial-scale 与各 TP/EP、dtype、backend 分别验证；不从单配置推断全覆盖 |
| G2.1：Dense W4A8 公共执行 | #44363；ModelSlim gate/up 混精另列 #44364。原验收版本在线/离线功能通过 | 新发布 head 的真机结果与原验收版本分开记录；数值等价未测 |
| G2.2：Diffusion MXFP8 公共执行 | #44365；原验收版本 I2V 输出正常 | 当前 PR 的 CI 失败归因与 review；新 head 真机未验收 |
| G2.3：局部能力描述和提前拒绝 | #44366，依赖 #44365；原版本 I2V 功能通过 | Draft；17 条描述不等于全部硬件组合通过 |
| G3：权重生命周期 | #43739 已实现在线 W4A8 完成度跟踪、懒分配、完成即量化、源存储释放与注册 loader 重载 | 评审要求的 NPU 源缓冲释放/启动峰值证据仍缺；不先搭建通用框架 |
| G4：代际差异归属 | Gemma 构建期 provider（kernel #638）与 SGLang #32745 已合入 | 部署核对 target wheel；仅跨组件表示契约按需进入 DeviceOperator |
| G5：硬件与精度证据 | 已有下述用户功能反馈、GSM8K 对照和标准测试结果 | 静态、CPU/mock、NPU kernel、模型功能、精度、性能分别记录 |
| G6：W4A8 serving 性能 | **归档、暂停；未来独立 PR，不阻塞 #43739 功能评审** | 用户恢复任务后再开展；没有稳定端到端收益，不继续扫参或要求重复整机测试 |

<a id="pr-merge-status"></a>

## 活跃 PR

2026-10-29 实时核对：以下 9 个 PR 均 Open、base 为 main、需要 review；#43740/#44366 为 Draft。GitHub 合并检查不是验收结论，操作前仍需刷新。

| PR | 单一职责 | 当前阻塞 / 下一步 |
| --- | --- | --- |
| [#43739](https://github.com/sgl-project/sglang/pull/43739) | 在线 MoE W4A8、加载生命周期与权重策略 | 可合并但受门禁/review 阻塞；权重策略线程已 resolve，加载线程等待 NPU 内存证据 |
| [#32602](https://github.com/sgl-project/sglang/pull/32602) | 独立在线 MoE W4A4 / 融合 GMM | 非 Draft，32 项隔离 CPU/mock 通过；缺 run-ci，当前 head 真机待验收 |
| [#43740](https://github.com/sgl-project/sglang/pull/43740) | Ascend 950 FIA mixed split | Draft；77 项 attention、10 项 batch/TBO 隔离回归通过，缺当前 head 真机结果 |
| [#40419](https://github.com/sgl-project/sglang/pull/40419) | speculative SSM/intermediate state BF16 | kernel #808 已合入；缺 run-ci，待 review |
| [#44363](https://github.com/sgl-project/sglang/pull/44363) | Dense W4A8 公共执行 | 6 项 CPU/mock 通过；缺 run-ci，待 review |
| [#44364](https://github.com/sgl-project/sglang/pull/44364) | Qwen3.5 Dense ModelSlim gate/up 混精 | 缺 run-ci；本次 mergeability 查询为 UNKNOWN，不写成无冲突 |
| [#44365](https://github.com/sgl-project/sglang/pull/44365) | Diffusion MXFP8 委托公共 kernel | 有 run-ci，存在实际 CI 失败；本地落后远端 1 个维护者提交，未自动同步 |
| [#44366](https://github.com/sgl-project/sglang/pull/44366) | MXFP 能力描述与 preflight | Draft、缺 run-ci；15 项能力及 3 项 adapter CPU/mock 通过，等待 #44365 |
| [#45550](https://github.com/sgl-project/sglang/pull/45550) | Diffusion MXFP8 自动选择 FA | 10 项隔离 CPU 回归通过；缺 run-ci，自动 FA + I2V 真机待验收；本次 mergeability 为 UNKNOWN |

**#43739 的 CI**：本次核对当前 head，标题与 lint 通过；17 项失败中 11 项为缺 `run-ci` 的 gate，6 项 finish 为级联失败，CPU/NPU suite 未运行。可复核 [NPU run 40848953069](https://github.com/sgl-project/sglang/actions/runs/40848953069)、[Base run 40848952910](https://github.com/sgl-project/sglang/actions/runs/40848952910)。不是已执行测试报错，也不能据此宣称硬件通过。

**#44365 的 CI**：10-22 已查日志包含 NPU z-image TextEncodingStage 超阈值、CPU 缺 `sgl_kernel.shm_group_initialize`、AMD 子测试无 GPU，以及 offload/encoder 测试失败。本次仍有失败；未做全量归因或同 main 对照，不能统称环境问题或本 PR 回归。

<a id="merge-dependencies"></a>

### 依赖

- 唯一未合入的线性依赖：**#44365 → #44366**。其余 8 个入口可独立审阅；W4A4 不再等待 W4A8。
- 已合入前置：SGLang #32745，kernel #638/#808/#824。源码合入不代表部署 wheel 已包含修复。
- #45550 延续已关闭 #32041 的自动路由增量；核心 MXFP8 attention 已由 #39438 合入，不重复实现。
- #36426 只放独立组合验证分支，不混入正式功能栈。基础分支更新时同轮同步组合分支。
- 后续更新用普通 merge/追加提交；先查实际 diff，去掉已合入重复代码。新 PR 只在用户点名后创建。

## 验收证据与边界

### 在线 W4A8（G1/G3/G5）

- 10-29 按授权回退至 `3cdd018` 并合并上游，性能实验移出；随后精简到 12 个文件。新增测试合为量化契约、加载生命周期两份，post-load/EPLB 回归留在已有文件。正式与组合分支各 **42 项隔离 CPU 通过、7 项 CUDA 跳过**；6 类故障注入覆盖策略、bias、源释放、重复 shard、提前量化和离线二次量化。
- 用户同日回传标准 pytest：**42 passed、7 skipped、17 warnings，14.90 s**（Python 3.11.10 / pytest 8.3.2）。完整导入与收集成功，CUDA 跳过和 pytest-asyncio/JIT/SWIG 告警未导致失败；未附执行 SHA。这是标准入口 CPU/mock 证据，不是实际 NPU 算子或启动峰值验收。
- 用户报告 **BF16 / 在线 W4A8 GSM8K 均 96.13**。未附逐次 server SHA、原始报告及 shot/prompt/采样配置，不能继承为回退合并后新 head 的精度结果。
- 权重采用 `round_mode="rint", scale_alg=2, dst_type_max=7.25`，不改激活策略、离线 packed 权重或 W4A4。此前用户 Ascend 950 权重 probe 的 40/40 样本重建误差改善、汇总 NMSE 相对下降约 5.69%；这不是整模型准确率提升。
- [权重策略评审](https://github.com/sgl-project/sglang/pull/43739#discussion_r4410300363)已回复并 resolve；[加载评审](https://github.com/sgl-project/sglang/pull/43739#discussion_r4410300039)代码已处理，**NPU 源存储释放与启动峰值测量仍未完成**，保持 unresolved。

### 其他已报告功能

| 场景 | 证据 | 不包含 |
| --- | --- | --- |
| Qwen3.5 Dense 离线 W8A8 TP2 | 10-16 用户运行普通脚本正常，主观输出质量正常；未附 SHA | TP4/all、数值等价、性能 |
| G2.1 在线/离线 Dense W4A8 | 10-14 用户确认启动/请求正常；原共享实现 `24e100f1e8`，离线载体 `2ef4c276be`；在线 27 个视觉 fc2 的 K=4304 保留 BF16 回退 | 全视觉 W4A8、新发布 head、数值等价与性能 |
| G2.2 Wan2.2 I2V 离线 MXFP8 | 10-14 用户反馈成功生成且观感正常；原实现 `6eb8e1dd1c`；已有日志为 1280×704、SDPA、CPU offload | 后续尺寸、MXFP8 attention、新发布 head、独立 warmup/数值/性能 |
| G2.3 局部能力试点 | 10-14 用户反馈同一 I2V 场景输出正常；原实现 `d6a46f90d9` | 全部 17 条能力、MoE/TP/EP/非法配置分支、新组合 head |

原 G2 版本已转为远端归档 tag，见 [分支表](branches.md#g2-runtime-record)。已跳过的数值对照不再作为部署前置。曾出现的 graph/乱码问题在停止并重启服务后正常，根因未证实，不保留禁 graph 作为通用处方。

<a id="g23-handoff"></a>

### 现成入口

- LLM 命令在 `llm/` 执行：`bash ./qwen3.5_dense_online_w4a8.sh 0` 或 `bash ./qwen3.5_dense_offline_w4a8.sh 0`。离线默认权重 `/mnt/share/weight/Qwen3.6-27B_mxfp-w4a8-0821`；在线默认模型不同，对照须选同模型版本。
- Diffusion 在 `diffusion/` 执行：`bash ./i2v_offline_mxfp8.sh 0`。现成权重 `/mnt/share/weight/Wan2.2-I2V-A14B-Diffusers-MXFP8-G2`，无需重新下载/repack。
- 当前 I2V 脚本面向 #45550 自动 FA 验证，设置 MXFP8 attention 开关，不强制 backend；请求 1024×720、81 帧、40 步。必须看到 `Using MXFP8 quantized Ascend Flash Attention.` 并生成视频；fallback 成功不能算自动 MXFP8 FA 通过，旧 G2 结果不覆盖此配置。
- 服务器分支/安装由用户手动选择，脚本不切分支或改 PYTHONPATH。普通运行不追加性能或数值验收框架。

<a id="g6-performance-archive"></a>

## G6：性能研究归档

**2026-10-29 按用户要求暂停，后续独立 PR。** 当前 #43739 不含 TP 路由融合/阈值、GDN 输入复用或相应实验环境变量。旧工具已退出主仓，恢复点见 [归档版本](branches.md#performance-archive)；不要拿当前正式分支直接运行旧入口。

保留影响下一步判断的结论：

1. 缩 Mamba 池 `1582 → 872`、固定 KV `2828544` 后，ShareGPT 吞吐 `645.69 → 636.55 tok/s`，未显示端到端收益。不再推荐扫池容量；单轮也不证明显著回退。
2. 旧 Ascend950DT_9574 / CANN 9.1 / torch_npu 2.10.0.post4 的单 rank eager 合成 MoE，在 M=2048/7240 偏向拆分路由；新 9582 / CANN 9.2 / post7 则融合比拆分快约 **7–14%**。不能把旧阈值 2048 当通用优化，也不能单独归因于某个组件升级。
3. 新栈原始 token 先量化再路由候选只在大 M 比融合快约 **1–2%**，不足以接入服务。融合/拆分合成输出 relative L2 约 **0.0364–0.0367**（旧栈为 0）；预量化/拆分为 0。这不是模型准确率或 graph 等价结论。
4. 换机后用户报告明显卡顿消失；预热后 BF16 **1020.71**、阈值版 W4A8 **974.76**、W4A8 复测 **966.85 tok/s**。模型 token 总量相同，但完整服务 SHA/缓存对等性未独立证实；不能据此归因阈值，也没有稳定 serving 收益。`Critical` 在用户环境属正常现象，不作为测试门禁或停顿根因。
5. 旧 profile 的矩阵计算节省被路由/状态清零成本抵消；跨 rank/重叠算子累计时间不是 wall time。ITL 可能按 token 平摊，不是独立停顿次数；缺时间线无法证明共同 stall。

恢复时先固定同机同栈、实际服务 SHA、权重、TP、池容量和缓存/预热策略，再定位完整关键路径。候选需 serving 吞吐/延迟与精度/内存边界共同支持；无净收益就停止。当前不再安排新的 benchmark 或移植候选。

## 暂缓与维护

- Qwen-VL 预处理整理仍暂缓；已合入 #43309 的 resize 修复不重复提交。
- 更新只保留日期、来源/版本、结论、未测范围、下一步。原始日志用于当次诊断，长期结论不依赖可清理的 `.pi/` 路径。
- 接手先读 [AGENTS](../AGENTS.md)、本页和 [branches.md](branches.md)，不要从历史命令自动恢复实验。
