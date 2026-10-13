# Ascend MXFP 重构目标与执行计划

更新日期：2026-10-12（Asia/Shanghai）。本文是后续会话的任务与验收入口，无需依赖聊天历史。

范围：SGLang Ascend 的 SRT Dense / MoE × W8A8 / W4A8 / W4A4 × 在线 / 离线 ModelSlim，以及 Diffusion MXFP8 的执行复用。CUDA / ROCm 和其他量化格式不在本计划的实现范围内。

- [实现概览与代码入口](quantization-overview.md)
- [分支、提交、PR 与 worktree 记录](branches.md)
- [已知契约与故障模式](known-pitfalls.md)
- [原版交互式 HTML 审阅报告：2026-10-10](reports/mxfp-architecture-roadmap-2026-10-10.html)

HTML 按原文件归档，固定代码基线为 `e35f60b9bcb51045fd0602d8ed12cff8021c040f`。其中的待办、PR 编号、bias 描述与状态是当时的快照，不能替代本文的进度或当前源码。本文从现在起固定使用 **G1–G5** 和 **G2.1–G2.3** 等任务编号；HTML 中的六个执行步骤保留原编号，两者不要混用。

### 维护分工：每种进度只保留一个当前表

| 信息 | 唯一维护入口 | 其他文档如何引用 |
| --- | --- | --- |
| 重构目标的实现、验证与交付进度 | 本文[重构目标跟踪表](#refactoring-status) | 用 G1–G5 / G2.1–G2.3 引用，不另列一份当前完成状态 |
| PR 是否合入、检查状态、合入顺序和依赖 | 本文[PR 合入跟踪表](#pr-merge-status)与[依赖说明](#merge-dependencies) | `branches.md` 只链接这里；旧状态表作为带日期的历史证据保留 |
| branch / worktree / HEAD / remote 与详细操作记录 | [branches.md](branches.md#current-branch-map) | 本文的实现提交是验收证据，不当作另一份滚动 HEAD 表 |
| 稳定的能力与代码入口 | [quantization-overview.md](quantization-overview.md) | PR 合入或行为变化后同步说明，不承载实时 CI 队列 |
| 原始架构审阅 | [2026-10-10 HTML](reports/mxfp-architecture-roadmap-2026-10-10.html) | 固定历史快照，不追改旧 PR 号或旧结论 |

## 1. 总目标与职责边界

最终目标是：相同数学与执行契约只实现一次；在线准备与离线加载各司其职；支持条件可提前检查；新模型能复用已有 Scheme / Kernel；每项支持声明都有相应验证证据。

| 依据 | 负责的问题 | 本计划如何落实 |
| --- | --- | --- |
| [#31783：Quantization 2026 H2](https://github.com/sgl-project/sglang/issues/31783) | 公共量化分层、权重生命周期、在线入口和能力校验 | 收敛执行实现、描述契约、统一验证信息源 |
| [#34559：Ascend MXFP 能力覆盖](https://github.com/sgl-project/sglang/issues/34559) | Ascend MXFP 模型、算子及在线/离线覆盖 | 补齐在线 MoE，并明确各路径的边界和验收证据 |
| [#35709：Ascend 跨代际适配](https://github.com/sgl-project/sglang/issues/35709) | provider 与运行时表示的归属 | 独立算子差异进入 kernel provider，跨组件差异按需进入 DeviceOperator |
| [#31774：能力与兼容校验](https://github.com/sgl-project/sglang/issues/31774) | 支持组合的统一校验方向 | 先完成 Ascend MXFP 局部切片，不另外造一套全局框架 |

Issue 的开关状态、复选框和旧 PR 状态都不能代替代码核对。

### 目标架构

```text
Config：选择格式、来源、层与量化策略
  ↓
Method：组织权重加载、准备与执行生命周期
  ↓
Scheme：格式相关参数、loader metadata、scale / packing 契约
  ↓
共享 Kernel：消费已准备的表示，执行 quant / matmul / GMM

在线：BF16/FP16 checkpoint → 加载 → 在线量化与布局准备 ─┐
                                                     ├→ 兼容的公共执行
离线：预量化 checkpoint → 参数加载与必要布局准备 ──────┘

独立算子内部的代际差异 → sgl-kernel-npu 稳定 API / 构建期 provider
跨组件可见的表示差异   → 按需使用 SGLang DeviceOperator
```

这是职责划分，不要求每条路径机械增加相同数量的类。不同格式只有在数学、dtype、scale 和 layout 契约兼容时才共享执行。

### 必须保留的区分

- **层类型与模型名称不同**：MoE 模型也包含非专家 Linear。这里 Dense 指 Linear 路径，MoE 指专家路径。
- **W4A8 是 FP4 权重、FP8 activation**，不能改成 W4A4 计算后仍称为 W4A8。
- **W4A4 名称不能唯一确定算法**：Dense 在线双级 L0/L1、ModelSlim 单级路径及 MoE 单级专家分别核对，不能强行合并。
- **在线与离线来源不同**：离线 packed `uint8 [N,K/2]` 权重不得再次量化或打包；必要的布局转换与重复打包是两回事。
- **实现、验证、发布是三个维度**：分支有实现、CPU 检查通过、NPU 验证通过、上游合并必须分别记录。
- shape 条件按实际 TP 分片后的维度判断。缺少硬件结果不等于不支持，也不等于已验证支持。

<a id="refactoring-status"></a>

## 2. 重构目标完成情况

“目标 2”在本文中明确指 **G2：公共执行复用与能力元数据**。第一项是 Dense/Linear W4A8 去重，即使它从 MoE W4A8 分支开发，也不是重写 MoE experts 的执行。

| 编号 | 目标 / 任务 | 实现进度（2026-10-12） | 验证与剩余验收 | 交付载体 / 合入跟踪 |
| --- | --- | --- | --- | --- |
| G1.1 | 修正 MXFP8 partial-scale 与布局边界 | 已有对应修复；Diffusion loader 仍需单独核对 | 余块与奇数 scale 的表示、分片后 shape 和目标 NPU 验证；占位修正不等于任意 K 可运行 | #32745；状态见下方 PR 表 |
| G1.2 | 补在线 MoE W4A8 / W4A4 与实际精度分配 | 两条功能分支已有实现，保持各自能力增量 | 专家确实量化；层级精度、融合、dispatch、TP/EP 与在线 RTN 精度需有证据 | #43739（原 #32601）/ #32602；状态见 PR 表 |
| **G2.1** | **Dense W4A8 在线/离线公共执行** | **实现已提交并推送** | 已有 8 个 CPU/mock 测试记录；NPU 数值、真实加载与请求验证待补，尚未整体验收 | `24e100f1e8`；本次查询无 PR，未合入上游 |
| **G2.2** | **Diffusion 离线 MXFP8 委托公共 kernel** | **已建分支；尚无实现增量** | post-load 与执行一起迁移；loader、scale、bias 回归及真实 checkpoint / 图片路径验证 | `junlin/diffusion-mxfp8-shared-execution`；本次查询无 PR |
| **G2.3** | **Ascend MXFP 局部能力描述与提前校验** | **已建分支；尚无实现增量** | 同一信息源驱动校验、明确 fallback、文档及测试矩阵；不能只新增静态表 | `junlin/ascend-mxfp-capabilities`；本次查询无 PR |
| G3 | 权重生命周期与公共在线入口 | 待设计 / 实施，与公共框架工作协调 | 参数、元数据、重载语义明确；入口迁移兼容既有 CLI | 尚未拆出交付 PR |
| G4 | 跨代际差异按契约归属收敛 | Gemma 切片已有实现；整体未完成 | provider 可追溯；构建 target、API 与配套 wheel 验证；只在必要处引入运行时契约层 | kernel #638 + #32745；状态见 PR 表 |
| G5 | 模型、算子、并行与 CI 的硬件验收 | 持续工作 | 本轮没有新增硬件结果；每项支持声明关联代码、软件栈、硬件、配置和日志 | 随各任务记录证据，CI 通过不替代实际硬件覆盖 |

此表区分“实现完成”“验收完成”和“上游合入”。功能 PR 的合入状态只更新第 5 节；某 PR 合入不自动把关联目标标为全部完成。G2.2 / G2.3 本次查到的 worktree 均干净、停留在建分支基线，不能据目录存在判断任务已经实现。

## 3. G2 的具体实施与验收

### G2.1：Dense W4A8 公共执行——实现完成

代码位置（相对于 SGLang worktree）：

- `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`
- `test/registered/unit/layers/quantization/test_npu_mxfp4_w4a8_linear.py`

已经将 `NPUMXFP4W4A8LinearMethod.apply()` 与 `NPUMXFP4W4A8OfflineLinearMethod.apply()` 委托给 `npu_w4a8_mxfp4_linear(x, weight, weight_scale, bias)`。公共函数接收 prepared weight / scale，负责输入 dtype、展平、MXFP8 activation quant、bias、matmul 和输出 shape 恢复，不判断 checkpoint 来源。

| 项目 | 可追溯记录 |
| --- | --- |
| 分支 / worktree | `junlin/w4a8-shared-execution` / `sglang/w4a8_shared_execution/` |
| 开发基线 | #43739 head `ded467157750ccfd20dd2256110fad9ff104dc08` |
| 实现提交 | [`24e100f1e8429bde02f3e81221c4392e9e69934b`](https://github.com/TallMessiWu/sglang/commit/24e100f1e8429bde02f3e81221c4392e9e69934b) |
| 本次核实 | 本地与远端 SHA 相同；worktree 干净；未找到该分支对应 PR |
| 已有验证记录 | 8 个隔离 CPU/mock 测试，含 36 组输入契约；静态及 lint 记录见 [branches.md](branches.md)；本次仅核对记录与提交，未重跑测试 |

- [x] 在线与离线只保留一份公共执行实现。
- [x] 保留在线量化、离线加载和各自 post-load；MoE / W4A4 不在该改动中。
- [x] 覆盖二维/多维、非连续输入、dtype、动态 bias 与离线不重复 quant/pack。
- [ ] 目标 NPU 下比较相同 prepared tensors 的两入口与重构前基线。
- [ ] 在线/离线真实加载、warmup 和代表性文本/图片路径验证。
- [ ] 按后续发布安排单独审阅和合入；当前原 #43739 分支未包含这笔新增提交。

当前 W4A8 bias 契约为 BF16，向量扩为 `[1,N]`。HTML 中较早的 FP32 bias 描述不用于本任务；也不能把 W4A8 的 bias 规则推广到 MXFP8。

2026-10-13 精度反馈：用户确认使用 mixed gate/up 分支并设置 BF16 SSM state；图片请求长输出乱码，纯文本拿破仑介绍也出现错误事实、重复与乱码，短请求 `1+1=2` 不能视为精度通过。新增 `llm/diagnostics/modelslim_w4a8_check.py`，用实际离线 post-load/apply 对比软件 FP4/UE8M0 解包结果，覆盖合成权重与 checkpoint 第 0 层 up/down 的部分输出行，以及 1/64 token、重复调用；参考使用同一量化输入，单独报告 activation quantization 误差。CPU 自检、实际 ModelSlim pack 函数往返、语法与 lint 通过；尚无 NPU 探针结果，根因未定位，下一步收集真实 Linear 误差后再区分模型加载、graph/state 与 checkpoint 量化质量。新环境的 resize 启动失败修复已发布到同一 mixed gate/up 分支，分支定位见 `branches.md`。

### G2.2：Diffusion 离线 MXFP8 执行复用——已建分支，待实施

已登记 `junlin/diffusion-mxfp8-shared-execution`，worktree 为 `sglang/diffusion_mxfp8_shared_execution/`；当前定位见 [branches.md](branches.md#current-branch-map)。后续会话先核对该目录的用户改动，再继续实施。

重点入口：

- `python/sglang/multimodal_gen/runtime/layers/quantization/mxfp8.py`：在线路径已选择 SRT `NPUMXFP8LinearMethod`。
- `python/sglang/multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py`：离线路径仍保留自己的 post-load、activation quant、matmul 和 shape 恢复。
- `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`：核对共享 MXFP8 kernel 的准备与执行契约。

本次源码核对基于 #32745 的 `cf48b477db`；实施时重新检查最新目标基线，已被主线完成的部分不重复修改。

**实施边界**：Scheme 保留 checkpoint 参数创建、物理 shape/dtype、loader metadata；能统一的后处理与计算委托现有公共 kernel。不同之处放在必要的格式适配位置，不复制新的 quant / matmul，也不让共享执行猜测 checkpoint 来源。

执行顺序与验收：

- [ ] 对照在线、离线、SRT 三条链路，列出输入/输出 dtype、逻辑 shape、物理 layout、stride、bias 与 scale 差异。
- [ ] 明确 checkpoint 的 `weight_scale` 与 SRT `weight_scale_inv` 的映射时机；保留加载键和元数据，不能仅按名字推断数学含义。
- [ ] 核对 G1.1：余块 scale 用逻辑 block 数，奇数 block 数按实际 kernel 契约配对/补齐；不凭补齐推断 kernel 接受任意 K。
- [ ] 同时迁移必要的 post-load 和 apply，防止重复 transpose、重复准备或沿用错误 stride；固定/动态 bias 行为保持正确。
- [ ] 用已有公共实现消除重复 activation quant、matmul 和 shape 恢复；确需扩展接口时检查现有调用方。
- [ ] 完成针对性 CPU/mock 回归，覆盖二维/多维、dtype、bias、scale 边界和 loader 参数映射。
- [ ] 真实 ModelSlim checkpoint 加载 → warmup → 图片请求；补目标硬件数值与性能证据。

依赖是 **共享 kernel 和 loader 的契约正确**，不是机械要求整个 #32745 或 G2.1 必须先合并。按最小必要提交组织独立 diff。

### G2.3：Ascend MXFP 局部能力表——已建分支，待实施

已登记 `junlin/ascend-mxfp-capabilities`，worktree 为 `sglang/ascend_mxfp_capabilities/`；当前定位见 [branches.md](branches.md#current-branch-map)。尚未登记实现或验收证据。

先检查 `python/sglang/kernels/spec.py` 中 `KernelSpec` / `FormatSignature` 与现有量化校验。平台 capability 和 dtype 描述已有基础，但不能视为 checkpoint、来源、层类型、scale、TP/EP 的完整运行时验证。

初版覆盖 SRT 的 12 个组合：`Dense/MoE × W8A8/W4A8/W4A4 × online/ModelSlim`。对每项记录已实现、明确不支持或待核实及原因，另列 Diffusion 已核实的 MXFP8 路径。并非要求把所有组合实现出来。

| 维度 | 最小需要说明的内容 |
| --- | --- |
| 来源与格式 | 在线 / ModelSlim、checkpoint 物理 dtype、packing、必要的格式版本 |
| 层与 runtime | Dense / MoE、SRT / Diffusion；模型特有约束单独标明 |
| 数学 | W/A dtype、block/group size、单级/双级 scale、scale 编码 |
| 表示 | prepared layout、stride、shape 与 TP 后分片约束 |
| 执行环境 | 平台能力、kernel/API、backend、fusion/clamp 条件 |
| 并行通信 | TP/EP、DeepEP mode、normal/low-latency 的实际表示 |
| 回退与证据 | 既有 fallback 的触发条件/实际精度；独立记录验证级别和日志 |

任务清单：

- [ ] 找出现有信息源与校验入口；避免再造重复的全局 registry，也不把所有模型策略塞进 KernelSpec。
- [ ] 用最小机器可读定义表达真实条件，可引用底层 kernel 能力；未知验证状态单独保存。
- [ ] 接入真实路径：配置齐备时检查可静态判断的组合；模型/分片信息齐备时检查 shape；必须依赖权重的数据留在明确的 post-load 校验点。
- [ ] 错误消息指出请求组合、不满足的条件和已有可用方案，避免到 matmul/reshape 时才暴露含糊错误。
- [ ] 表达已有 fallback，输出实际选择的精度/后端；不新增静默降精度、静默 BF16 回退或离线 packed 权重的无效回退。
- [ ] 支持文档与参数化测试由同一信息源生成或校验，覆盖允许、拒绝、fallback 和待硬件验证的区别。
- [ ] 核对原有支持路径不被误拒绝；按宣称支持的 TP/EP 与后端补硬件 probe。

验收不是“新增了一张表”，而是存在可追踪的校验调用方、文档使用方和测试使用方。G2.2 与 G2.3 可分别实施，能力表也不必等待全部格式实现完毕。

## 4. 其他工作流与完成条件

### G1：正确性、在线能力和实际精度

**G1.1 scale 边界**：例如 K=4304 时 `floor(K/32)=134`、`ceil(K/32)=135`，按两个 scale 配对的容量为 136。加载逻辑不能把余块丢掉，也不能把奇数 block 数直接 reshape 成 pairs。实际支持还取决于分片、权重/activation padding 和算子契约，不能只修占位张量就宣称端到端完成。

**G1.2 在线 MoE**：复用已有 W4 GMM 和 Ascend runner，在线补权重准备与分发入口；保留 main 的 scale helper、融合/clamp 策略，以及 W4A8 DeepEP normal=BF16 / low-latency=MXFP8。明确 experts 与非专家 Linear 的实际精度，W4A4 单级与双级不能互换。硬件精度问题与架构去重分别归因。

验收：正确加载与分发、每层实际精度可解释、代表性 TP/EP/dispatch 正确、在线 RTN 精度有证据；CPU shape 检查不作为算法精度结论。

### G3：权重生命周期与入口

后续与公共 Weight Loader / post-load 工作协调，明确 `创建 → 加载 → 准备 → 执行 → 重载/更新` 每一步的权重、scale、bias、参数属性与所有者。区分允许重复准备、需要重建和应明确拒绝的状态，避免双重 transpose 或旧 cache。

公共在线入口迁移应兼容现有 `--quantization mxfp8 / mxfp_w4a8 / mxfp4`，由 Config 决定来源、层选择和量化策略。该工作不作为 G2 两处执行去重的隐式前置，不在小重构中顺带改 CLI。

### G4：跨代际职责

先登记每个代际判断改变了什么：若只影响独立算子内部，实现稳定公共 I/O 并通过 `sgl-kernel-npu` 构建期 provider 选择；若影响 producer、buffer 和多个消费者共享的 dtype/layout/metadata，才考虑 `DeviceOperator`。量化是否启用及采用哪种方案仍归量化配置层。

Gemma RMSNorm 是已有切片。继续逐个核对 MXFP 的代际判断，不强制每个量化方法增加 DeviceOperator，不把缓存 SoC 判断等同于完成分层。

### G5：验收证据

顺序：静态/CPU 契约 → NPU kernel 数值 → checkpoint 加载 → warmup → 文本/图片请求 → TP/EP/通信 → 精度 → 性能 → 代表性 CI。

每份证据至少记录：代码 SHA、kernel wheel/构建 target、CANN/torch/torch_npu 版本、硬件、模型/checkpoint、量化与 TP/EP/后端参数、测试命令、数据集和日志位置。只声明实际覆盖的组合；旧分支跑过的数字不能直接继承为新 head 的验收。

## 5. PR 合入计划与重构推进顺序

<a id="pr-merge-status"></a>

### 5.1 PR 合入跟踪表

**2026-10-12 通过 GitHub CLI 核实。** 保留用户原计划的 1–8 序号，后补的 kernel #824 列为第 9 项；序号便于对应原计划，实际前置按第 5.2 节执行。当前待合入的是 7 个个人 PR：全部 Open、`base=main`、`MERGEABLE`，均仍需 review；其中 3 个保持 Draft。`MERGEABLE` 只表示没有 Git 合并冲突，不表示 CI、硬件验收或合入条件已全部满足。

| 原序号 | PR | 交付目标 / 对应重构目标 | 合入情况 | 当前 head 检查聚合 | 前置与下一步 |
| --- | --- | --- | --- | --- | --- |
| 1 | [kernel #638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | Gemma RMSNorm 稳定 API，构建期按 SoC 选 provider；G4 | **Open，非 Draft** | SUCCESS；需 review | 无 PR 前置；作为 #32745 的 API / wheel 前置，完成 provider staging 与 target wheel 验证 |
| 2 | [SGLang #32745](https://github.com/sgl-project/sglang/pull/32745) | 接入 Gemma API，修正 MXFP8 partial-scale；G1.1 / G4 | **Open，非 Draft** | FAILURE；需 review | kernel #638 先交付配套 wheel；本 PR 是 W4A8 与 FIA 分支共享修复来源；#36426 用于组合验证 |
| 3 | [SGLang #32041](https://github.com/sgl-project/sglang/pull/32041) | Wan2.2 Diffusion FA MXFP8；能力补齐背景，不等于 G2.2 去重 | **Closed，未合并；移出待合入队列** | 不再作为待合入检查项 | 核心 Attention 能力由 #39438 合入；若有独有剩余功能，先与作者确认，再单独评估 |
| 4 | [SGLang #40419](https://github.com/sgl-project/sglang/pull/40419) | 投机 SSM 与相关 intermediate state 使用一致 BF16；MTP 正确性配套 | **Open，非 Draft** | FAILURE；需 review | 可与 Gemma / MXFP 栈并行；结合已合入的 kernel #808 检查 dtype 与 layout 契约 |
| 5 | [kernel #808](https://github.com/sgl-project/sgl-kernel-npu/pull/808) | Ascend 950 recurrent GDN 与 BF16 state 检查；MTP 配套 | **Merged，2026-09-29；移出待合入队列** | 合并前 SUCCESS | 不重开、不继续在原 PR 追加；后续推进 #40419，并核对 #742 / #651 与 layout 配套 |
| 6 | [SGLang #43739](https://github.com/sgl-project/sglang/pull/43739)（原 [#32601](https://github.com/sgl-project/sglang/pull/32601)） | 在线 MoE W4A8 + 融合 GMM；G1.2 | **Open，Draft** | FAILURE；需 review | 先同步 main 和更新后的 #32745；完成后才更新 #32602；可与 MTP / cache 路径并行 |
| 7 | [SGLang #32602](https://github.com/sgl-project/sglang/pull/32602) | 在线 MoE W4A4 + 融合 GMM1-SwiGLU；G1.2 | **Open，Draft** | FAILURE；需 review | 承接更新后的 #43739；保留 experts 单级与 Dense 双级 scale 的区别 |
| 8 | [SGLang #43740](https://github.com/sgl-project/sglang/pull/43740)（原 [#34387](https://github.com/sgl-project/sglang/pull/34387)） | Ascend 950 mixed chunked-prefill 的 FIA prefill / decode 拆分；联调配套 | **Open，Draft** | FAILURE；需 review | 先同步 main 和更新后的 #32745；之后可与 W4 MoE 栈并行，不是 G2 的默认前置 |
| 9（后补） | [kernel #824](https://github.com/sgl-project/sgl-kernel-npu/pull/824) | cache location view 依据实际 tensor 长度，避免越界；正确性配套 | **Open，非 Draft** | SUCCESS；需 review | 独立同步 kernel main，不需混入 #638；补齐边界回归与目标 NPU 验证 |

两条旧 PR #32601 / #34387 仍为 Closed；上表用新编号跟踪后续交付，旧号只作追溯。重新建 PR 沿用了原分支，因此无需为换号再创建分支。

七个待合入 PR 的本地 HEAD 与 GitHub head 在本轮核对一致，SHA 统一记录在 [branches.md 当前定位表](branches.md#current-branch-map)。五个 SGLang PR 的 FAILURE 包含 gate / finish 检查失败；本轮只记录检查聚合，未逐项重新归因，不能直接判定为代码回归，也不能概括成“只缺 review”。缺少最新 head 的 NPU 证据时，仍须按对应验收项补验，不自动转 Ready 或执行上游合并。

### 配套、验证与替代 PR

这些 PR 不计入上面的 7 个个人待合入 PR，但会影响组合验证和交付判断。

| PR | 本轮状态 | 对原计划的作用 / 后续核对 |
| --- | --- | --- |
| [SGLang #36426](https://github.com/sgl-project/sglang/pull/36426) | Open，非 Draft；mergeable=UNKNOWN；检查 FAILURE | 视觉塔 ModelSlim config 的组合验证依赖；#32745 的 scale 边界需要相应加载链路才能触发。已有同文件改动，不再称为文件完全独立；不混入正式功能 PR 栈 |
| [kernel #742](https://github.com/sgl-project/sgl-kernel-npu/pull/742) | Open，Draft；mergeable=UNKNOWN；未返回检查聚合 | 历史 MTP 构建组合使用此分支并包含 #651；联调前重验当前 head 的包含关系和 state 迁移修复，不能仅凭旧构建记录判断 |
| [kernel #651](https://github.com/sgl-project/sgl-kernel-npu/pull/651) | Open，非 Draft；mergeable=UNKNOWN；检查 FAILURE | 核对 `move_intermediate_cache` 的 `h_block_size 2→1` 及相关 state 迁移契约；这是历史 950 MTP 的运行前提之一，不是已合入 #808 的待合入前置 |
| [kernel #747](https://github.com/sgl-project/sgl-kernel-npu/pull/747) | **Merged，2026-09-17** | 统一 SSM state layout；配套 wheel 必须包含其效果，与 SGLang #39589 配对验证 |
| [SGLang #39589](https://github.com/sgl-project/sglang/pull/39589) | **Merged，2026-09-18** | SGLang 侧对应 layout；检查 prefill / decode / verify 的表示一致性 |
| [SGLang #39438](https://github.com/sgl-project/sglang/pull/39438) | **Merged，2026-09-17** | 已提供 Wan2.2 MXFP8 Attention，是 #32041 关闭的替代实现背景；不代表 Diffusion 离线 Linear 的 G2.2 已完成 |

`UNKNOWN` 或没有检查聚合表示本次没有得到确定结果，不能填写为“无冲突”或“CI 通过”。表中合并日期按 Asia/Shanghai 记录。

<a id="merge-dependencies"></a>

### 5.2 合入顺序、分支更新顺序与并行范围

```text
kernel #638
    └─ API / 配套 wheel → SGLang #32745
                            ├─ 共享修复 → #43739（原 #32601）→ #32602
                            └─ 共享修复 → #43740（原 #34387）

SGLang #40419：独立推进，与已合入 kernel #808 的 BF16 契约配套
kernel #824：独立推进，无须叠入 #638
SGLang #36426：视觉塔组合验证，不混入上述正式 PR 栈
kernel #742 / #651：MTP 组合运行条件另行核对
kernel #747 + SGLang #39589：已合入的 layout 配套，检查实际部署版本
```

这里“可并行”指不受另一条链的前置阻塞，不保证后续版本始终零文件重叠。#43739 / #32602 之间有实际提交依赖，不能沿用最初“二者均独立”的描述。

1. **Gemma 根修复：#638 → #32745。** 更新 kernel main 后验证构建期 provider staging 与稳定 API；#734 的重叠实现已纳入历史分支，不重复叠入。#32745 再同步 SGLang main，保留 Gemma API 接入和 partial-scale 修复，适配当前测试目录及 CI 收集入口。910 有 ImportError 回退，950 需要含 #638 的配套 wheel；跨仓依赖不靠混入提交解决。
2. **独立正确性工作：#40419 与 #824。** #40419 保留 main 已有的 AscendC GDN prefill 路径，不恢复已被覆盖的 gating / prefill 补丁；在当前 pool 结构核对 speculative SSM 和 intermediate state 的 BF16 分配、非 NPU 行为及 layout。#824 保留实际 tensor 长度修复，覆盖 `step < MAX_STEP`、分配块末端及两种索引 dtype。两项不必等待第 1 条完成。
3. **MoE 功能栈：#43739 → #32602。** #43739 先同步 main，再合入更新后的 #32745，保留 main 的 scale helper、融合 / clamp 条件，以及 DeepEP normal=BF16、low-latency=MXFP8。完成后 #32602 再同步 main 与更新后的 #43739，处理自身增量；保留 experts 单级 W4A4 / Dense 双级路径，校正实际精度分配的注释。此栈的能力补齐与 G2.1 的公共 Linear 去重分别交付。
4. **FIA 分支：#43740。** 跟随更新后的 #32745，适配当前 attention、调度与图执行接口，验证 mixed-prefill 分拆；迁移旧测试路径，清理重复 Gemma 增量。其前置满足后可与第 3 条并行。
5. **从待合入队列移除 #808 / #32041。** 原计划要求 #40419 先于 #808，以避免 BF16 断言让旧 FP32 pool 直接失败；现在 #808 已合入，应优先补齐 SGLang 侧契约，不能仍计划“等待 #808”。#32041 仅在确认独有剩余功能后重新评估；若继续移植，检查旧 `docs_new/` 到当前 `docs/docs/` 的路径变化。

**后续更新规则**：操作前刷新 main、PR head、worktree 与用户改动。全部正式 PR 继续 `base=main`；若 B 依赖 A，先完成 A 与 main 的更新，再将最新 main 与 A 合入 B，解决两类冲突并重新验证。使用普通 merge、追加提交和普通 push，不改写已发布历史。前置合入 main 后核对内容、同步 main 并清理重复增量。每个目标在自己的 worktree 操作，保留测试组合分支、主仓脏文件与 kernel gitlink。

本轮仅维护计划和状态记录，不执行上述分支更新。下一次代码更新完成后检查 diff、冲突标记、适用 lint 与针对性测试，推送后核对本地 / 远端 SHA 和 ahead/behind `0 0`；使用 Gitmoji 英文提交。硬件验证和 CI 归因按当前 head 记录，不能继承旧 head 的通过结论。

### 5.3 重构任务与已有功能栈的关系

```text
核对基线与 G1.1 的 scale / loader 契约 ──→ G2.2 Diffusion 复用
                                      └→ G2.3 能力校验试点
G2.1 Dense W4A8：实现已提交 ─────────────→ 对应 NPU 验证与审阅
G1.2 在线 MoE：沿用现有功能 PR ──────────→ 层级精度与并行验证
G4 跨代际切片：独立推进；G3 与公共框架协调
上述每个改动 ──────────────────────────→ G5 证据与发布状态更新
```

G2 新分支按实际依赖取最小基线，不把整个 Gemma / MTP / FIA 栈作为架构重构的默认前置。跨仓依赖通过 API / wheel 验证。G2.1 已从 #43739 开发，但新增实现提交尚未回流该功能 PR；G2.2 / G2.3 已分别建分支，不能据此认为已有实现。具体分支定位见 [branches.md](branches.md#current-branch-map)。

## 6. 跟踪与后续会话使用规则

每次推进任务时，在对应清单勾选已满足的条件，更新第 2 节的任务状态；PR 检查、Draft、合并或替代关系只更新第 5 节，branch / HEAD / worktree 变化更新 `branches.md`。再追加一条进度记录，不再新建第二份“当前 PR 状态”表：

| 日期 | 任务 | 变化与证据 | 后续动作 |
| --- | --- | --- | --- |
| 2026-10-10 | 基线审阅 | 原版 HTML 完成，包含 12 条 SRT 路径与架构对照 | 作为历史快照归档，不自动更新其结论 |
| 2026-10-12 | G2.1 | `24e100f1e8` 已提交并推送；本次核对本地/远端一致，无 PR；已有 CPU/mock 记录见分支文档 | NPU 验证、审阅与合入 |
| 2026-10-12 | 全计划 | 建立本跟踪文件，归档原 HTML，明确 G2.2 / G2.3 的范围和验收 | 分别实施 G2.2、G2.3 |
| 2026-10-12 | 合入队列与跟踪分工 | 纳入原 1–8 项及 kernel #824；改用 #43739（原 #32601）、#43740（原 #34387）；通过 `gh` 核对 7 个待合入个人 PR 和配套 PR；已合入 / 已关闭项退出待合入队列 | PR 状态统一维护第 5 节，分支定位维护 `branches.md` |
| 2026-10-12 | G2.2 / G2.3 准备 | 本地已建两个独立 worktree，均无工作区改动或相对建分支基线的实现增量；本次未查到对应 PR | 实施后补提交、验证证据和交付信息 |

**NPU Qwen-VL 预处理补丁待办（2026-10-13，用户要求暂缓）**

- [ ] 精度问题定位后，收敛本仓 NPU 预处理补丁：评估复用 Transformers 的分组、resize、rescale/normalize 流程，仅保留 NPU 必要的 patch/layout 适配，降低复制整个 `_preprocess` 带来的接口漂移。图片和视频一起覆盖，并记录支持的 Transformers 接口范围。
- [ ] 补真实 HF processor 的 CPU 集成回归和目标 NPU warmup/图片/视频验证；本次隔离 CPU 契约检查不能替代该验收。

新增记录模板：`日期 / 任务 ID / 实现提交或证据链接 / PR / 改动 / 已验证范围与日志 / 未验证范围 / 下一步`。滚动分支定位引用 `branches.md`；验证证据中的 SHA 固定保留。不要用一个“完成”同时代替实现、硬件验收和上游合并。以上表格是有日期的人工维护记录，不是自动轮询面板；报告或操作前用 `gh` 刷新。

交给新会话时可以直接说：

> 阅读 AGENTS.md 和 docs/mxfp-refactoring-plan.md，执行 G2.2（Diffusion 离线 MXFP8 公共执行复用）与 G2.3（Ascend MXFP 局部能力描述、提前校验、文档和测试）。先核实 G2.1 的已有成果、最新基线和用户改动，分别在 sglang/ 直属独立 worktree 实施，按文中的验收清单验证并更新进度。

工作范围、提交与发布按用户当次授权和仓库规则执行，已经明确授权的动作不重复询问。默认单 agent，保留不相关改动。
