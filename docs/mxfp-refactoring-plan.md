# Ascend MXFP 重构目标与执行计划

更新日期：2026-10-15（Asia/Shanghai）。本文是任务、验收与 PR 合入入口，无需依赖聊天历史。

范围：SGLang Ascend SRT Dense / MoE × W8A8 / W4A8 / W4A4 × 在线 / 离线 ModelSlim，以及 Diffusion MXFP8 执行复用。CUDA / ROCm 和其他量化格式不在本计划的实现范围内。

| 信息 | 唯一维护入口 |
| --- | --- |
| 目标、验收证据、未测范围 | 本文[目标表](#refactoring-status)与第 3 节 |
| PR 状态、检查与依赖 | 本文[PR 表](#pr-merge-status)与[合入顺序](#merge-dependencies) |
| branch / worktree / HEAD / remote | [branches.md](branches.md#current-branch-map) |
| 稳定能力与代码路径 / 故障机制 | [quantization-overview.md](quantization-overview.md) / [known-pitfalls.md](known-pitfalls.md) |

[2026-10-10 HTML 审阅](reports/mxfp-architecture-roadmap-2026-10-10.html)是基于 `e35f60b9bcb51045fd0602d8ed12cff8021c040f` 的历史快照；其中六个执行步骤、旧 PR 号和状态不替代本文 G1–G5 / G2.1–G2.3 编号。删除的中间过程从 Git 历史追溯，不另建重复状态表。

## 1. 目标与职责边界

相同数学与执行契约只实现一次，在线准备与离线加载各司其职，支持条件可提前检查，新模型复用已有 Scheme / Kernel，支持声明有对应验证证据。

依据：[量化分层 #31783](https://github.com/sgl-project/sglang/issues/31783)、[Ascend MXFP 覆盖 #34559](https://github.com/sgl-project/sglang/issues/34559)、[跨代际分层 #35709](https://github.com/sgl-project/sglang/issues/35709)、[能力校验 #31774](https://github.com/sgl-project/sglang/issues/31774)。Issue 状态和复选框不能代替代码核对。

```text
Config：格式、来源、层与策略
  → Method：加载、准备、执行生命周期
    → Scheme：参数、loader metadata、scale / packing
      → 共享 Kernel：消费 prepared tensors，执行 quant / matmul / GMM

在线 BF16/FP16 → 量化与布局准备 ─┐
                               ├→ 契约兼容的公共执行
离线预量化权重 → 加载与布局准备 ─┘

独立算子内部差异 → kernel 稳定 API / 构建期 provider
跨组件 dtype/layout/metadata 差异 → 按需使用 DeviceOperator
```

- Dense 指 Linear 路径，MoE 指专家路径；不能按模型名称混淆二者。
- W4A8 是 FP4 权重 + FP8 activation；Dense 在线 W4A4 双级 L0/L1 与离线/MoE 单级格式不可强行合并。离线 packed uint8 不得二次量化或打包。
- shape 条件按实际 TP 分片判断；实现、CPU 检查、NPU 功能、数值等价、性能和上游合入分别记录。缺少硬件结果既不等于不支持，也不等于已验证。

<a id="refactoring-status"></a>

## 2. 重构目标完成情况

| 编号 | 目标 / 任务 | 实现进度（2026-10-15） | 验证与剩余验收 | 交付载体 / 合入跟踪 |
| --- | --- | --- | --- | --- |
| G1.1 | 修正 MXFP8 partial-scale 与布局边界 | 2026-10-16 补齐 SRT ModelSlim RowParallel 未对齐 TP 的 scale 切片与 W/X 补零，并移植到组合验证分支；Diffusion loader 仍需单独核对 | 两分支各 6 项隔离 CPU 测试通过，含 K=4304、TP1/2/4/8 全部 rank；对应 TP 的目标 NPU 加载、warmup 和请求尚未验证 | #32745；分支与 HEAD 见 `branches.md`，状态见下方 PR 表 |
| G1.2 | 补在线 MoE W4A8 / W4A4 与实际精度分配 | 两条功能分支已有实现，保持各自能力增量 | 专家确实量化；层级精度、融合、dispatch、TP/EP 与在线 RTN 精度需有证据 | #43739（原 #32601）/ #32602；状态见 PR 表 |
| **G2.1** | **Dense W4A8 在线/离线公共执行** | **实现已发布 PR；原验收分支在线/离线功能通过** | 原 8 项 CPU/mock；新 PR 去除无关配置测试后 6 项通过。2026-10-14 用户确认在线/离线请求正常；在线视觉 27 个 fc2 保留 BF16 回退。新发布 head 尚无同场景真机通过证据，数值等价未测 | #44363；混精独立 #44364。原实现 `24e100f1e8`、离线验收载体 `2ef4c276be` |
| **G2.2** | **Diffusion 离线 MXFP8 委托公共 kernel** | **实现已发布 PR；原验收分支图生视频功能通过** | 2026-10-14 用户确认 I2V 输出成功、视频观感正常；新 PR 3 项 CPU/mock 通过。新发布 head 尚无同场景真机通过证据；按用户要求跳过数值比较，独立 warmup、量化误差和性能未实测 | #44365；原实现 `6eb8e1dd1c`，发布定位见 `branches.md` |
| **G2.3** | **Ascend MXFP 局部能力描述与提前校验** | **局部试点已发布 PR；原验收分支图生视频功能通过** | 17 条能力描述；原 14 项 CPU/mock，新 PR 15 项及叠加的 3 项 adapter 测试通过。用户确认原分支 I2V 正常；新组合 head 的同场景真机、其他能力、TP/EP/backend、数值与性能未验收 | #44366，依赖 #44365；原实现 `d6a46f90d9`，发布定位见 `branches.md` |
| G3 | 权重生命周期与公共在线入口 | 待设计 / 实施，与公共框架工作协调 | 参数、元数据、重载语义明确；入口迁移兼容既有 CLI | 尚未拆出交付 PR |
| G4 | 跨代际差异按契约归属收敛 | Gemma 切片已有实现；整体未完成 | provider 可追溯；构建 target、API 与配套 wheel 验证；只在必要处引入运行时契约层 | kernel #638 + #32745；状态见 PR 表 |
| G5 | 模型、算子、并行与 CI 的硬件验收 | 持续工作 | G2.1 在线/离线、G2.2 / G2.3 图生视频均已有用户真机功能通过反馈；数值、性能、并行与其他能力覆盖仍分别记录 | 随各任务记录证据，CI 通过不替代实际硬件覆盖 |

原分支的用户真机反馈只证明报告场景的功能结果；新 PR head 的 CPU/mock 或 lint 不能继承为 NPU 验收。PR 状态统一见第 5 节。

**G1.1 本轮验证（2026-10-16）**：按全局权重起点读取 scale，并将左侧偏移计入 scale 数量，覆盖 TP4 部分 rank 需要 35 个 block 的情况。实际 scheme、Parameter 与离线 kernel 源码在隔离 serving 导入后通过 CPU 回归，包含 checkpoint 权重参考计算、二维/三维输入 padding、已分片 scale、TP1/Column/已对齐路径；未扩展未对齐 TP 的预量化 tuple 输入。标准 SRT 测试入口因缺 `orjson` 未完成，未调用真实 NPU 算子。组合验证分支的默认全仓注册检查仍有 572 条历史 taxonomy 错误，修改前后输出完全一致；以该分支为比较基线的注册检查通过。

主仓启动脚本另有 37 项隔离 shell/PTY 测试通过，覆盖按卡号数量决定 TP、保留默认 TP2、`all` 的 1/2/8 卡、旧设备掩码、清理确认/取消及 LoRA 不清理。下一步由现有启动脚本在目标 NPU 验证 TP2、TP4 和设备数量允许时的 `all`，检查加载、warmup、文本/图片请求；目前尚无这些配置的真机通过结果，也无精度或性能结论。

**Qwen3.5 TP2 MRoPE 复测（2026-10-16）**：用户在组合分支的上一版 TP 修复上运行 `./qwen3.5_dense_offline_w8a8.sh 6 7`，请求进入语言模型后报 `Expected positions shape to be [80] in rope mode, but got [3, 80]`。按 Qwen3.5-27B 配置，Q 的本地宽度从 TP1 的 6144 降为 TP2 的 3072，触发原有 `<=4096` NPU 分支，其固定 `[0,0,0]` section 选择了普通 RoPE。两分支已单独提交并同步最小修复：二维位置复用现有 native MRoPE，保留三轴、交错与 partial rotary 语义；未展平位置或降低 TP。各分支 2 项隔离 CPU/mock 回归通过，包含 TP1/2/4/8 全部 rank、1/80 token、不同三轴坐标、分段/交错排列、未旋转尾部及一维 NPU 分流，原 6 项 MXFP8 测试也通过。这不是 NPU 复测通过；仍需相同脚本完成加载、warmup 和实际请求，性能未测。

**Qwen3.5 NPU prefill 融合接入（2026-10-16）**：两个 SGLang 分支均仅修改 `srt/models/qwen3_5.py` 和对应 CPU 回归测试，复用现有 `sgl_kernel_npu.norm.split_qkv_rmsnorm_mrope.triton_split_qkv_rmsnorm_mrope`，没有修改 kernel 或主仓 gitlink。仅将 NPU 的 EXTEND/MIXED/SPLIT_PREFILL 接入融合路径，传入本地 Q/KV head 数、Gemma 的 `1 + weight`、交错标志与实际 rotary_dim；一维文本位置扩为三轴，cos/sin 转为 QKV 的 device/dtype，无 gate 时保持返回 None，无 MRoPE 配置仍走原 native。decode、verify、draft 与其他后端保持原分流，原 decode 方法正文不变。按审阅要求，prefill/MRoPE 测试已从两个新增文件合并到已有 `test/registered/rotary/test_mrope_axis_map.py`，删除重复参数矩阵，净减少 192 行，保留三轴位置路由、prefill/decode 分流及 TP2 融合参数 3 项回归。连同原文件 5 项，两分支各 8 项隔离 CPU/mock 测试通过；PR 分支恢复旧路由或去掉 Gemma 权重偏移的 3 种故障注入均被对应测试检出。完整测试入口因本机缺 `orjson` 未运行。本次没有修改生产代码；此前 32 组 adapter + kernel wrapper CPU 调用契约检查仍作为历史证据，Triton 启动被 mock。未执行 NPU kernel 或模型 e2e，没有数值等价或性能结论；服务器需已安装包含上述接口的 wheel，由用户更新组合分支后沿用现有脚本完成真机复测。

## 3. G2 实施与验收摘要

以下 SGLang 路径相对于 `python/sglang/`；滚动 HEAD 只在 `branches.md` 维护。实现已存在，不重新实现。

### G2.1：Dense W4A8 公共执行与混精配套

- `linear_method_npu.py` 中在线/离线 `apply()` 共用 `npu_w4a8_mxfp4_linear()`，各自保留加载、量化和 post-load；公共执行消费 prepared weight/scale，统一 BF16 `[1,N]` bias。原提交 `24e100f1e8` 基于 `ded4671577`，发布为 #44363。
- 混精修复 #44364 独立处理 Qwen3.5 Dense 的完整、异构 gate/up 描述：拆成两套现有 `ColumnParallelLinear`，保留各自 weight/scale loader，再按 gate/up 顺序进入原激活。发布前未发现上游等价实现；#38870 的 Quark MTP 修复不等价，已合入 #43309 的 resize 不重复提交。
- **用户真机反馈（2026-10-14）**：在线、离线启动与请求正常。离线部署 `2ef4c276be`、checkpoint `/mnt/share/weight/Qwen3.6-27B_mxfp-w4a8-0821`，保留 BF16 SSM，去掉禁 graph 参数并停止重启服务后正常。历史乱码根因未证实，不固化禁 graph 或新增 update/replay 修复。
- 在线使用 `llm/qwen3.5_dense_online_w4a8.sh`；日志中 27 个视觉 fc2 因 K=4304 保留既有 BF16 回退，未再次附完整源码 SHA。这不等于全视觉 W4A8 通过。prepared tensors 的 NPU 数值等价、完整请求矩阵、并行与性能未测；可选诊断脚本不代表这些验收已完成。

### G2.2：Diffusion 离线 MXFP8 公共执行

- `modelslim_mxfp8_scheme.py` 保留 checkpoint 参数、loader metadata 和 odd-scale padding，post-load/apply 委托 `NPUMXFP8LinearMethod`。原提交 `6eb8e1dd1c` 基于 `62ff466ac0`，已有 3 项 CPU/mock 与 27 组调用契约对照；发布为 #44365。
- **用户真机反馈（2026-10-14）**：Wan2.2 I2V-A14B 离线 ModelSlim MXFP8 成功生成视频，人工观感正常。此前日志显示双 transformer 加载完成、Torch SDPA、CPU offload，并进入去噪。
- 已有 checkpoint `/mnt/share/weight/Wan2.2-I2V-A14B-Diffusers-MXFP8-G2`，两套 transformer 的 repack 元数据检查通过，无需下载或转换。[repack 记录](branches.md#g2-runtime-record)与[格式陷阱](known-pitfalls.md#wan-modelslim-repack)保留必要证据。
- 脚本为 `diffusion/i2v_offline_mxfp8.sh`，使用 `gyro.jpg`、81 帧、40 步。成功运行所附日志为 1280×704，脚本随后调整为 1280×720；新尺寸不因旧反馈自动记为已测。独立 warmup、数值等价、性能及 MXFP8 attention 未验收。

### G2.3：局部能力描述与提前校验

- `mxfp_capabilities.py` 的 17 条描述接入 Config、Method、MoE runner、已有 fallback、支持文档与测试。原提交 `d6a46f90d9` 基于 `62ff466ac0`，原有 14 项 CPU/mock；区分 runtime、层类型、来源、dtype/scale/layout/backend，在配置、分片和 post-load 阶段提前拒绝已知不兼容组合。
- **用户真机反馈（2026-10-14）**：“2.3也能运行，出来结果也正常。”按约定的 Wan2.2 I2V 离线 MXFP8 场景记录功能通过；未附完整日志或视频文件，服务器 SHA、尺寸和 backend 未逐项核验，本机未执行 NPU。
- 原 G2.2/G2.3 验收分支相互独立。发布 #44366 时叠加 #44365，并同步 Diffusion partial-scale 校验、文档和回归；其他单级 ModelSlim loader 仍保留 K=64 限制。新组合分支有 15 项能力测试和 3 项 adapter CPU/mock 测试通过；MDX 注释修正后再次通过 15 项能力测试。
- 功能反馈不扩展为全部能力、非法配置拒绝分支、MoE、TP/EP、数值或性能通过。新发布 head 尚无同场景真机通过证据；旧混精或其他功能分支的结果不作为本分支证据。

<a id="g23-handoff"></a>

### 按需复跑与后续接手

本轮实际模型功能验收已结束，后续按 PR review、实际错误或用户新任务推进。用户已跳过的数值对照、基线比较、额外 smoke、重新下载/repack 不再作为本轮前置。

如需复跑，仍由用户手动选择服务器分支。原 G2.3 验收分支是 `junlin/ascend-mxfp-capabilities`，记录的实现为 `d6a46f90d97039e14476be47468c996f11caf603`；它与 `junlin/pr-ascend-mxfp-capabilities` 发布分支不同，不能自动替换。主仓保持 `main`。

在执行目录沿用已有入口：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 bash diffusion/i2v_offline_mxfp8.sh
```

脚本使用上述现有权重和 `diffusion/gyro.jpg`，视频、日志、临时文件、缓存均在执行目录下 `outputs_i2v_mxfp8/`；不默认写 `/tmp` 或 `/root`，不自动 fetch、切分支、改 PYTHONPATH 或部署环境。有新错误再依实际日志定位；修改进入对应独立 worktree，保留其他脏文件。

## 4. 其他目标的完成条件

- **G1.1 scale 边界**：K=4304 需 `ceil(K/32)=135` 个 block，pair 容量为 136；不可丢余块或直接 reshape 奇数 block。实际支持还取决于分片、padding 和算子，placeholder 修正不等于任意 K 可运行。
- **G1.2 在线 MoE**：复用 W4 GMM/Ascend runner，补在线准备与分发。保留融合/clamp、DeepEP normal=BF16 / low-latency=MXFP8，解释 experts 与非专家 Linear 的实际精度；TP/EP/dispatch 和在线 RTN 精度分别需要证据。
- **G3 生命周期**：明确 `创建 → 加载 → 准备 → 执行 → 重载/更新` 的权重、scale、bias、属性和所有者，避免双重 transpose 或旧 cache。与公共 loader 工作协调并兼容现有 CLI，不作为 G2 的隐式前置。
- **G4 跨代际**：独立算子差异进入 kernel provider，跨组件表示才按需进入 DeviceOperator；Gemma 是已有切片，不强制每个量化方法增加框架。
- **G5 证据**：按实际覆盖记录 SHA、硬件、kernel wheel/target、CANN/torch/torch_npu、checkpoint、TP/EP/backend、命令与日志。静态、CPU、NPU kernel、模型加载/warmup、请求、数值和性能是不同证据维度，不是每次任务都要重跑的固定清单。

## 5. PR 合入计划

<a id="pr-merge-status"></a>

### 当前 PR 跟踪（2026-10-15 快照；#32745 于 2026-10-16 更新）

保留原计划序号；序号不是合入顺序。所有 Open PR 均 `base=main`、仍需 review。`UNKNOWN` 不推定为无冲突，CI 跳过不等于硬件通过；本表不是自动监控面板。

| 序号 | PR | 职责 | 状态 / Git 合并检查 | 当前 CI 与下一步 |
| --- | --- | --- | --- | --- |
| 1 | [kernel #638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | Gemma 稳定 API / 构建期 provider；G4 | Open，非 Draft；MERGEABLE | 检查通过，1 项跳过；#32745 的 wheel/API 前置 |
| 2 | [SGLang #32745](https://github.com/sgl-project/sglang/pull/32745) | Gemma API、MXFP8 partial-scale / TP 对齐；G1.1/G4 | Open，非 Draft；MERGEABLE | 10-16 新 head lint 通过；整体未通过：要求包含基线 `b67785153a55`、现有标题不符合新格式、extra 缺 `run-ci-extra`；部分检查仍运行/排队。本轮只提交最小 TP 修复，未同步 main、调整标题或添加标签 |
| 3 | [SGLang #32041](https://github.com/sgl-project/sglang/pull/32041) | Wan2.2 MXFP8 Attention 历史实现 | Closed，未合并 | 移出队列；核心 Attention 由 #39438 替代，不等于 G2.2 去重 |
| 4 | [SGLang #40419](https://github.com/sgl-project/sglang/pull/40419) | speculative SSM/intermediate state BF16 | Open，非 Draft；UNKNOWN | lint 通过，整体 FAILURE；与已合入 kernel #808 配套 |
| 5 | [kernel #808](https://github.com/sgl-project/sgl-kernel-npu/pull/808) | Ascend 950 recurrent GDN / BF16 断言 | Merged，2026-09-29 | 已移出队列，不在原 PR 继续追加 |
| 6 | [SGLang #43739](https://github.com/sgl-project/sglang/pull/43739)（原 #32601） | 在线 MoE W4A8 / 融合 GMM；G1.2 | Open，非 Draft；UNKNOWN | lint 通过，整体 FAILURE；已不再是 Draft，依赖更新后的 #32745 |
| 7 | [SGLang #32602](https://github.com/sgl-project/sglang/pull/32602) | 在线 MoE W4A4 / 融合 GMM；G1.2 | Open，Draft；UNKNOWN | lint 通过，整体 FAILURE；承接 #43739 |
| 8 | [SGLang #43740](https://github.com/sgl-project/sglang/pull/43740)（原 #34387） | Ascend 950 FIA mixed split | Open，Draft；UNKNOWN | lint 通过，整体 FAILURE；依赖更新后的 #32745 |
| 9 | [kernel #824](https://github.com/sgl-project/sgl-kernel-npu/pull/824) | cache location view 边界修复 | Open，非 Draft；MERGEABLE | 检查通过，1 项跳过；独立推进 |
| 10 | [SGLang #44363](https://github.com/sgl-project/sglang/pull/44363) | G2.1 公共 W4A8 执行 | Open，非 Draft；MERGEABLE | lint 通过；初始 gate 缺 `run-ci`，整体未通过 |
| 11 | [SGLang #44364](https://github.com/sgl-project/sglang/pull/44364) | Qwen3.5 Dense ModelSlim 混精 | Open，非 Draft；MERGEABLE | lint 通过；初始 gate 缺 `run-ci`，整体未通过 |
| 12 | [SGLang #44365](https://github.com/sgl-project/sglang/pull/44365) | G2.2 Diffusion MXFP8 复用 | Open，非 Draft；MERGEABLE | lint 通过；已有实际 AMD/NPU 测试失败，部分运行/排队；不能仅归为 gate 阻塞 |
| 13 | [SGLang #44366](https://github.com/sgl-project/sglang/pull/44366) | G2.3 能力描述与 preflight | Open，Draft；MERGEABLE | MDX 注释修正后新 head lint 已通过；Draft gate 阻断，整体未通过 |

#44363–#44366 的原始真机反馈与新 head CPU 验证在第 3 节分开记录。#44365 的具体测试失败尚未归因，不推定为本改动回归；#44366 的 MDX 修正已通过[新 head lint](https://github.com/sgl-project/sglang/actions/runs/38705953919/job/116325305442)。状态及日志快照在 `.pi/docs-refresh-20261015/`，不把本地证据路径当作上游可访问报告。

### 配套与替代关系

| PR | 2026-10-15 状态 | 作用 |
| --- | --- | --- |
| [SGLang #36426](https://github.com/sgl-project/sglang/pull/36426) | Open，非 Draft；UNKNOWN，CI FAILURE | GDN 名字解析/视觉塔 quant config 组合验证；不混入正式功能栈 |
| [kernel #742](https://github.com/sgl-project/sgl-kernel-npu/pull/742) / [#651](https://github.com/sgl-project/sgl-kernel-npu/pull/651) | 分别 Open Draft / Open 非 Draft；均 CONFLICTING | 历史 MTP state 迁移配套，含 UB block 调整；恢复时核对当前覆盖与实际部署，不当作已合入 #808 的待合入前置 |
| [kernel #747](https://github.com/sgl-project/sgl-kernel-npu/pull/747) / [SGLang #39589](https://github.com/sgl-project/sglang/pull/39589) | 均 Merged | SSM layout 配套；部署 wheel 与 SGLang 必须匹配 |
| [SGLang #39438](https://github.com/sgl-project/sglang/pull/39438) | Merged | Wan2.2 MXFP8 Attention，替代 #32041 的核心功能 |
| [SGLang #43309](https://github.com/sgl-project/sglang/pull/43309) | Merged | resize 兼容修复，已核对等价，不重复提交 |
| [SGLang #32266](https://github.com/sgl-project/sglang/pull/32266) / [kernel #823](https://github.com/sgl-project/sgl-kernel-npu/pull/823) | 均 Closed，未合并 | 历史实现与对照来源，移出队列 |

<a id="merge-dependencies"></a>

### 合入顺序与并行范围

```text
kernel #638 ── wheel/API ──→ SGLang #32745 ──→ #43739 ──→ #32602
                                          └→ #43740
SGLang #40419：与已合入 kernel #808 配套，独立推进
kernel #824：独立推进

G2.1 #44363：独立最小切片，不依赖整个 #43739 栈
混精 #44364：独立正确性修复，不携带 resize
G2.2 #44365 ──→ G2.3 #44366（Draft）
```

- #32745 需要包含 #638 provider 的配套 wheel；#734 的重叠实现已并入，不重复叠加。跨仓依赖不能用 Git 祖先关系代替。
- #43739 先同步 main 和更新后的 #32745，再更新 #32602；保留 main 的 scale helper、融合/clamp、DeepEP 契约与单级/双级区别。#43740 随 #32745 更新后可与 MoE 栈并行。
- #40419 保留 main 的 GDN backend/prefill，仅维护自己的 dtype 契约；#824 不混入 #638。#36426 保持组合验证用途。
- #44365 先合入，再同步 #44366；能力增量可相对前置分支审阅。#44363 与 #44364 可独立审阅，但组合运行仍需核对实际配置与代码版本。
- 操作前刷新 main、PR head、worktree 与脏文件。全部正式 PR 保持 `base=main`，通过普通 merge/追加提交和 push 更新；未经授权不强推。前置合入后核对内容并清理重复增量，代码在各自直属 worktree 修改。

## 6. 后续维护与暂缓项

每次只更新目标表、PR 表和分支定位中发生变化的部分，保留足以复核的 `日期 / SHA / 场景 / 证据 / 未测范围 / 下一步`。不再追加逐次命令、CI 计数或重复“尚未发布”的过程流水；完整历史用 Git 追溯。

- **G2**：原分支功能验收已结束，四个 PR 已发布。后续依据 review、实际 CI 失败或新模型日志继续；用户跳过的数值/基线比较不重新设为前置。
- **NPU Qwen-VL 预处理整理**：2026-10-13 用户要求暂缓，尚未恢复。后续如授权，评估复用 Transformers 分组、resize、rescale/normalize，仅保留 NPU 必要 patch/layout，图片与视频一起覆盖；不要把已合入的 #43309 再提一遍。
- **证据更新**：新 head 的测试、日志、视频反馈各自绑定版本；用户反馈标明来源，不能写成本机 NPU 已执行或完整数值等价。

接手先读 AGENTS.md、本文件的目标/PR 表和 [branches.md](branches.md#current-branch-map)；按需复跑入口见 [g23-handoff](#g23-handoff)。默认单 agent，按当次授权提交发布并保留不相关改动。
