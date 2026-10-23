# Ascend MXFP 重构目标与执行计划

更新日期：2026-10-23（Asia/Shanghai；本次仅刷新 #32602 的独立化与依赖，其余 PR 全表状态仍为 10-22 快照）。本文是任务、验收与 PR 合入入口，无需依赖聊天历史。

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

| 编号 | 目标 / 任务 | 实现进度 | 验证与剩余验收 | 交付载体 / 合入跟踪 |
| --- | --- | --- | --- | --- |
| G1.1 | 修正 MXFP8 partial-scale 与布局边界 | SRT ModelSlim RowParallel 未对齐 TP 的 scale 切片与 W/X 补零已随 #32745 于 2026-10-21 合入；Diffusion loader 仍需单独核对 | 两分支各 6 项隔离 CPU 测试通过，含 K=4304、TP1/2/4/8 全部 rank；TP2 用户反馈见下，合入不补齐其他 TP 或新 head 的目标 NPU 验收 | #32745 已合入；分支与正式合入提交见 `branches.md` |
| G1.2 | 补在线 MoE W4A8 / W4A4 与实际精度分配 | #43739 于 10-22 同步 main、收敛为 7 文件；#32602 于 10-23 单独同步 main、去掉 W4A8 增量并收敛为 8 文件，两项不再互为源码/合入前置 | #43739 原 25 项量化 + 21 项 Gemma 隔离 CPU 回归不变；独立 #32602 有 19 项 W4A4 + 13 项共享基线回归通过。两项当前 head 的模型加载、warmup、请求、RTN 精度与 TP/EP 仍待真机 | #32602 于 10-23 按用户授权转为 Ready for review、独立推进；W4A8 正式及组合分支未改动 |
| **G2.1** | **Dense W4A8 在线/离线公共执行** | **实现已发布 PR；原验收分支在线/离线功能通过** | 原 8 项 CPU/mock；新 PR 去除无关配置测试后 6 项通过。2026-10-14 用户确认在线/离线请求正常；在线视觉 27 个 fc2 保留 BF16 回退。新发布 head 尚无同场景真机通过证据，数值等价未测 | #44363；混精独立 #44364。原实现 `24e100f1e8`、离线验收载体 `2ef4c276be` |
| **G2.2** | **Diffusion 离线 MXFP8 委托公共 kernel** | **实现已发布 PR；原验收分支图生视频功能通过** | 2026-10-14 用户确认 I2V 输出成功、视频观感正常；新 PR 3 项 CPU/mock 通过。新发布 head 尚无同场景真机通过证据；按用户要求跳过数值比较，独立 warmup、量化误差和性能未实测 | #44365；原实现 `6eb8e1dd1c`，发布定位见 `branches.md` |
| **G2.3** | **Ascend MXFP 局部能力描述与提前校验** | **局部试点已发布 PR；原验收分支图生视频功能通过** | 17 条能力描述；原 14 项 CPU/mock，新 PR 15 项及叠加的 3 项 adapter 测试通过。用户确认原分支 I2V 正常；新组合 head 的同场景真机、其他能力、TP/EP/backend、数值与性能未验收 | #44366，依赖 #44365；原实现 `d6a46f90d9`，发布定位见 `branches.md` |
| G3 | 权重生命周期与公共在线入口 | 待设计 / 实施，与公共框架工作协调 | 参数、元数据、重载语义明确；入口迁移兼容既有 CLI | 尚未拆出交付 PR |
| G4 | 跨代际差异按契约归属收敛 | Gemma kernel provider 与 SGLang 配套均已合入，Gemma 源码切片交付完成；其他契约与硬件验收继续按需推进 | provider 可追溯；仍需对应构建 target、API 与部署 wheel 的验证；只在必要处引入运行时契约层 | kernel #638 + #32745 均已合入；状态见 PR 表 |
| G5 | 模型、算子、并行与 CI 的硬件验收 | 持续工作 | G2.1 在线/离线、G2.2 / G2.3 图生视频均已有用户真机功能通过反馈；数值、性能、并行与其他能力覆盖仍分别记录 | 随各任务记录证据，CI 通过不替代实际硬件覆盖 |

原分支的用户真机反馈只证明报告场景的功能结果；新 PR head 的 CPU/mock 或 lint 不能继承为 NPU 验收。PR 状态统一见第 5 节。

**G1.1 本轮验证（2026-10-16）**：按全局权重起点读取 scale，并将左侧偏移计入 scale 数量，覆盖 TP4 部分 rank 需要 35 个 block 的情况。实际 scheme、Parameter 与离线 kernel 源码在隔离 serving 导入后通过 CPU 回归，包含 checkpoint 权重参考计算、二维/三维输入 padding、已分片 scale、TP1/Column/已对齐路径；未扩展未对齐 TP 的预量化 tuple 输入。标准 SRT 测试入口因缺 `orjson` 未完成，未调用真实 NPU 算子。组合验证分支的默认全仓注册检查仍有 572 条历史 taxonomy 错误，修改前后输出完全一致；以该分支为比较基线的注册检查通过。

主仓启动脚本另有 37 项隔离 shell/PTY 测试通过，覆盖按卡号数量决定 TP、保留默认 TP2、`all` 的 1/2/8 卡、旧设备掩码、清理确认/取消及 LoRA 不清理。初次发布时尚无 TP2、TP4 或 `all` 的真机通过结果；后续 TP2 用户反馈见下，TP4 和设备数量允许时的 `all` 仍待现有脚本验证。

**Qwen3.5 TP2 MRoPE 复测（2026-10-16）**：用户在组合分支的上一版 TP 修复上运行 `./qwen3.5_dense_offline_w8a8.sh 6 7`，请求进入语言模型后报 `Expected positions shape to be [80] in rope mode, but got [3, 80]`。按 Qwen3.5-27B 配置，Q 的本地宽度从 TP1 的 6144 降为 TP2 的 3072，触发原有 `<=4096` NPU 分支，其固定 `[0,0,0]` section 选择了普通 RoPE。两分支已单独提交并同步最小修复：二维位置复用现有 native MRoPE，保留三轴、交错与 partial rotary 语义；未展平位置或降低 TP。各分支 2 项隔离 CPU/mock 回归通过，包含 TP1/2/4/8 全部 rank、1/80 token、不同三轴坐标、分段/交错排列、未旋转尾部及一维 NPU 分流，原 6 项 MXFP8 测试也通过。这不是 NPU 复测通过；仍需相同脚本完成加载、warmup 和实际请求，性能未测。

**Qwen3.5 NPU prefill 融合接入（2026-10-16）**：两个 SGLang 分支均仅修改 `srt/models/qwen3_5.py` 和对应 CPU 回归测试，复用现有 `sgl_kernel_npu.norm.split_qkv_rmsnorm_mrope.triton_split_qkv_rmsnorm_mrope`，没有修改 kernel 或主仓 gitlink。仅将 NPU 的 EXTEND/MIXED/SPLIT_PREFILL 接入融合路径，传入本地 Q/KV head 数、Gemma 的 `1 + weight`、交错标志与实际 rotary_dim；一维文本位置扩为三轴，cos/sin 转为 QKV 的 device/dtype，无 gate 时保持返回 None，无 MRoPE 配置仍走原 native。decode、verify、draft 与其他后端保持原分流，原 decode 方法正文不变。按审阅要求，prefill/MRoPE 测试已从两个新增文件合并到已有 `test/registered/rotary/test_mrope_axis_map.py`，删除重复参数矩阵，净减少 192 行，保留三轴位置路由、prefill/decode 分流及 TP2 融合参数 3 项回归。连同原文件 5 项，两分支各 8 项隔离 CPU/mock 测试通过；PR 分支恢复旧路由或去掉 Gemma 权重偏移的 3 种故障注入均被对应测试检出。完整测试入口因本机缺 `orjson` 未运行。本次没有修改生产代码；此前 32 组 adapter + kernel wrapper CPU 调用契约检查仍作为历史证据，Triton 启动被 mock。未执行 NPU kernel 或模型 e2e，没有数值等价或性能结论；服务器需已安装包含上述接口的 wheel，由用户更新组合分支后沿用现有脚本完成真机复测。

**Qwen3.5 TP2 后续真机反馈（2026-10-16）**：用户确认运行 `./qwen3.5_dense_offline_w8a8.sh 6 7` 已正常，且“精度看起来也正常”。据此记录该脚本 TP2 场景的用户报告功能通过及主观输出质量正常；本次反馈未附服务器 SHA 或原始日志，不扩展为本机执行、数值等价、量化评测分数、性能或 TP4/`all` 通过。

**Dense + #36426 组合同步验证（2026-10-20）**：组合分支已同步最新 dense 与 #36426，版本及增量见 [branches.md](branches.md#current-branch-map)。三组目标测试通过隔离 CPU/mock 回归：ModelSlim MXFP8 6 项、Gemma RMSNorm 21 项、MRoPE 8 项，共 35 项；另有 kernel registry/BaseFusedOp 81 项通过，4 项依赖完整 serving/JIT 环境的测试未运行。当前 ModelSlim 配置的实际名称解析逻辑通过 10 项 CPU probe，覆盖语言/视觉前缀、GDN 融合与拆分、FLOAT 跳过及混合精度拒绝。最终 4 文件增量的语法、Ruff、isort、codespell、diff 检查和测试注册检查通过。日志保存在 `.pi/pr32745-combo-sync-20261020/`；这些结果不代表标准完整环境入口、NPU 算子或模型 e2e 通过。旧版本的用户结果不继承为本次验收。该 Dense 组合分支已按用户 2026-10-22 授权清理，此段仅保留历史证据，不再作为服务器运行入口。

**组合分支启动反馈（2026-10-21）**：用户确认运行 `1593019ff9`、Transformers 5.12.1，离线 W8A8 服务在图片预热的 resize 调用报错；当前分支固定依赖为 5.19.0。官方旧版源码与实际 wrapper 的隔离 CPU 复现得到相同异常，5.19.0 契约检查通过，归因为同步源码后服务器依赖未对齐。处理及证据见 [已知陷阱](known-pitfalls.md#qwen-vl-图片预热与-transformers-接口不匹配)；升级依赖后的恢复结果尚未回传，不记为 NPU 功能或精度通过；旧组合载体已于 2026-10-22 清理。

**今日选定 PR 准备（2026-10-22）**：用户选择 #43739 在线 MoE W4A8 与 #45550 Diffusion 自动 FA，由本机准备、用户在服务器运行。两分支普通 merge/push 同步同一 main，具体 SHA 见 `branches.md`。#43739 清除旧 Gemma registry、无关离线 ModelSlim GDN/视觉塔增量，保留在线专家与必要的 Dense bias/视觉 fallback；增加 config/ignored-layer、runner/backend 拒绝、BF16 分配回归与 post-load packed weight/scale 日志。main 基线 13 项共享 W4/MXFP8 + 21 项 Gemma、更新后 25 项量化 + 21 项 Gemma 隔离 CPU 测试通过。#45550 10 项 CPU 回归通过，当前 main 上相同构造回归仍检出三种 self-attention 误选 SDPA。适用 Ruff hook、格式、isort、codespell、AST、注册检查与 PR diff 检查通过；完整测试入口均在缺 `orjson` 时失败，未到功能测试。证据在 `.pi/pr-ready-20261022/`。两项均尚未获得当前 head 的 NPU 功能、精度或性能结果，未向 committer 宣称可合入。

**MoE + #36426 组合准备（2026-10-22）**：按用户授权新建独立 MoE W4A8 组合分支并普通推送，仅在最新 #43739 上叠加 #36426 的完整 4 文件补丁，正式 PR 分支不变；版本和清理记录见 [branches.md](branches.md#current-branch-map)。本轮正式分支基线及组合分支各 25 项量化回归通过；组合另有 21 项 Gemma 与 10 项 ModelSlim 名称解析隔离 CPU 检查通过。四个文件的 AST、Ruff、格式、isort 和 diff 检查通过，补丁 patch-id 与实时 PR 一致。标准入口仍在缺 `orjson` 时导入失败；无 NPU 加载/warmup/请求、精度或性能结果。用户可在新组合分支沿用 `llm/qwen3.5_moe_online_w4a8.sh` 测试；#36426 主要覆盖离线 ModelSlim 与视觉 INT8，尚未证明是在线 W4A8 的必要前置。组合结果不能直接当作未叠加 #36426 的正式 PR head 验收。证据在 `.pi/cleanup-moe-combo-20261022/`。

**W4A4 独立化（2026-10-23）**：#32602 在原 worktree 普通 merge 最新 main，去掉继承的 W4A8 在线入口/config、W4A8 Dense bias 与无关 ModelSlim/旧 Gemma 增量；历史提交仍在，但相对 main 的 8 文件不再依赖 #43739。只修改 W4A4 kernel/在线入口、对应配置、runner 分流、Dense L0 scale 修复、必要的 NPU 算子延迟绑定及两份回归测试。公共 main 基线先通过 13 项隔离 CPU 回归，独立树随后通过 32 项（19 项 W4A4 + 同一组 13 项基线）；覆盖在线 BF16/FP16、离线 packed 不二次量化、W13/W2 layout/stride、BF16 dispatch、runner/backend、融合 FP4 dtype/group-list 与单级/双级 scale。保留 main 的 SiTU/自定义激活分流；W4A4 只在 ascend_tp 普通 SiLU 融合，`swiglu_limit` 仍拒绝。AST 对照确认 W4A8 kernel/Dense 定义与 main 一致，#43739 及其组合 HEAD/工作区未动。Ruff、格式、isort、codespell、语法、全仓注册与 diff 检查通过；标准入口仍因缺 `orjson` 在导入阶段失败。无新 NPU 模型/算子、精度、并行或性能结果。初次发布保留 Draft，同日用户授权后转为 Ready for review，并[回复 TamirBaydasov](https://github.com/sgl-project/sglang/pull/32602#issuecomment-6359951386)说明已可审阅、真机验证仍待完成；GitHub 状态及 `ready_for_review` 事件均已核实，未修改代码、标签或手动触发 CI。后续由用户手动选择该分支，沿用 `llm/qwen3.5_moe_online_w4a4.sh`。版本/发布见 `branches.md`，实现证据在 `.pi/w4a4-independent-20261023/`，同类 PR 检索与 Ready 操作证据在 `.pi/w4a4-ready-20261023/`。

服务器仍沿用 `llm/qwen3.5_moe_online_w4a8.sh` 与 `diffusion/i2v_offline_mxfp8.sh`，手动选择对应分支；后者设置 `SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION=1` 且不显式指定 attention backend，必须看到 `Using MXFP8 quantized Ascend Flash Attention.` 并完成视频生成，不能把 fallback 后生成成功算成自动 MXFP8 FA 验收。MoE 需确认 W13/W2 packed 准备、加载/warmup、文本和图片请求；版本/依赖以服务器实际记录为准，尤其 Qwen-VL 的 Transformers 5.19.0 契约。

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

### 当前 PR 跟踪（2026-10-22 全表；10-23 更新 #32602）

SGLang #32745 与 kernel #638 / #808 / #824 已合入并移出待合入队列，本计划剩余 **9 个活跃 SGLang PR**，其中 #43740 / #44366 为 Draft，#32602 已于 10-23 转为 Ready for review。保留原计划序号；序号不是合入顺序。所有 Open PR 均 `base=main`、`reviewDecision=REVIEW_REQUIRED`；10-22 的 #43740 仍 `CONFLICTING`，#32602 已于 10-23 独立化后复核为 `MERGEABLE / BLOCKED`；其余七项 10-22 无冲突，#43739 / #45550 当时最终复核均为 `MERGEABLE / BLOCKED`。无冲突不等于满足合入条件，CI 跳过不等于硬件通过；本表不是自动监控面板。

| 序号 | PR | 职责 | 状态 / Git 合并检查 | 当前 CI 与下一步 |
| --- | --- | --- | --- | --- |
| 1 | [kernel #638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | Gemma 稳定 API / 构建期 provider；G4 | Merged，2026-10-16 | 合入前检查通过，1 项跳过；#32745 的 kernel 源码前置已满足，部署仍需核对 target wheel |
| 2 | [SGLang #32745](https://github.com/sgl-project/sglang/pull/32745) | Gemma API、MXFP8 partial-scale / TP 对齐；G1.1/G4 | Merged，2026-10-21 | 已移出队列；#43739 / #43740 的源码前置满足，下一步各自同步包含正式合入的 main |
| 3 | [SGLang #32041](https://github.com/sgl-project/sglang/pull/32041) | Wan2.2 MXFP8 Attention 历史实现 | Closed，未合并，2026-10-06 | 维护者明确因核心功能由 #39438 替代而关闭；剩余自动 FA 路由由 #45550 延续，不等于 G2.2 去重 |
| 4 | [SGLang #40419](https://github.com/sgl-project/sglang/pull/40419) | speculative SSM/intermediate state BF16 | Open，非 Draft；MERGEABLE | lint 通过；缺 `run-ci` 阻断 CI；与已合入 kernel #808 配套，可独立推进 CI 与 review |
| 5 | [kernel #808](https://github.com/sgl-project/sgl-kernel-npu/pull/808) | Ascend 950 recurrent GDN / BF16 断言 | Merged，2026-09-29 | 已移出队列，不在原 PR 继续追加 |
| 6 | [SGLang #43739](https://github.com/sgl-project/sglang/pull/43739)（原 #32601） | 在线 MoE W4A8，复用 main GMM；G1.2 | Open，非 Draft；更新后无冲突 | 今日选定：已同步 main、去重至 7 文件并更新标题/正文；新标题检查与 lint 通过；新 gate 日志明确缺 `run-ci`。本地 46 项隔离 CPU 回归通过，等待用户真机与 review |
| 7 | [SGLang #32602](https://github.com/sgl-project/sglang/pull/32602) | 在线 MoE W4A4 / 融合 GMM；G1.2 | Open，非 Draft；MERGEABLE / BLOCKED（10-23） | 独立 8 文件，不再等待 #43739；按用户授权转为 Ready for review。32 项隔离 CPU/mock、GitHub lint 与最新标题检查通过。历史 Draft gate 失败不代表转 Ready 后已跑硬件 CI，仍缺 `run-ci`；等待当前 head 真机与 review |
| 8 | [SGLang #43740](https://github.com/sgl-project/sglang/pull/43740)（原 #34387） | Ascend 950 FIA mixed split | Open，Draft；CONFLICTING | 同步含 #32745 的 main 并保留 FIA 增量；旧 head lint 通过，Draft gate 阻断、缺 `run-ci`；可与两个 MoE PR 并行 |
| 9 | [kernel #824](https://github.com/sgl-project/sgl-kernel-npu/pull/824) | cache location view 边界修复 | Merged，2026-10-16 | 合入前检查通过，1 项跳过；已移出队列，后续部署核对 wheel 包含此修复 |
| 10 | [SGLang #44363](https://github.com/sgl-project/sglang/pull/44363) | G2.1 公共 W4A8 执行 | Open，非 Draft；MERGEABLE | lint / 标题通过；缺 `run-ci` 阻断 CI，可独立推进 CI 与 review |
| 11 | [SGLang #44364](https://github.com/sgl-project/sglang/pull/44364) | Qwen3.5 Dense ModelSlim 混精 | Open，非 Draft；MERGEABLE | lint / 标题通过；缺 `run-ci` 阻断 CI，可独立推进 CI 与 review |
| 12 | [SGLang #44365](https://github.com/sgl-project/sglang/pull/44365) | G2.2 Diffusion MXFP8 复用 | Open，非 Draft；MERGEABLE | 已有 `run-ci`，lint / 标题通过；本地落后 1 个维护者 CI 修复提交，合并当前 main 预演无冲突。CI 含实际失败与级联跳过，选取日志结论见下；不在今日两个 PR 范围 |
| 13 | [SGLang #44366](https://github.com/sgl-project/sglang/pull/44366) | G2.3 能力描述与 preflight | Open，Draft；MERGEABLE | lint / 标题通过；Draft gate 阻断、缺 `run-ci`；先合入 #44365 再同步，不能继承前置 PR 的 CI 结果 |
| 14 | [SGLang #45550](https://github.com/sgl-project/sglang/pull/45550)（延续 #32041） | Diffusion MXFP8 自动选择 FA | Open，非 Draft；更新后无冲突 | 今日选定：已同步 main，仍 3 文件；标题检查与 lint 通过；新 gate 明确缺 `run-ci`。10 项 CPU 回归通过，等待实际自动路由 + I2V 生成证据及 review |

#32745 于 2026-10-21 14:50:39（Asia/Shanghai）squash 合入，正式提交及 main 包含关系见 [branches.md](branches.md#current-branch-map)。旧基线、标题和 review 阻塞不再列为待办；合入不会把历史 CI 失败或未测配置改记为通过，模型验证边界仍见 G1.1/G5。

#44363–#44366 的原始真机反馈与新 head CPU 验证在第 3 节分开记录。2026-10-22 选取 #44365 的日志确认：NPU z-image 的 TextEncodingStage 约 595–599 ms 超过 442.566 ms 阈值；CPU 缺 `sgl_kernel.shm_group_initialize`；一个 AMD job 的 moss 两卡子测试报无 CUDA GPU，另有 Kandinsky offload / Qwen encoder 单测失败；部分 GPU job 被 fail-fast 跳过，coverage 缺测试。尚未完成全量归因或同 main 对照，不能概括为全是环境问题或本改动回归。其本地/远端 head 差异见 [branches.md](branches.md#current-branch-map)。本轮只更新选定 #43739/#45550 的分支、标题/正文，由 push 自动触发检查，未改 `run-ci` 标签或手动重跑；没有新增 wheel 或 NPU 验收结论。审计在 `.pi/pr-status-20261022/`，发布证据在 `.pi/pr-ready-20261022/`；这些本地路径不是上游可访问报告。

#45550 按用户确认仅延续 #32041 的自动路由增量，正文按实际范围重写并链接[维护者关闭说明](https://github.com/sgl-project/sglang/pull/32041#issuecomment-6008020716)，保留原作者 co-author。仅当无已选后端 / layer default 且当前 MXFP8 resolver 支持时自动选 FA，保留显式选择、模型后端约束及交叉注意力的原自动路径。隔离 CPU 测试 10 项通过；同一构造回归在未修改 main 的三种 self-attention 层均复现 SDPA 误选。真实 selector / resolver 与构造函数体配合导入隔离和 NPU capability mock，未完成完整运行环境收集或 NPU 推理 / 精度 / 性能验证；Ruff、格式、isort、语法与 diff 检查通过。版本与发布证据见 [branches.md](branches.md#current-branch-map)，本地证据在 `.pi/pr32041-continuation/`。

### 配套与替代关系

| PR | 2026-10-21 状态 | 作用 |
| --- | --- | --- |
| [SGLang #36426](https://github.com/sgl-project/sglang/pull/36426) | Open，非 Draft；MERGEABLE，CI 有失败，仍需 review | GDN 名字解析/视觉塔 quant config；当前叠加最新 MoE W4A8 作独立组合验证，旧 Dense 组合已清理。不计入上述 9 个活跃 PR，不混入正式功能栈 |
| [kernel #742](https://github.com/sgl-project/sgl-kernel-npu/pull/742) / [#651](https://github.com/sgl-project/sgl-kernel-npu/pull/651) | 分别 Open Draft / Open 非 Draft；均 CONFLICTING | 历史 MTP state 迁移配套，含 UB block 调整；未纳入本轮活跃队列，恢复时核对当前覆盖与实际部署，不当作已合入 #808 的待合入前置 |
| [kernel #747](https://github.com/sgl-project/sgl-kernel-npu/pull/747) / [SGLang #39589](https://github.com/sgl-project/sglang/pull/39589) | 均 Merged | SSM layout 配套；部署 wheel 与 SGLang 必须匹配 |
| [SGLang #39438](https://github.com/sgl-project/sglang/pull/39438) | Merged | Wan2.2 MXFP8 Attention，替代 #32041 的核心功能 |
| [SGLang #43309](https://github.com/sgl-project/sglang/pull/43309) | Merged | resize 兼容修复，已核对等价，不重复提交 |
| [SGLang #32266](https://github.com/sgl-project/sglang/pull/32266) / [kernel #823](https://github.com/sgl-project/sgl-kernel-npu/pull/823) | 均 Closed，未合并 | 历史实现与对照来源，移出队列 |

<a id="merge-dependencies"></a>

### 合入顺序与并行范围

```text
已合入前置：kernel #638 / #808 / #824、SGLang #32745

MoE 并行：    main ──→ #43739 W4A8
                  └─→ #32602 W4A4（10-23 已独立并转 Ready）
Diffusion 链：#44365 MXFP8 复用 ──→ #44366 能力校验（Draft）

独立推进：#43740 FIA mixed split（Draft；原 #32745 前置已满足）
          #40419 SSM dtype（kernel #808 已合入）
          #44363 Dense W4A8 公共执行
          #44364 ModelSlim gate/up 混精
          #45550 MXFP8 自动 FA 路由（#32041 续作；#39438 已合入）
```

- **只剩 Diffusion 一条线性依赖链：#44365 → #44366。** #32602 已解除对 #43739 的源码/合入依赖，可独立审阅和验证；当前可并行推进的八个入口是 #43739、#32602、#44365、#43740、#40419、#44363、#44364、#45550。Draft 状态、CI 和真机验收仍分别满足；#44366 可提前审阅，正式合入仍等待 #44365。
- **2026-10-22 用户选定 #43739 与 #45550 优先完成运行 ready**：两者已同步 main、普通推送，本地回归通过，接下来由用户真机执行并回传证据，再向 committer 请求 CI/review。#44365 仍值得优先跟进以解锁 #44366，但不混入本轮自动 FA 分支。#44363 / #44364 / #40419 可独立推进；#43740 仍需同步 main、解冲突与 Draft 就绪检查。
- #638 / #824 与 #32745 均已合入，正式合入提交见 [branches.md](branches.md#current-branch-map)。部署仍需包含 #638 provider 的配套 target wheel；源码合入不等于 wheel 已发布、安装或完成目标硬件验收。#734 的重叠实现已包含在 #638，不重复叠加。
- #43739 与独立 #32602 均已从 main 承接 #32745 正式合入内容；不再要求 W4A8 → W4A4 顺序。#32602 历史上包含旧 W4A8 提交，但净增量已经去除其在线功能；普通 merge/push 保留历史不等于保留功能依赖。后续两 PR 合入时核对公共 `quant.py` 的等价小改动并去重，不复制量化实现。继续保留 main 的 scale helper、激活/clamp、DeepEP 契约与单级/双级区别；#43740 同样直接从 main 承接已合入内容，无需等待任一 MoE PR。
- #40419 保留 main 的 GDN backend/prefill，仅维护自己的 dtype 契约；它与 #44363 / #44364 / #44365 均可独立推进，无需等待整个 MoE/FIA 栈。#36426 保持组合验证用途。
- #44365 先合入，再同步 #44366；能力增量可相对前置分支审阅。#44363 与 #44364 可独立审阅，但组合运行仍需核对实际配置与代码版本。
- #45550 直接基于 main，前置 #39438 已合入；没有未合入代码前置，可与 G2 两个 PR 及 MoE/FIA 并行。后续若 G2 修改 MXFP8 resolver，组合验证仍需核对该接口契约。
- 操作前刷新 main、PR head、worktree 与脏文件。全部正式 PR 保持 `base=main`，通过普通 merge/追加提交和 push 更新；未经授权不强推。前置合入后核对内容并清理重复增量，代码在各自直属 worktree 修改。

## 6. 后续维护与暂缓项

每次只更新目标表、PR 表和分支定位中发生变化的部分，保留足以复核的 `日期 / SHA / 场景 / 证据 / 未测范围 / 下一步`。不再追加逐次命令、CI 计数或重复“尚未发布”的过程流水；完整历史用 Git 追溯。

- **G2**：原分支功能验收已结束，四个 PR 已发布。后续依据 review、实际 CI 失败或新模型日志继续；用户跳过的数值/基线比较不重新设为前置。
- **NPU Qwen-VL 预处理整理**：2026-10-13 用户要求暂缓，尚未恢复。后续如授权，评估复用 Transformers 分组、resize、rescale/normalize，仅保留 NPU 必要 patch/layout，图片与视频一起覆盖；不要把已合入的 #43309 再提一遍。
- **证据更新**：新 head 的测试、日志、视频反馈各自绑定版本；用户反馈标明来源，不能写成本机 NPU 已执行或完整数值等价。

接手先读 AGENTS.md、本文件的目标/PR 表和 [branches.md](branches.md#current-branch-map)；按需复跑入口见 [g23-handoff](#g23-handoff)。默认单 agent，按当次授权提交发布并保留不相关改动。
