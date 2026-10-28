# 活跃分支、PR 与 worktree

本文维护 branch / worktree / HEAD / remote 定位和必要的发布证据。当前 PR 状态与合入顺序只维护在[重构计划](mxfp-refactoring-plan.md#pr-merge-status)，模型验收只维护在[目标表与摘要](mxfp-refactoring-plan.md#refactoring-status)。

<a id="current-branch-map"></a>

## 当前分支定位（2026-10-22 全表；10-23 更新 W4A4；10-28 更新 W4A8 路由对照）

2026-10-22 全表核对真实 worktree 注册表、本地 HEAD、remote 和 GitHub PR head；10-23 单独更新 #32602，并确认 W4A8 正式/组合分支未变。表中 HEAD 均为本地 checkout；#44365 与 GitHub 的差异、#32745 与 kernel 的正式合入提交见表后说明。原验收分支与新 PR 发布分支分别保留；当前状态会漂移，操作前仍须实时查询。

本轮按用户选择推进 #43739 与 #45550，分别在原 worktree 普通 merge / push 同步 main；`upstream-main` 只读参考同步到同一基线。随后按用户授权清理旧分支/worktree，并将 #36426 组合验证载体从 Dense 改为最新 MoE W4A8；正式 #43739 分支未叠加该补丁。#32745 主 clone、原 G2 验收分支、旧 #32266、LoRA 和 repack 工具保留；kernel 主 checkout / 主仓 gitlink 未移动。

| PR / 任务 | 分支 | 本地 HEAD | worktree（相对主仓） |
| --- | --- | --- | --- |
| [kernel #638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | `codex/a5-gemma-rmsnorm-csrc` | `4a4f838e8f` | `sgl-kernel-npu/` |
| [SGLang #32745](https://github.com/sgl-project/sglang/pull/32745) | `junlin_qwen3.5_dense_w8a8` | `c89095b6e5` | `sglang/qwen3.5_dense_w8a8/` |
| [SGLang #40419](https://github.com/sgl-project/sglang/pull/40419) | `junlin_npu_mamba_ssm_dtype` | `756dac5a66` | `sglang/npu_mamba_ssm_dtype/` |
| [SGLang #43739](https://github.com/sgl-project/sglang/pull/43739)（原 [#32601](https://github.com/sgl-project/sglang/pull/32601)） | `junlin_qwen3.5_moe_w4a8` | `bacf338c55` | `sglang/qwen3.5_moe_w4a8/` |
| NPU Mamba 清零验证（基于 #43739，无 PR） | `junlin/npu-mamba-clear-index-copy` | `d38039ef6e` | `sglang/npu_mamba_clear_index_copy/` |
| [SGLang #32602](https://github.com/sgl-project/sglang/pull/32602) | `junlin_qwen3.5_moe_w4a4` | `9c40660a60` | `sglang/qwen3.5_moe_w4a4/` |
| [SGLang #43740](https://github.com/sgl-project/sglang/pull/43740)（原 [#34387](https://github.com/sgl-project/sglang/pull/34387)） | `junlin_a5_fia_mixed_split` | `01bb8dc685` | `sglang/a5_fia_mixed_split/` |
| G2.1：Dense W4A8 公共执行 | `junlin/w4a8-shared-execution` | `24e100f1e8` | `sglang/w4a8_shared_execution/` |
| G2.2：Diffusion MXFP8 公共执行 | `junlin/diffusion-mxfp8-shared-execution` | `6eb8e1dd1c` | `sglang/diffusion_mxfp8_shared_execution/` |
| G2.3：Ascend MXFP 能力校验 | `junlin/ascend-mxfp-capabilities` | `d6a46f90d9` | `sglang/ascend_mxfp_capabilities/` |
| Wan repack rotation / 量化元数据兼容 | `junlin/wan-repack-rotation-keys` | `1916bc208b` | `sglang/wan_repack_rotation_keys/` |
| ModelSlim Dense gate/up 混合精度与预处理兼容 | `junlin/modelslim-mixed-gate-up` | `2ef4c276be` | `sglang/modelslim_mixed_gate_up/` |
| [SGLang #44363](https://github.com/sgl-project/sglang/pull/44363)：G2.1 发布 | `junlin/pr-w4a8-shared-execution` | `4f998c97fd` | `sglang/pr_w4a8_shared_execution/` |
| [SGLang #44364](https://github.com/sgl-project/sglang/pull/44364)：ModelSlim gate/up 混精发布 | `junlin/pr-modelslim-mixed-gate-up` | `ec1383b441` | `sglang/pr_modelslim_mixed_gate_up/` |
| [SGLang #44365](https://github.com/sgl-project/sglang/pull/44365)：G2.2 发布 | `junlin/pr-diffusion-mxfp8-shared-execution` | `4cfdf3d543` | `sglang/pr_diffusion_mxfp8_shared_execution/` |
| [SGLang #44366](https://github.com/sgl-project/sglang/pull/44366)：G2.3 发布 | `junlin/pr-ascend-mxfp-capabilities` | `9f9aa27254` | `sglang/pr_ascend_mxfp_capabilities/` |
| [SGLang #45550](https://github.com/sgl-project/sglang/pull/45550)：#32041 自动 FA 路由续作 | `junlin/diffusion-mxfp8-fa-auto-selection` | `ceace1a8d3` | `sglang/diffusion_mxfp8_fa_auto_selection/` |
| #43739 + #36426 组合验证（无 PR） | `junlin_qwen3.5_moe_w4a8_pr36426` | `00e216bb88` | `sglang/qwen3.5_moe_w4a8_pr36426/` |
| 旧 ModelSlim / 视觉塔分支（#32266） | `junlin_qwen3.5_moe_w8a8` | `3d72a9dc00` | `sglang/qwen3.5_moe_w8a8/` |
| MoE LoRA 实验（无 PR） | `junlin-ascend-moe-lora` | `bc9a4d0753` | `sglang/ascend_moe_lora/` |
| 本轮 main 只读参考（非自动跟随） | `upstream-main` | `3d508d614f` | `sglang/upstream-main/` |

- `sglang/qwen3.5_dense_w8a8/` 是共享 `.git` 的主 clone，其余 SGLang checkout 为直属 worktree。`sglang/` 被主仓忽略，不是子模块；不得在已有 worktree 内切换功能分支。
- `sgl-kernel-npu/` 是主仓子模块。主仓 gitlink 仍为 `500ea7bb0e`，checkout 为上表 #638 head；两者差异是保留的既有状态，本轮文档更新未调整 checkout 或主仓 gitlink。
- #32745 于 2026-10-21 14:50:39（Asia/Shanghai）由源 head `c89095b6e5` squash 为 [`844d2b5dec`](https://github.com/sgl-project/sglang/commit/844d2b5decf6257aa192744cbdaaffb8ad797263)。合入后首轮 GitHub 查询时官方 `main` 即此提交，compare 为 `identical`；后续 #45550 使用重新 fetch 的 main，基线见下。源分支与正式 squash 提交分别保留；下游同步 main 后按内容核对并去除重复增量，不再将原源 head 作为待合入前置。
- 2026-10-22：#43739 与 #45550 共同同步 `upstream/main@3d508d614f28d2c35da2fdafe422c9bfe454c1b8`，普通 merge / push 后本地、origin 与 GitHub head 一致，ahead/behind 均 `0 0`。#43739 为 `493372ce8fee4d8f69dd3d482cf711d840ce2bd7`：冲突采用 main 的已合入 Gemma/MXFP8 实现，并去掉旧 registry 及无关 ModelSlim GDN/视觉塔增量，最终相对 main 为 7 文件；#45550 为 `ceace1a8d392efb0f3719c3a405e6470d4d5ea59`，仍仅 3 文件自动 FA 路由增量，未叠加 #44365。验证与待真机范围见[计划](mxfp-refactoring-plan.md#refactoring-status)，证据在 `.pi/pr-ready-20261022/`。
- 2026-10-23：按用户要求将 #32602 从 W4A8 功能栈拆出。原 worktree 普通 merge `upstream/main@904691e0b5dc4516c5bc9771874d03dd0cadb470`，去掉继承的在线 W4A8、旧 Gemma 和无关 ModelSlim 增量，独立 W4A4 相对 main 仅 8 文件。新 head `9c40660a609dab64476f72d0d8f939eeadecfdaa` 已普通推送，本地/origin/GitHub SHA 一致，ahead/behind `0 0`；历史提交保留但不再构成源码/合入前置，未强推。#43739 `493372ce8f` 及其 #36426 组合 `52809fabf1` 的 HEAD 和工作区均未变；未新建 PR/worktree。验证与依赖见[计划](mxfp-refactoring-plan.md#refactoring-status)，证据在 `.pi/w4a4-independent-20261023/`。
- 2026-10-28：#43739 在原分支增加普通 TP 路由量化对照开关 `SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT`，默认保持融合，仅显式 `0` 使用路由后量化。正式 head `bacf338c556f92db35460c6d2f68fc205783cb28`；组合分支同轮普通 merge/push 至 `00e216bb88bdc0562071fef44b3a4daef0b35119`，本地与 GitHub SHA 一致。两分支各通过 17 项隔离 CPU 契约测试，尚未执行新开关的 NPU 服务 A/B 或模型精度验收；命令见 [LLM 诊断说明](../llm/README.md#1b-同版本-w4a8-服务路由-ab)，证据在 `.pi/w4a8-route-serving-20261028/`。未同步新 main，未发布性能结果。
- #45550 初次发布基于 `f7b8c6fe40`，功能提交 `79dba73be8` 仅延续 #32041 的自动 FA 选择，未复制已由 #39438 合入的 kernel。原作者 `hxhhhlalala <hyh_hh@163.com>` 的 co-author 保留；原 PR、署名与初次隔离测试证据在 `.pi/pr32041-continuation/`。
- kernel 正式合入证据：#638 于 2026-10-16 15:16:44（Asia/Shanghai）squash 为 [`684b4c4036`](https://github.com/sgl-project/sgl-kernel-npu/commit/684b4c40361d38dd60fd888a5f5e2e224cab0c7a)，#824 于 15:16:53 squash 为 [`e1ef758b22`](https://github.com/sgl-project/sgl-kernel-npu/commit/e1ef758b2262f0b3edd886306a1951d457bafd00)。2026-10-16 查询时官方 `main` 为后者，GitHub compare 确认包含前者；2026-10-21 复核两项合入记录不变。主子模块仍保留 #638 源 head；#824 源分支/worktree 已清理，正式 squash 提交仍在 main。PR 状态和下一步见[合入计划](mxfp-refactoring-plan.md#pr-merge-status)。
- 已清理的 kernel `main-638-824@e629e45a51` 是 2026-10-09 固定组合，未跟随后续 PR head。后续构建使用包含正式合入的官方 main，而非该旧组合；历史 26 项 CPU provider/build-target 测试不等于 wheel 或 NPU 通过。构建说明见 [kernel 文档](sgl-kernel-npu-build.md)。
- #44365 的 GitHub 当前 head 为 `06f27b4dda`，本地 `4cfdf3d543` 落后 1 个提交；已核对为维护者移植的 #44380 CI fake 修复，普通合并当前 main 的预演无冲突，去重后只余 2 文件功能增量。本轮未移动此 worktree。其他活跃 PR 的本地 HEAD 与 GitHub head 一致。
- 2026-10-22 清理：删除 kernel `cache-loc-assign-bounds@1c6fad69a1`、`main-638-824@e629e45a51` 的两个 worktree/本地分支；前者远端此前已删除，本轮删除后者远端。清理后 kernel fork 仅余 `main`（当时 `84ce0315e4`，随后同步上游见下），官方 `upstream/main@a5c12d19fe` 已包含 #638 / #808 / #824；主子模块 #638 源分支保留，过期 tracking refs 已 prune。另删除 `sglang/g2_validation_baseline/`（detached `62ff466ac0`）及旧 `sglang/qwen3.5_dense_w8a8_pr36426/`（`1593019ff9`），后者本地/远端分支同名 `junlin_qwen3.5_dense_w8a8_pr36426` 一并删除。删除前均无未提交源码，仅有可再生缓存；未使用强制 worktree 删除。
- 同轮删除 9 条已合入、远端 tip 与 PR head 一致的 SGLang fork 分支：`junlin-remove-vendor-refs-comments`（#36768）、`junlin-remove-vllm-ascend-comments`（#34829）、`codex/fix-modelslim-mxfp4-packed-weight`（#32013）、`junlin_codeowners`（#31917）、`junlin_qwen3_moe_w8a8`（#30768）、`junlin_qwen3_dense_w8a8`（#28505）、`codex/diffusion-mxfp4-docs`（#25904）、`junlin_qwen3_dense_w4a4`（#23795）、`junlin_qwen3_dense_w4a8`（#23650）。九个正式 merge commit 均在 upstream/main；本轮未扩大到其余旧分支或备份。
- 新组合分支 `junlin_qwen3.5_moe_w4a8_pr36426@52809fabf1cafb662c43303d5790329db75ba929` 从最新 #43739 `493372ce8fee4d8f69dd3d482cf711d840ce2bd7` 创建，只叠加 #36426 `9ea4881cc32b2c43afbfd10795183752e5d33f22` 的 4 文件、+97/-43 补丁，patch-id 与实时 PR diff 完全一致，保留原作者 co-author。普通推送后本地/origin SHA 一致，ahead/behind `0 0`；正式 #43739 HEAD 不变，未新建 PR。后续 MoE 基础分支更新时一并同步、验证、推送组合分支。#36426 主要针对 ModelSlim 离线加载与视觉 INT8，尚无在线 W4A8 必须依赖它的证据。验证边界见[目标表](mxfp-refactoring-plan.md#refactoring-status)，清理与组合证据在 `.pi/cleanup-moe-combo-20261022/`；旧 Dense 组合验证记录 `.pi/pr32745-combo-sync-20261020/` 仅作历史，不再是运行入口。
- 2026-10-16 的 MXFP8 TP 修复先在 #32745 分支发布，再单独移植到组合验证分支，各只修改 scheme 与回归测试两个文件。前者读取 `layer.tp_group`，后者沿用 `layer.tp_rank`，未合并其他上游提交。两分支均普通推送，远端 SHA 与本地一致；验证边界见[目标表](mxfp-refactoring-plan.md#refactoring-status)。

## 远程地址

| 仓库 | `origin` | `upstream` |
| --- | --- | --- |
| 主仓 | <https://github.com/TallMessiWu/sglang_quant_eval.git> | 无 |
| SGLang | <https://github.com/TallMessiWu/sglang.git> | <https://github.com/sgl-project/sglang.git> |
| sgl-kernel-npu | <https://github.com/TallMessiWu/sgl-kernel-npu.git> | <https://github.com/sgl-project/sgl-kernel-npu.git> |

主仓 `.gitmodules` 的 kernel URL 指向开发 fork；子模块保留 `upstream` 作为官方同步来源。

**2026-10-22 fork main 同步**：用户明确 `main` 只作上游镜像，不在其上开发。SGLang `origin/main` 原为 `81beaa4fb6`，有 3 个 fork 独有旧提交；用户确认丢弃后，以精确旧 SHA 的 `--force-with-lease` 对齐 `upstream/main@780a5b1accd16788d9651f170e787c6c81c66f4f`，未新建归档分支。kernel `origin/main` 从 `84ce0315e4` 正常快进 10 个提交至 `upstream/main@a5c12d19fe12dc03aca474852f63cf49f5df8eb2`，未被 checkout 占用的本地 `main` 同步快进。两仓远端 main SHA 均与对应上游一致，ahead/behind `0 0`。所有功能分支、worktree HEAD 和主仓 gitlink 未变；SGLang 的 `sglang/upstream-main/` 仍保留本轮测试基线 `3d508d614f`，不自动跟随新远端 main。证据在 `.pi/fork-main-sync-20261022/`；本次只同步 Git refs，没有新增功能或硬件验证。

## G2 发布来源与验证边界

- 原 G2.1 `24e100f1e8` 基于 #43739 的 `ded4671577`；混精提交 `185fad010a` 叠在 G2.1 上，之后 `2ef4c276be` 回移已合入的 resize 修复。原 G2.2 `6eb8e1dd1c` 与 G2.3 `d6a46f90d9` 均独立基于 `62ff466ac0`。
- 2026-10-15 按用户授权从实时 `upstream/main@9fa9308e952398ee235c5b43dfda7025e91d5f92` 提取最小增量：#44363 为公共 W4A8 执行及必要的 BF16 `[1,N]` bias 契约；#44364 为四文件混精修复；#44365 为两文件 Diffusion adapter 与测试。三者无需整个 Gemma / MoE / FIA 功能栈。
- #44366 叠加 #44365 的原发布提交 `4cfdf3d543`，同步 Diffusion partial-scale 校验、回归与文档导航；能力增量相对该前置为 13 文件。后续 `9f9aa27254` 仅将 MDX 表格标记改为合法注释，并同步测试中的标记；它不代表已同步 #44365 的当前远端 head。
- **混精去重**：发布前核对 main 的合并 gate/up 结构及两条 loader，并检索相关 PR，未发现等价 ModelSlim Dense 异构拆分；#38870 修的是 Quark MTP 排除列表。**resize 去重**：#43309 已合入，main 与回移提交对应文件的 blob 同为 `c5c9ea5c24180459e8faf16623fac26d13242a85`，patch-id 也相同，四个新 PR 均不重复携带该 diff。
- 2026-10-15 初次发布及 MDX 修正时，四条发布分支均普通 push，本地、origin 与 GitHub head 一致，tracking ahead/behind `0 0`。当前 #44365 的差异见上；其余原验收分支与服务器环境未在本轮修改。
- 初次发布隔离 CPU 检查：G2.1 6 项、混精 7 项、G2.2 3 项、G2.3 15 项，以及 G2.3 组合分支上的 3 项 adapter 测试通过。真实目标模块/函数体配合 serving/hardware 隔离与 NPU mock；标准入口因缺 `orjson` 未完成。2026-10-15 MDX 修正后再跑 15 项能力检查通过。
- 本地 Ruff/isort/codespell、语法和 diff 检查通过，G2.3 另有 JSON/CI 注册检查；CI 结果见唯一 PR 表。本轮未执行模型复跑，新发布 head 尚无同场景真机通过证据；原分支的用户功能反馈不能继承为新 head、数值等价或全部能力覆盖。

证据目录：`.pi/pr-status-20261022/` 保存最新 PR、远端分支、worktree、合并预演及选取的 CI 失败日志；`.pi/pr-ready-20261022/` 保存两个选定 PR 的同步、检查与发布证据。`.pi/pr-status-20261021/` 保存上轮 PR、门禁日志、#32745 正式合入和 main 包含关系及本地 worktree 审计；`.pi/pr-status-20261016/` 保留 kernel 合入时的快照。`.pi/g2-pr-20261015/` 保存初次测试日志、PR 正文及 SHA 核对；`.pi/docs-refresh-20261015/` 保存当时 GitHub 状态、MDX 错误日志与修正后的 CPU 结果。它们是本地辅助证据，不是上游可访问的测试报告。

<a id="g2-runtime-record"></a>

## G2 运行载体与接手

当前验收结论及未测范围见[计划摘要](mxfp-refactoring-plan.md#refactoring-status)，直接运行命令见 [G2.3 接手入口](mxfp-refactoring-plan.md#g23-handoff)。不再将已成功的旧分支当作首次待运行任务，也不自动用新 PR 分支替换用户部署。

- Wan repack 工具 `1916bc208b` 修正 rotation/量化描述键映射，并提供 `--quant-description-file`，拒绝多候选时静默选择全 FLOAT 描述；7 项隔离 CPU 文件回归通过。工具与运行分支独立。
- 2026-10-14 用户使用该工具完成现有 Wan2.2 I2V checkpoint 转换。两套 transformer 各保留 400 个 MXFP8 weight/scale、207 个 FLOAT weight、80 个 rotation；元数据检查不等于逐字节或数值比较。旧服务器日志为 `/tmp/g2-repack-FYhE4Mpr/run.log`，仅作历史定位，后续输出统一留在执行目录下。
- 主仓 `be39df1` 新增 `diffusion/i2v_offline_mxfp8.sh`，`a3e81f1` 将高度调整为 720；`e5ea39a5aa` 发布 Dense 在线/离线 W4A8 普通脚本。脚本不自动管理分支、worktree 或 PYTHONPATH。
- Dense 混精/graph 诊断最终以 `2ef4c276be` 用户复测正常收敛：保留 BF16 SSM、去掉禁 graph 参数并停止重启服务。旧异常根因未证实，没有新增生产 graph 修复；机制与禁止误判事项见 [known-pitfalls.md](known-pitfalls.md)。

## 保留的历史结论

这里只保留影响后续判断的结果。逐次 merge、临时冲突、过期 CI 计数和已删除 worktree 的操作过程可从 Git 历史追溯，例如 `git show fa52bfa:docs/branches.md`；不再保留重复的“当前状态”快照。

| 日期 | 结论与后续意义 |
| --- | --- |
| 2026-08-25 | 在线 MoE W4A8/W4A4 分支删除与已合入 #30318/#30319 重复的离线 scheme；后续复用已有 kernel，不重复注册同名类。非专家 Linear 的全 BF16 实验已撤销，精度隔离使用 `ignored_layers`。 |
| 2026-09-17 | #734 的 provider staging 已并入 #638；`causal_conv1d` 放开由上游 #802 覆盖，重复补丁已移除。Ascend 950 使用原生 kernels target；具体构建规则见 kernel 文档。 |
| 2026-09-18 至 20 | #36426 组合曾修复 GDN 名字解析及视觉塔 quant config，用户确认 MXFP8 离线输出恢复正常。历史 graph rebind 补丁已在后续同步时让位上游实现，不是当前 #32745 的新增 diff，也不能用于解释所有后续 graph 异常。 |
| 2026-09-20 | MTP 联调区分了 BF16 state、跨仓 `(HV,V,K)` 布局和 gating/prefill 接口问题；部署组合需匹配，不能凭正方形 state 的 shape 判断布局。机制保存在 [MTP 契约](known-pitfalls.md#npu-mtp-state-contract)。 |
| 2026-09-24 | #808 与只放开注册的 #823 在 BF16 矩阵中均过容差；#808 的 dtype 断言拒绝 FP32 state，#823 会静默产生极大值/NaN。#808 微基准快 1.4–4.6×，不是 e2e 收益；决定保留整份 #808，关闭 #823。旧实验组合还混有 cache-loc 差异，不能当作纯 kernel A/B。 |
| 2026-09-29 | kernel #808 squash 为 `9e0891b24a`，7 个文件与原 head `c524e45eff` 相同；原 head 非 main 祖先不代表实现未合入。后续仍需 SGLang #40419 的 BF16 分配配套。 |
| 2026-10-09 | 用户授权删除旧 kernel 实验分支/worktree；#808/#823 的旧目录不再作为构建入口。现存目录以上表及实时注册表为准。 |
| 2026-10-10 | 七个 Ascend 功能分支按依赖普通 merge/push 同步 main；#40419 收敛为 SSM dtype 分配，保留 main 的 GDN backend/prefill。CPU/静态验证不替代更新后 NPU 验证。 |
| 2026-10-12 | #32601/#34387 关闭后改用 #43739/#43740，沿用原分支，旧讨论不迁移为新 PR 的验证结果。G2 实现及脚本随后独立发布。 |
| 2026-10-14 至 15 | G2.1 在线/离线及 G2.2/G2.3 I2V 获用户功能通过反馈；随后以独立发布分支创建四个 PR，见上表和目标跟踪。 |

历史单独候选包括 `sparse_attention_score` 的 Ascend 950 适配；恢复该项前需核对当前上游覆盖并取得对应任务授权，不自动混入 #638 或 G2。

## 操作前核对

```bash
git -C sglang/qwen3.5_dense_w8a8 worktree list --porcelain
git -C sglang/<worktree> status --short --branch
git -C sglang/<worktree> remote -v
git -C sglang/<worktree> rev-parse HEAD
git -C sgl-kernel-npu worktree list --porcelain
gh pr view <number> --repo sgl-project/sglang \
  --json state,isDraft,headRefOid,mergeable,reviewDecision,statusCheckRollup
```

只在任务需要新分支且确认未被占用时创建直属 worktree：

```bash
git -C sglang/qwen3.5_dense_w8a8 worktree add ../<worktree-name> -b <branch> <verified-base>
```

已有分支用 `worktree add ../<worktree-name> <branch>`。提交依赖用 `git merge-base --is-ancestor` 核实；跨仓 API/wheel 依赖不能用 Git 祖先关系代替验证。所有正式 PR 以 main 为 base，依赖和更新顺序见[合入计划](mxfp-refactoring-plan.md#merge-dependencies)。
