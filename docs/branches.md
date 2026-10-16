# 活跃分支、PR 与 worktree

本文维护 branch / worktree / HEAD / remote 定位和必要的发布证据。当前 PR 状态与合入顺序只维护在[重构计划](mxfp-refactoring-plan.md#pr-merge-status)，模型验收只维护在[目标表与摘要](mxfp-refactoring-plan.md#refactoring-status)。

<a id="current-branch-map"></a>

## 当前分支定位

表中快照以 2026-10-15 的 worktree 注册表、本地 HEAD 和 GitHub PR head 核对为基础；2026-10-16 单独更新了 #32745 及其组合验证分支。原验收分支与新 PR 发布分支分别保留；当前状态会漂移，操作前仍须实时查询。

| PR / 任务 | 分支 | HEAD | worktree（相对主仓） |
| --- | --- | --- | --- |
| [kernel #638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | `codex/a5-gemma-rmsnorm-csrc` | `4a4f838e8f` | `sgl-kernel-npu/` |
| [SGLang #32745](https://github.com/sgl-project/sglang/pull/32745) | `junlin_qwen3.5_dense_w8a8` | `ef34aaead1` | `sglang/qwen3.5_dense_w8a8/` |
| [SGLang #40419](https://github.com/sgl-project/sglang/pull/40419) | `junlin_npu_mamba_ssm_dtype` | `756dac5a66` | `sglang/npu_mamba_ssm_dtype/` |
| [kernel #824](https://github.com/sgl-project/sgl-kernel-npu/pull/824) | `cache-loc-assign-bounds` | `1c6fad69a1` | `sgl-kernel-npu-worktrees/cache-loc-assign-bounds/` |
| [SGLang #43739](https://github.com/sgl-project/sglang/pull/43739)（原 [#32601](https://github.com/sgl-project/sglang/pull/32601)） | `junlin_qwen3.5_moe_w4a8` | `ded4671577` | `sglang/qwen3.5_moe_w4a8/` |
| [SGLang #32602](https://github.com/sgl-project/sglang/pull/32602) | `junlin_qwen3.5_moe_w4a4` | `425b0a9abb` | `sglang/qwen3.5_moe_w4a4/` |
| [SGLang #43740](https://github.com/sgl-project/sglang/pull/43740)（原 [#34387](https://github.com/sgl-project/sglang/pull/34387)） | `junlin_a5_fia_mixed_split` | `01bb8dc685` | `sglang/a5_fia_mixed_split/` |
| G2.1：Dense W4A8 公共执行 | `junlin/w4a8-shared-execution` | `24e100f1e8` | `sglang/w4a8_shared_execution/` |
| G2.2：Diffusion MXFP8 公共执行 | `junlin/diffusion-mxfp8-shared-execution` | `6eb8e1dd1c` | `sglang/diffusion_mxfp8_shared_execution/` |
| G2.3：Ascend MXFP 能力校验 | `junlin/ascend-mxfp-capabilities` | `d6a46f90d9` | `sglang/ascend_mxfp_capabilities/` |
| G2 历史只读基线（本轮不复跑） | detached HEAD（只读对照） | `62ff466ac0` | `sglang/g2_validation_baseline/` |
| Wan repack rotation / 量化元数据兼容 | `junlin/wan-repack-rotation-keys` | `1916bc208b` | `sglang/wan_repack_rotation_keys/` |
| ModelSlim Dense gate/up 混合精度与预处理兼容 | `junlin/modelslim-mixed-gate-up` | `2ef4c276be` | `sglang/modelslim_mixed_gate_up/` |
| [SGLang #44363](https://github.com/sgl-project/sglang/pull/44363)：G2.1 发布 | `junlin/pr-w4a8-shared-execution` | `4f998c97fd` | `sglang/pr_w4a8_shared_execution/` |
| [SGLang #44364](https://github.com/sgl-project/sglang/pull/44364)：ModelSlim gate/up 混精发布 | `junlin/pr-modelslim-mixed-gate-up` | `ec1383b441` | `sglang/pr_modelslim_mixed_gate_up/` |
| [SGLang #44365](https://github.com/sgl-project/sglang/pull/44365)：G2.2 发布 | `junlin/pr-diffusion-mxfp8-shared-execution` | `4cfdf3d543` | `sglang/pr_diffusion_mxfp8_shared_execution/` |
| [SGLang #44366](https://github.com/sgl-project/sglang/pull/44366)：G2.3 发布 | `junlin/pr-ascend-mxfp-capabilities` | `9f9aa27254` | `sglang/pr_ascend_mxfp_capabilities/` |
| kernel 编译组合（无 PR） | `main-638-824` | `e629e45a51` | `sgl-kernel-npu-worktrees/main-638-824/` |
| #32745 / #36426 / #40419 历史组合验证（无 PR） | `junlin_qwen3.5_dense_w8a8_pr36426` | `a5bbbe6b4d` | `sglang/qwen3.5_dense_w8a8_pr36426/` |
| 旧 ModelSlim / 视觉塔分支（#32266） | `junlin_qwen3.5_moe_w8a8` | `3d72a9dc00` | `sglang/qwen3.5_moe_w8a8/` |
| MoE LoRA 实验（无 PR） | `junlin-ascend-moe-lora` | `bc9a4d0753` | `sglang/ascend_moe_lora/` |
| 本地只读参考（不是实时远端 main） | `upstream-main` | `0dd66def7c` | `sglang/upstream-main/` |

- `sglang/qwen3.5_dense_w8a8/` 是共享 `.git` 的主 clone，其余 SGLang checkout 为直属 worktree。`sglang/` 被主仓忽略，不是子模块；不得在已有 worktree 内切换功能分支。
- `sgl-kernel-npu/` 是主仓子模块。主仓 gitlink 仍为 `500ea7bb0e`，checkout 为上表 #638 head；两者差异是保留的既有状态，不随本轮文档提交。
- kernel `main-638-824` 是 2026-10-09 的固定构建组合：`84ce0315e4` + #638 的 squash `cefb0f9d` + #824 的 squash `e629e45a`。它未跟随两个 PR 的后续 head 更新；26 项 CPU provider/build-target 测试通过不等于 wheel 构建或 NPU 通过。构建说明见 [kernel 文档](sgl-kernel-npu-build.md)。
- `junlin_qwen3.5_dense_w8a8_pr36426` 只用于组合验证，不能因它包含上游适配就把 #36426 混入正式功能栈。
- 2026-10-16 的 MXFP8 TP 修复先在 #32745 分支发布，再单独移植到组合验证分支，各只修改 scheme 与回归测试两个文件。前者读取 `layer.tp_group`，后者沿用 `layer.tp_rank`，未合并其他上游提交。两分支均普通推送，远端 SHA 与本地一致；验证边界见[目标表](mxfp-refactoring-plan.md#refactoring-status)。

## 远程地址

| 仓库 | `origin` | `upstream` |
| --- | --- | --- |
| 主仓 | <https://github.com/TallMessiWu/sglang_quant_eval.git> | 无 |
| SGLang | <https://github.com/TallMessiWu/sglang.git> | <https://github.com/sgl-project/sglang.git> |
| sgl-kernel-npu | <https://github.com/TallMessiWu/sgl-kernel-npu.git> | <https://github.com/sgl-project/sgl-kernel-npu.git> |

主仓 `.gitmodules` 的 kernel URL 指向开发 fork；子模块保留 `upstream` 作为官方同步来源。

## G2 发布来源与验证边界

- 原 G2.1 `24e100f1e8` 基于 #43739 的 `ded4671577`；混精提交 `185fad010a` 叠在 G2.1 上，之后 `2ef4c276be` 回移已合入的 resize 修复。原 G2.2 `6eb8e1dd1c` 与 G2.3 `d6a46f90d9` 均独立基于 `62ff466ac0`。
- 2026-10-15 按用户授权从实时 `upstream/main@9fa9308e952398ee235c5b43dfda7025e91d5f92` 提取最小增量：#44363 为公共 W4A8 执行及必要的 BF16 `[1,N]` bias 契约；#44364 为四文件混精修复；#44365 为两文件 Diffusion adapter 与测试。三者无需整个 Gemma / MoE / FIA 功能栈。
- #44366 叠加 #44365 的 `4cfdf3d543`，同步 Diffusion partial-scale 校验、回归与文档导航；能力增量相对该前置为 13 文件。后续 `9f9aa27254` 仅将 MDX 表格标记改为合法注释，并同步测试中的标记。
- **混精去重**：发布前核对 main 的合并 gate/up 结构及两条 loader，并检索相关 PR，未发现等价 ModelSlim Dense 异构拆分；#38870 修的是 Quark MTP 排除列表。**resize 去重**：#43309 已合入，main 与回移提交对应文件的 blob 同为 `c5c9ea5c24180459e8faf16623fac26d13242a85`，patch-id 也相同，四个新 PR 均不重复携带该 diff。
- 四条发布分支均普通 push，本地、origin 与 GitHub head 一致，tracking ahead/behind `0 0`。原验收分支与服务器环境未改动，没有强推。
- 初次发布隔离 CPU 检查：G2.1 6 项、混精 7 项、G2.2 3 项、G2.3 15 项，以及 G2.3 组合分支上的 3 项 adapter 测试通过。真实目标模块/函数体配合 serving/hardware 隔离与 NPU mock；标准入口因缺 `orjson` 未完成。本轮 MDX 修正后再跑 15 项能力检查通过。
- 本地 Ruff/isort/codespell、语法和 diff 检查通过，G2.3 另有 JSON/CI 注册检查；CI 结果见唯一 PR 表。本轮未执行模型复跑，新发布 head 尚无同场景真机通过证据；原分支的用户功能反馈不能继承为新 head、数值等价或全部能力覆盖。

证据目录：`.pi/g2-pr-20261015/` 保存初次测试日志、PR 正文及 SHA 核对；`.pi/docs-refresh-20261015/` 保存本轮 GitHub 状态、MDX 错误日志与修正后的 CPU 结果。它们是本地辅助证据，不是上游可访问的测试报告。

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
