# 分支与版本定位

更新：2026-10-30，本轮实时刷新 #43739、其 #36426 组合分支及 #32602；其余版本记录仍为 10-29 快照。PR 状态/依赖/验收只在 [目标计划](mxfp-refactoring-plan.md) 维护；本文记录版本和恢复入口，不记录逐次操作流水。

## 仓库关系

| Checkout | origin | upstream | 主仓关系 |
| --- | --- | --- | --- |
| `sglang/<worktree>/` | `TallMessiWu/sglang` | `sgl-project/sglang` | 忽略的独立 worktree，不是子模块 |
| `sgl-kernel-npu/` | `TallMessiWu/sgl-kernel-npu` | `sgl-project/sgl-kernel-npu` | 子模块，gitlink 只按用户要求更新 |
| `vllm-ascend/`、`MindIE-SD/`、`msmodelslim/` | 以实际 remote / `.gitmodules` 为准 | 各自官方仓库 | 参考子模块，不随清理自动更新 |

**`sglang/qwen3.5_dense_w8a8/` 持有共享 `.git`，必须保留。** 所有修改分支均在 `sglang/` 直属目录拥有独立 worktree，不在已有 worktree 切换功能分支。先运行：

```bash
git -C sglang/qwen3.5_dense_w8a8 worktree list --porcelain
```

SGLang fork 的默认分支已按用户授权从旧 Diffusion 分支改为 `main`；仅改设置，没有改写 main 内容。两个 fork 的 main 仅作上游镜像。

<a id="current-branch-map"></a>

## 当前 SGLang worktree

以下路径均相对 `sglang/`。清理后共 **14 个**，工作区均干净；除特别注明外，本地与 origin 同步。表中短 SHA 是本轮定位快照，操作前重新查询。

| 用途 | 分支 | worktree | 本地 HEAD / 差异 |
| --- | --- | --- | --- |
| #43739 | `junlin_qwen3.5_moe_w4a8` | `qwen3.5_moe_w4a8` | `c0e0fab8e6a0` |
| #43739 + #36426，仅组合验证 | `junlin_qwen3.5_moe_w4a8_pr36426` | `qwen3.5_moe_w4a8_pr36426` | `83a7157ab8a5` |
| #32602 | `junlin_qwen3.5_moe_w4a4` | `qwen3.5_moe_w4a4` | `831438498f89` |
| #43740 | `junlin_a5_fia_mixed_split` | `a5_fia_mixed_split` | `4f38ae231f2b` |
| #40419 | `junlin_npu_mamba_ssm_dtype` | `npu_mamba_ssm_dtype` | `756dac5a667b` |
| #44363 | `junlin/pr-w4a8-shared-execution` | `pr_w4a8_shared_execution` | `4f998c97fd76` |
| #44364 | `junlin/pr-modelslim-mixed-gate-up` | `pr_modelslim_mixed_gate_up` | `ec1383b44129` |
| #44365 | `junlin/pr-diffusion-mxfp8-shared-execution` | `pr_diffusion_mxfp8_shared_execution` | `4cfdf3d543df`；远端 `06f27b4dda`，本地落后 1 个提交 |
| #44366 | `junlin/pr-ascend-mxfp-capabilities` | `pr_ascend_mxfp_capabilities` | `9f9aa272543e` |
| #45550 | `junlin/diffusion-mxfp8-fa-auto-selection` | `diffusion_mxfp8_fa_auto_selection` | `ceace1a8d392` |
| MoE LoRA，尚无 PR | `junlin-ascend-moe-lora` | `ascend_moe_lora` | `bc9a4d0753f5` |
| Wan repack 工具 | `junlin/wan-repack-rotation-keys` | `wan_repack_rotation_keys` | `1916bc208baa` |
| 已合入 #32745 的主 clone | `junlin_qwen3.5_dense_w8a8` | `qwen3.5_dense_w8a8` | `c89095b6e50f`；保留共享 git 元数据 |
| 只读上游快照 | `upstream-main` | `upstream-main` | `3d508d614f28`；不是最新 main，不自动移动 |

#43739 当前完整 SHA：`c0e0fab8e6a039793782c6be92a7f0b245d32212`。组合 SHA：`83a7157ab8a5e0da96186d87e7dc946e4d80fdab`。10-30 通用修复通过普通 merge 同步到组合和 #32602；三分支本地与 origin 均同步。组合仍只增加 #36426 的 4 文件补丁（+97/-43），已核对忽略 hunk 行号后的补丁内容与同步前一致，应用来源为 `9ea4881cc32b2c43afbfd10795183752e5d33f22`；正式 PR 不含该补丁。

### #43739 加载峰值对照基线

修改前使用正式 W4A8 分支历史提交 **`493372ce8fee4d8f69dd3d482cf711d840ce2bd7`**，即加载优化提交 `ab8f7cd9ce` 的直接父提交：已有在线 W4A8 Linear/MoE，仍在 post-load 量化，没有完成度 loader。它是当前正式分支的祖先，fetch `junlin_qwen3.5_moe_w4a8` 后可按提交 detached 检出，不另建 before 功能分支。修改后使用上表 #43739 的正式 head，不用 `_pr36426` 组合分支。

`3cdd0182b9` 及 `archive/pr43739-perf-20261029` 已包含加载优化，不能当作“加载优化前”。历史基线到当前 head 还包含上游同步、权重策略和通用修复，因此是版本前后对照，不是 loader 单因素实验。独立入口和命令见 [W4A8 加载峰值](../llm/README.md#w4a8-load-memory)。

### #32602 rebase 恢复点

2026-10-30 经用户明确授权重排并使用限定旧远端提交的 `--force-with-lease` 发布，本地与 origin 同步。保留 main `52604e1d46` 的同步结果；基线同步提交 `d1ef1f8f1a`、W4A4 净功能提交 `3abdc8dc08`、评审修复 `02342dab5039`；随后公共权重准备整理 `b5a1140f70` 使用普通推送。初轮重排未修改父分支；后续按用户要求把通用修复归入 #43739，再通过 `831438498f89` 普通 merge 回 #32602，未再次改写历史。

先于强推发布的远端恢复 tag：

| Tag | 保留版本 |
| --- | --- |
| `archive/pr32602-pre-rebase-20261030` | `745da0dccd`，本地已合并 main、尚未发布的版本 |
| `archive/pr32602-published-20261030` | `9c40660a60`，rebase 前的 PR head |

## 子模块边界

清理前后子模块内部均干净，保留原 checkout，不更新主仓 gitlink：

- `sgl-kernel-npu`：`4a4f838e8f`，本地分支 `codex/a5-gemma-rmsnorm-csrc` 的远端已删除；origin 现只保留 main。该 checkout 与主仓记录的 `500ea7` 不同，不能顺手 stage。
- `vllm-ascend`：`932818a5b1`，与主仓 gitlink 不同，保留用户选择。
- `MindIE-SD`：`c2b8f9b31d`；`msmodelslim`：`24f9195272`。参考代码版本不等于部署版本。

<a id="g2-runtime-record"></a>

## 已归档分支

2026-10-29 用户确认：下表各 HEAD 先发布并核对远端 tag，再删除对应分支及 6 个旧 worktree。**归档 tag 保留源码，不代表该版本仍适合当前部署。** 正式 `junlin/pr-*` 发布分支未删除。

以下 tag 均带前缀 **`archive/cleanup-20261029/`**，后接表中原分支全名：

| 原分支（也是 tag 后缀） | HEAD | 保留原因 |
| --- | --- | --- |
| `junlin/w4a8-shared-execution` | `24e100f1e8429bde02f3e81221c4392e9e69934b` | G2.1 原实现/验收来源 |
| `junlin/modelslim-mixed-gate-up` | `2ef4c276be4b3fa6b7e4578a2d738ec00c0dd668` | G2.1 离线验收载体 |
| `junlin/diffusion-mxfp8-shared-execution` | `6eb8e1dd1c91678284d1355cdb182a85ab4e9863` | G2.2 原验收来源 |
| `junlin/ascend-mxfp-capabilities` | `d6a46f90d97039e14476be47468c996f11caf603` | G2.3 原验收来源 |
| `junlin/npu-mamba-clear-index-copy` | `d38039ef6e32a866a3c421a275cf19f504f9bd58` | 暂停的清零实验，无稳定 serving 收益 |
| `junlin_qwen3.5_moe_w8a8` | `3d72a9dc00d6f9b0139847e54fb99a21cbe85ca1` | 已关闭 #32266 的历史实现 |
| `junlin_diffusion_w4a4` | `5a2befa888901a4e21774c5ce25fb2de3391c25b` | 已合入 #22338 的旧分支 |
| `junlin_diffusion_w8a8` | `e31f5582976f1afcbb08aaf4a232814b0ce63ef3` | 已合入 #24918 的旧分支 |
| `archive/gemma-rmsnorm-sglang-capability-dispatch` | `2a86efda4207091c19bc437e83cefb2a133b068b` | 已关闭 #32908，现由构建期 provider 取代 |
| `junlin_npu_platform_capabilities` | `b2b02108736d30ab1c3b3d2199ce33e318df3201` | 已关闭 #32776 的旧能力设计 |

Wan 原 G2 权重路径与功能边界见 [验收入口](mxfp-refactoring-plan.md#g23-handoff)。repack 工具仍保留：显式选择量化描述，权重旋转张量与描述键同步映射，双 transformer 完整性分别检查；不要自动重跑转换。

<a id="performance-archive"></a>

### G6 性能恢复点

| 仓库 / 入口 | SHA |
| --- | --- |
| SGLang tag `archive/pr43739-perf-20261029` | `a9b402d91698455dd439eaa4f7cba3a5bc636235` |
| SGLang tag `archive/pr43739-perf-pr36426-20261029` | `0c5fc52e8749ddabbdc60ea62174f605462ff30f` |
| 主仓旧探针/命令版本（从 Git 取回） | `5fe3bc7bd5bfe008aad4bd900954fd5ebd002958` |

结论统一见 [G6 归档](mxfp-refactoring-plan.md#g6-performance-archive)。恢复修改必须另建 `sglang/<name>/` worktree，不在当前正式 PR 或临时目录上继续实验。

### 未删除项

远端还有大量旧 backup/probe/其他作者风格的 refs，归属与独有提交用途未充分确认，本轮不批量删除。保留主 clone、LoRA、repack、只读上游快照及全部活跃 PR/组合分支；不因“不是当前任务”就删除。
