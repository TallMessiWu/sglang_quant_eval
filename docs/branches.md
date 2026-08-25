# 活跃分支、PR 与 worktree

> 最近核对：2026-08-25（Asia/Shanghai）。GitHub 状态来自实时 PR 查询，本地状态来自 `git worktree list --porcelain`、`git branch --show-current` 和 `git rev-parse HEAD`。这些信息会漂移，操作前必须重新查询。

## 远程地址

| 仓库 | `origin`（开发 fork） | `upstream`（官方仓） |
| --- | --- | --- |
| SGLang | <https://github.com/TallMessiWu/sglang.git> | <https://github.com/sgl-project/sglang.git> |
| sgl-kernel-npu | <https://github.com/TallMessiWu/sgl-kernel-npu.git> | <https://github.com/sgl-project/sgl-kernel-npu.git> |

主仓 `.gitmodules` 的 `sgl-kernel-npu` URL 指向开发 fork；子模块内保留 `upstream` 作为官方同步来源。

## 当前 open PR（author: `TallMessiWu`）

| 仓库 / PR | 状态 | head 分支 | GitHub / 本地 HEAD | 本地 worktree |
| --- | --- | --- | --- | --- |
| SGLang [#32745](https://github.com/sgl-project/sglang/pull/32745) — Qwen3.5 GemmaRMSNorm on Ascend 950 | Open | `junlin_qwen3.5_dense_w8a8` | `cc552daed5` | `sglang/qwen3.5_dense_w8a8/`（主 clone） |
| SGLang [#32266](https://github.com/sgl-project/sglang/pull/32266) — Qwen3.5 MoE W8A8 MXFP8 | Open Draft | `junlin_qwen3.5_moe_w8a8` | `3d72a9dc00` | `sglang/qwen3.5_moe_w8a8/` |
| SGLang [#32601](https://github.com/sgl-project/sglang/pull/32601) — Qwen3.5 MoE W4A8 MXFP | Open Draft | `junlin_qwen3.5_moe_w4a8` | `5f45082517` | `sglang/qwen3.5_moe_w4a8/` |
| SGLang [#32602](https://github.com/sgl-project/sglang/pull/32602) — Qwen3.5 MoE W4A4 MXFP4 | Open Draft | `junlin_qwen3.5_moe_w4a4` | `68121eeeea` | `sglang/qwen3.5_moe_w4a4/` |
| SGLang [#34387](https://github.com/sgl-project/sglang/pull/34387) — A5 mixed chunked-prefill FIA split | Open Draft | `junlin_a5_fia_mixed_split` | `c61c0f16a2` | `sglang/a5_fia_mixed_split/` |
| sgl-kernel-npu [#638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) — portable Gemma RMSNorm API | Open | `codex/a5-gemma-rmsnorm-csrc` | `99421ab5b6` | `sgl-kernel-npu/` |

核对时，上表 6 个 GitHub head SHA 均与本地 checkout 完全一致，本地 6 个 SGLang worktree 与 kernel checkout 均无未提交改动。#32745 与 #638 已从 draft 转为正式 PR。

2026-08-25 把 #32266、#32601、#32602、#34387 重新基于 `upstream/main` 合并，四个 PR 的 `mergeable` 已从 `CONFLICTING` 回到 `MERGEABLE`。

主仓 `sgl-kernel-npu` gitlink 已于 2026-08-25 记录到 PR #638 的 `99421ab5b6`，与子模块 checkout 一致，`git status` 不再显示 `M sgl-kernel-npu`。该 commit 位于开发 fork 的 `codex/a5-gemma-rmsnorm-csrc` 分支而非 `main`；#638 后续如有新提交或被合并压缩，gitlink 会重新落后，只在用户明确要记录新版本时才更新。

## SGLang worktree 结构

```text
sglang/
├── qwen3.5_dense_w8a8/   # 主 clone；持有共享 .git；PR #32745
├── a5_fia_mixed_split/   # 派生 worktree；base upstream/main @ 5efe9b104d；PR #34387
├── ascend_moe_lora/      # 派生 worktree；junlin-ascend-moe-lora；base upstream/main @ f2c84de022
├── qwen3.5_moe_w8a8/     # 派生 worktree；PR #32266
├── qwen3.5_moe_w4a8/     # 派生 worktree；PR #32601
└── qwen3.5_moe_w4a4/     # 派生 worktree；PR #32602
```

强制规则：**每个需要修改 SGLang 代码的分支，都必须在 `sglang/` 下有独立 worktree。** 不得在现有目录中切换功能分支，也不得直接在外部临时 worktree 修改。

共享仓库曾在系统临时目录注册过一个 `junlin_codeowners` worktree，其目录已消失，注册记录已于 2026-08-25 用 `git worktree prune` 清除。分支 `junlin_codeowners`（`78b216ea63`）本身保留；要复用它必须在 `sglang/` 下新建 worktree，不得再在仓库外目录修改。

## 分支依赖

本地提交祖先关系与 PR 栈为：

```text
#32601 / junlin_qwen3.5_moe_w4a8
  └─ #32602 / junlin_qwen3.5_moe_w4a4   # 含 #32601 的 merge 提交，不含其后的 lazy-op 修复

#32745 / junlin_qwen3.5_dense_w8a8   # 独立修复，与 sgl-kernel-npu #638 配对
  ├─ #32266 / junlin_qwen3.5_moe_w8a8      # 取 #32745 的 Gemma provider 版本
  └─ #34387 / junlin_a5_fia_mixed_split    # merge 了 #32745，未 rebase 前 diff 含其提交
```

#32266 与 #32601 现在只共享 `qwen3_vl.py`、`modelslim.py`、`modelslim_mxfp8.py` 的少量改动，不再是提交层面的父子关系；三者都以 `main` 为 base，可以独立 review。

- SGLang [#30768](https://github.com/sgl-project/sglang/pull/30768)（MoE W8A8）已于 2026-07-29 合并。
- SGLang [#32013](https://github.com/sgl-project/sglang/pull/32013)（ModelSlim packed MXFP4 loader）已于 2026-07-29 合并。
- SGLang [#34829](https://github.com/sgl-project/sglang/pull/34829)（清理 NPU 量化注释）已于 2026-08-21 合并；对应 worktree 已于 2026-08-25 移除，分支 `junlin-remove-vllm-ascend-comments`（`11013cb37c`）仍在本地与 fork 远程。
- 活跃 PR 正文或 diff 若仍显示前两个前置 PR，表示功能分支尚未完成基于最新 `main` 的栈清理；不要再把它们记为 open dependency。
- SGLang [#30318](https://github.com/sgl-project/sglang/pull/30318)（2026-08-16）与 [#30319](https://github.com/sgl-project/sglang/pull/30319)（2026-08-18，均由 `LinyuanLi0046` 提交）已把**离线 ModelSlim `W4A8_MXFP` / `W4A4_MXFP4` MoE** 路径合入 main，对应 `ModelSlimW4A8MXFP4MoE` / `NPUW4A8MXFP4MoEMethod` 与 `ModelSlimW4A4MXFP4MoE` / `NPUW4A4MXFP4MoEMethod`。#32601 与 #32602 原本各自实现了一套同名不同类的离线 scheme，已于 2026-08-25 删除，只保留在线量化入口（`--quantization mxfp_w4a8` / `mxfp4` 的 experts 分支），并把在线分支加进上游的 kernel 类。新增 MoE 量化方案前先查 `moe_quant_schemes` 表，上游条目在前，重复注册会变成死代码。

## 操作前核对

```powershell
# 真实 worktree 注册表
git -C sglang/qwen3.5_dense_w8a8 worktree list --porcelain

# 当前目录的分支、HEAD、远程
git -C sglang/<worktree> status --short --branch
git -C sglang/<worktree> remote -v
git -C sglang/<worktree> rev-parse HEAD

# kernel 仓
git -C sgl-kernel-npu status --short --branch
git -C sgl-kernel-npu remote -v
```

新增 SGLang 功能分支时，从主 clone 创建独立目录：

```powershell
git -C sglang/qwen3.5_dense_w8a8 worktree add ../<worktree-name> -b <branch> upstream/main
```

如果分支已经存在，去掉 `-b` 并传现有分支名。创建前先确认该分支没有被其他 worktree 占用。
