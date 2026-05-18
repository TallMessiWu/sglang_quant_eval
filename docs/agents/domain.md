# Domain docs

本仓库采用 **single-context** 布局：领域上下文只有一份。

## 位置

- `CONTEXT.md`（仓库根）— 项目领域语言、关键概念、跨子模块的约定
- `docs/adr/` — 架构决策记录（ADR），按 `NNNN-<slug>.md` 编号

## 为何 single-context

虽然仓库下有 `sglang/` 子模块以及多个 worktree 分支（`junlin_mxfp4`、`junlin_qwen3_dense*`、`junlin_qwen3_moe_w8a8` 等），但这些拆分的是**实现分支**而非**领域**。整体仍然是同一套领域语言：

- MXFP8 / MXFP4 量化（在线 / 离线 / ModelSlim）
- Ascend NPU（A2/A3，CANN ≥ 8.0.RC3）
- Diffusion (Wan2.2) 与 LLM (Qwen3/3.5 Dense + MoE) 两条主线
- 在线量化 vs 离线 ModelSlim 加载

因此一份 `CONTEXT.md` 已足够。

## Agent 消费规则

读 domain 文档的技能（`improve-codebase-architecture`、`diagnose`、`tdd` 等）应：

1. 先读根 `CONTEXT.md` 获取领域术语和当前重点
2. 涉及"为什么这么做"的判断时，按时间倒序扫 `docs/adr/`
3. 若主仓 `AGENTS.md` 与 `CONTEXT.md` 重复，以 `AGENTS.md` 内容为准（它已涵盖分支规则、worktree 目录、关键代码路径、已知陷阱）

## 与 AGENTS.md 的分工

- `AGENTS.md`（= `CLAUDE.md`）— 流程性指令：分支约定、提交流程、调试 CI、agent 协作模式开关
- `CONTEXT.md` — 领域知识：MXFP8/MXFP4 数据布局、NPU 算子语义、量化路径的因果链
- `docs/adr/` — 一次性决策：例如「为何 MoE gmm1 用 fused swiglu_quant 而不是拆三步」

如果某条信息不确定该放哪：流程类放 AGENTS.md，知识类放 CONTEXT.md，**带不可逆决策的**放 ADR。

> 注：本仓当前尚未创建 `CONTEXT.md` 和 `docs/adr/`，将由后续相关 skill（例如首次跑 `improve-codebase-architecture`）或手动创建按需补齐。
