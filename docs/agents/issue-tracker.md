# Issue tracker

本仓库使用 **Local markdown** issue tracker。Issue 作为本仓库内的 markdown 文件存在，不使用 GitHub Issues、GitLab Issues 或外部系统（Jira / Linear）。

## 位置

所有 issue 放在 `.scratch/<feature>/` 下，每个 issue 一个 markdown 文件。`<feature>` 是该 issue 所属的功能/方向短 slug（例如 `mxfp4-diffusion`、`qwen3-moe-w8a8`）。

目录示例：

    .scratch/
      mxfp4-diffusion/
        001-blockwise-scale-layout.md
        002-cpu-offload-quant-timing.md
      qwen3-moe-w8a8/
        001-gmm1-fused-swiglu-quant.md

## 文件结构

每个 issue 文件以 YAML frontmatter 开头，body 是 markdown：

    ---
    id: mxfp4-diffusion/001
    title: 简短标题
    status: 待评估            # 见 docs/agents/triage-labels.md
    created: 2026-05-18
    updated: 2026-05-18
    ---

    ## 背景

    ...

    ## 当前理解

    ...

    ## 验收标准 / Acceptance criteria

    - [ ] ...

`status` 字段的合法取值见 `docs/agents/triage-labels.md`。

## 与上游 issue 的关系

本仓的 SGLang 适配工作在上游有 issue 跟踪：
- Diffusion: `sgl-project/sglang#14424`
- LLM Qwen3: `sgl-project/sglang#21584`

本地 `.scratch/` 下的 issue 用来记录**实现细节、踩坑、子任务**等不适合或还未发到上游的内容。如果某条本地 issue 对应一个上游 issue，请在 frontmatter 加 `upstream:` 字段：

    upstream: https://github.com/sgl-project/sglang/issues/14424

## Agent 如何操作

- 列出 issue：直接 `ls .scratch/<feature>/`，或在仓库内 grep `^status:` frontmatter
- 创建：在 `.scratch/<feature>/` 下新建 `NNN-<slug>.md`，编号在该 feature 下递增
- 更新状态：修改 frontmatter `status:` 字段，更新 `updated:` 日期
- 关闭：状态置为 `不予处理`，或在 PR 提交时附 `Closes .scratch/<feature>/NNN`

不要调用 `gh issue` / `glab issue` 这类 CLI 处理本仓 issue——那是上游 issue 的工具。
