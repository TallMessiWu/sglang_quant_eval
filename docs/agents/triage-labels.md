# Triage labels

`triage` 技能在处理 incoming issue 时会用以下 5 个 canonical 角色推动状态机。本仓库**使用中文字符串**作为标签值；canonical key（左列）仅供 agent 内部引用。

| Canonical key | 本仓字符串 | 含义 |
| --- | --- | --- |
| `needs-triage` | **待评估** | 新进 issue，等维护者过一遍优先级/范围 |
| `needs-info` | **待补充信息** | 等 reporter 提供更多细节才能继续 |
| `ready-for-agent` | **可交付 agent** | 规格完整，AFK agent 可直接接手实现 |
| `ready-for-human` | **需人工实现** | 规格完整但只能由人来做（涉及硬件验证、需要外部 review 等） |
| `wontfix` | **不予处理** | 明确不修；写明原因 |

## 落地方式

本仓为 Local markdown issue tracker（见 `docs/agents/issue-tracker.md`），标签作为 issue 文件 frontmatter 里的 `status:` 字段值：

    ---
    id: mxfp4-diffusion/003
    title: process_weights_after_loading 中 transpose 后必须不加 contiguous
    status: 可交付 agent
    ---

任一时刻 issue 只能处于一种状态。状态变更时同步更新 frontmatter 的 `updated:` 字段。

## Agent 使用规则

- 新建 issue 默认 `status: 待评估`
- 不要使用本表之外的字符串（如「TODO」「fixme」），否则 `triage` 技能识别不到
- 状态转换建议路径：
  - `待评估` → `待补充信息` / `可交付 agent` / `需人工实现` / `不予处理`
  - `待补充信息` → `待评估` / `可交付 agent` / `需人工实现`
  - `可交付 agent` → `需人工实现`（升级）或 closed
  - 任何状态都可转为 `不予处理`
