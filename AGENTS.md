# AGENTS.md

本仓库用于研究、实现和验证 SGLang 在华为 Ascend NPU 上的 MXFP8/MXFP4 量化，覆盖 LLM `srt`、Diffusion `multimodal_gen`、Dense/MoE 以及在线/离线 ModelSlim 路径。

- SGLang fork：<https://github.com/TallMessiWu/sglang>
- 关联 Issue：[Diffusion #14424](https://github.com/sgl-project/sglang/issues/14424)、[LLM #21584](https://github.com/sgl-project/sglang/issues/21584)
- 当前 PR、分支、HEAD、worktree 与远程地址：[docs/branches.md](docs/branches.md)
- 能力矩阵与关键代码路径：[docs/quantization-overview.md](docs/quantization-overview.md)
- 已知故障模式与机制：[docs/known-pitfalls.md](docs/known-pitfalls.md)

## SGLang worktree 硬规则

1. **所有需要修改 SGLang 代码的分支，都必须有一个位于 `sglang/` 直属目录下的独立 worktree。** 不得在临时目录、仓库外目录或其他分支的 worktree 中修改。
2. `sglang/qwen3.5_dense_w8a8/` 是当前主 clone，持有共享 `.git`；其他 SGLang 目录是它派生的 worktree。
3. 修改前必须从任一现有 SGLang checkout 执行：

   ```powershell
   git -C sglang/qwen3.5_dense_w8a8 worktree list --porcelain
   ```

   以命令输出为准，不按文档快照猜测。
4. 进入与目标分支对应的 worktree 修改；**不要在现有 worktree 内 `git checkout`/`git switch` 到另一功能分支**。
5. 新建修改分支时，同时创建 `sglang/<worktree-name>/`。如果该分支已在 `sglang/` 外注册，先停止修改，再将其迁移或移除后重建到 `sglang/`。
6. `sglang/` 整体被主仓 `.gitignore` 忽略，不是子模块；SGLang 提交和推送都在具体 worktree 内完成，主仓不记录其 gitlink。
7. worktree、PR 或远程映射变化后，更新 [docs/branches.md](docs/branches.md)，不要把易漂移的 HEAD/PR 状态复制回本文件。

当前映射见 [docs/branches.md](docs/branches.md)。其中还记录了一个历史遗留、位于系统临时目录的 `junlin_codeowners` worktree；它不是活跃 PR worktree，禁止继续在那里改代码，复用该分支前必须迁回 `sglang/`。

## 仓库与远程约定

| Checkout | `origin` | `upstream` | 主仓关系 |
| --- | --- | --- | --- |
| `sglang/<worktree>/` | `TallMessiWu/sglang` | `sgl-project/sglang` | 本地 worktree，主仓忽略 |
| `sgl-kernel-npu/` | `TallMessiWu/sgl-kernel-npu` | `sgl-project/sgl-kernel-npu` | 主仓子模块，记录 gitlink |
| `vllm-ascend/` | 以本地 remote 为准 | `vllm-project/vllm-ascend` | 参考子模块 |
| `MindIE-SD/`、`msmodelslim/` | 以 `.gitmodules` 为准 | 以本地 remote 为准 | 参考子模块 |

- LLM serving (`srt`) 的 MXFP8/MXFP4 实现优先对齐 `vllm-ascend` 的语义和 NPU API 契约。
- Diffusion (`multimodal_gen`) 优先参考 `MindIE-SD`。
- 离线 checkpoint 的真实存储格式以当前 `msmodelslim` 导出代码和 `quant_model_description.json` 为准。
- Qwen3 与 Qwen3.5 共用底层 Linear/MoE 量化算子；不要为模型名复制一套量化实现。

## 量化开发约定

- **在线量化**：从 BF16/FP16 权重加载，在 `process_weights_after_loading` 中量化；常用入口为 `--quantization mxfp8`、`mxfp_w4a8`、`mxfp4`。
- **离线 ModelSlim**：加载预量化 checkpoint，scheme 由 `quant_model_description.json` 自动检测；启动脚本通常不需要显式传 `--quantization modelslim`。
- NPU-only import 必须受 `sglang.srt.utils.is_npu()` 保护；不要在跨平台模块顶层无条件 `import torch_npu`，也不要用未安装 NPU 插件时恒假的 `current_platform.is_npu()` 代替。
- ModelSlim MXFP4 权重是 packed `uint8 [out, in/2]`；已打包权重不得二次 cast/pack。
- A5 默认 ATB attention 可能在 warmup/解码失败；本仓库 `llm/` 服务脚本用 `ASCEND_USE_FIA=1` 选择 FIA 路径。
- SGLang 上游文档改动写入 `docs_new/docs/` 的 `.mdx`，不要改 legacy `docs/`。
- Gemma RMSNorm 的 SoC provider 由 `sgl-kernel-npu` wheel **构建期**选择；SGLang 只调用稳定 API，不新增 `is_npu_a5()` 一类运行时硬件代际分支。相关工作使用 `$sgl-kernel-npu-dev` skill，用户文档见 [docs/sgl-kernel-npu-build.md](docs/sgl-kernel-npu-build.md)。
- 更细的 layout、dtype、offset、routing、GMM 与视觉塔契约统一放在 [docs/known-pitfalls.md](docs/known-pitfalls.md)，不要继续扩写 AGENTS.md。

## 验证与结论边界

- 先确认目标 worktree、分支、diff 和用户已有改动；保留不相关的脏文件。
- 优先做最小、可证伪验证。Python 语法、AST、shape probe 和 CPU 单测只能证明静态契约，不能表述为 A5/A2/A3 端到端通过。
- NPU kernel/layout 变更需要对应硬件 probe；模型加载、warmup、文本/图片请求、精度和性能结论需要真实 e2e 日志。
- 大规模 merge/rebase 后先重新跑 baseline，再叠加新改动，避免把上游移植回归误归因于当前修改。
- PR head、CI、review、remote、依赖和 worktree 都是时效信息；报告或修改前实时查询 GitHub，并与本地 `HEAD` 对照。

## 提交与发布

- 用户要求提交时使用 `gitmoji-commit` skill 生成 Gitmoji 英文提交信息。
- SGLang 改动在对应 `sglang/<worktree>/` 中提交并推到 `origin`；主仓无需更新 SGLang 指针。
- `sgl-kernel-npu` 是主仓子模块：在子模块内提交/推送后，只有用户要记录该版本时才更新主仓 gitlink。
- 不主动覆盖、重置或清理用户的分支、worktree、子模块改动。
- 本仓默认单 agent；除非用户明确要求，不创建 agent team 或子 agent。

## 文档入口

- 活跃 PR / branch / worktree / remote：[docs/branches.md](docs/branches.md)
- 量化状态、入口与代码路径：[docs/quantization-overview.md](docs/quantization-overview.md)
- 详细陷阱：[docs/known-pitfalls.md](docs/known-pitfalls.md)
- `sgl-kernel-npu` 构建与 target wheel：[docs/sgl-kernel-npu-build.md](docs/sgl-kernel-npu-build.md)
- `sgl-kernel-npu` 仓库级工作流 skill：[.agents/skills/sgl-kernel-npu-dev/](.agents/skills/sgl-kernel-npu-dev/)
- Ascend API：[docs/npu-api/](docs/npu-api/)
- 领域语言：[CONTEXT.md](CONTEXT.md)
- Agent issue/triage/domain 约定：[docs/agents/](docs/agents/)
- 启动与评测脚本：`llm/`、`diffusion/`
