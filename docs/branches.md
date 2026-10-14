# 活跃分支、PR 与 worktree

## 维护分工

本文负责 **branch / worktree / HEAD / remote 定位和详细操作、验证记录**。PR 合入顺序、依赖和当前 Open / Draft / Merged / CI 状态统一维护在 [PR 合入跟踪表](mxfp-refactoring-plan.md#pr-merge-status)；重构进度统一维护在 [G1–G5 目标跟踪表](mxfp-refactoring-plan.md#refactoring-status)。这里不再维护第二份当前合入状态表。

<a id="current-branch-map"></a>

## 当前计划涉及的分支定位（2026-10-14）

2026-10-14 通过 `git worktree list --porcelain`、本地 HEAD 和 `gh pr view` 核对。七个待合入个人 PR 的本地 HEAD 均与 GitHub head 一致。此表只记录定位，操作前仍以实时命令为准；其他测试、历史分支见下文记录和实际 worktree 注册表。

| PR / 任务 | 分支 | HEAD | worktree（相对主仓） |
| --- | --- | --- | --- |
| [kernel #638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | `codex/a5-gemma-rmsnorm-csrc` | `4a4f838e8f` | `sgl-kernel-npu/` |
| [SGLang #32745](https://github.com/sgl-project/sglang/pull/32745) | `junlin_qwen3.5_dense_w8a8` | `cf48b477db` | `sglang/qwen3.5_dense_w8a8/` |
| [SGLang #40419](https://github.com/sgl-project/sglang/pull/40419) | `junlin_npu_mamba_ssm_dtype` | `756dac5a66` | `sglang/npu_mamba_ssm_dtype/` |
| [kernel #824](https://github.com/sgl-project/sgl-kernel-npu/pull/824) | `cache-loc-assign-bounds` | `1c6fad69a1` | `sgl-kernel-npu-worktrees/cache-loc-assign-bounds/` |
| [SGLang #43739](https://github.com/sgl-project/sglang/pull/43739)（原 [#32601](https://github.com/sgl-project/sglang/pull/32601)） | `junlin_qwen3.5_moe_w4a8` | `ded4671577` | `sglang/qwen3.5_moe_w4a8/` |
| [SGLang #32602](https://github.com/sgl-project/sglang/pull/32602) | `junlin_qwen3.5_moe_w4a4` | `425b0a9abb` | `sglang/qwen3.5_moe_w4a4/` |
| [SGLang #43740](https://github.com/sgl-project/sglang/pull/43740)（原 [#34387](https://github.com/sgl-project/sglang/pull/34387)） | `junlin_a5_fia_mixed_split` | `01bb8dc685` | `sglang/a5_fia_mixed_split/` |
| G2.1：Dense W4A8 公共执行 | `junlin/w4a8-shared-execution` | `24e100f1e8` | `sglang/w4a8_shared_execution/` |
| G2.2：Diffusion MXFP8 公共执行 | `junlin/diffusion-mxfp8-shared-execution` | `6eb8e1dd1c` | `sglang/diffusion_mxfp8_shared_execution/` |
| G2.3：Ascend MXFP 能力校验 | `junlin/ascend-mxfp-capabilities` | `d6a46f90d9` | `sglang/ascend_mxfp_capabilities/` |
| G2.2 / G2.3 数值与模型对照基线 | detached HEAD（只读对照） | `62ff466ac0` | `sglang/g2_validation_baseline/` |
| Wan repack rotation / 量化元数据兼容 | `junlin/wan-repack-rotation-keys` | `1916bc208b` | `sglang/wan_repack_rotation_keys/` |
| ModelSlim Dense gate/up 混合精度与预处理兼容 | `junlin/modelslim-mixed-gate-up` | `2ef4c276be` | `sglang/modelslim_mixed_gate_up/` |

G2.1 的开发基线为 #43739 的 `ded4671577`；G2.2 / G2.3 的建分支基线均为 `62ff466ac0`。实现与验收进度见目标跟踪表，不由 HEAD 或目录存在与否推断。

2026-10-14 验收工具准备：再次实时核实 G2.2 / G2.3 origin 分支仍分别为 `6eb8e1dd1c` / `d6a46f90d9`，两个功能 worktree 均干净；在 `sglang/` 下新增上述 detached 基线 worktree，没有修改生产实现或叠加兼容提交。`validation/run_g2.sh` 获取固定分支后核对 SHA，复用同 SHA 的直属 worktree，缺失时只新建 detached 验收 checkout，不切换、重置已有分支。主仓预存的未跟踪 `validation/g2_npu.py` 经审查补齐；`.pi/` 和 kernel 原有差异保留。

2026-10-14 Wan 格式检查后新增独立工具修复：从 `62ff466ac0` 建立直属 worktree `sglang/wan_repack_rotation_keys/`，提交 [`6e19df8e6f8b1647116cf1e8f3311fb7e07aba2d`](https://github.com/TallMessiWu/sglang/commit/6e19df8e6f8b1647116cf1e8f3311fb7e07aba2d) 并普通推送至 `origin/junlin/wan-repack-rotation-keys`，local/remote SHA 相同、ahead/behind `0 0`，未创建 PR。只修改 `wan_repack.py` 及其文件格式回归：保留 loader 识别的 `attn1.q_rot/k_rot` 名称、独立转换 JSON-only attention 元数据，并在模板缺 `model_index.json` 或 transformer config 时先报错。G2.2/G2.3 的分支与 SHA 保持不变；该工具生成的同一份 checkpoint 将用于两边对照，无需把工具提交混入运行时分支。

验证为真实 CPU safetensors 文件读写的 4 项隔离回归（含 TI2V 单 transformer、I2V 双 transformer、FP8/uint8/BF16 字节保留及模板缺失）：基线出现 rotation 键错误、JSON-only 元数据缺失和缺配置未拒绝，修复后通过。harness 直接加载真实 converter、logging、CI 注册模块，仅绕过 SGLang 包初始化；标准入口仍缺 `orjson`，未运行 NPU。证据 `/tmp/g2-repack-red.log`、`/tmp/g2-repack-green.log`，harness `/tmp/g2_repack_test_harness.py`。lint、format、语法及 diff 检查通过。服务器源权重检查与后续命令见目标表 G2.2 段落。

2026-10-14 I2V 检查后继续在同一工具分支修复多描述文件选择：用户的 high/low noise 目录均同时含 `quant_model_description_w8a8_fp8_only.json`（607 个 FLOAT weight）和 `quant_model_description_w8a8_mxfp8.json`（400 个 MXFP8 + 207 个 FLOAT weight）。旧代码排序取首项，会把实际 FP8/uint8 权重配上全 FLOAT 描述。新增 `--quant-description-file`，在每个 expert 目录分别解析指定文件；默认多候选时拒绝，并在复制模板前检查全部 expert。提交 [`1916bc208baa931c3bd834d55f778c70d7d154d2`](https://github.com/TallMessiWu/sglang/commit/1916bc208baa931c3bd834d55f778c70d7d154d2) 已普通推送至原分支，local/remote SHA 相同、ahead/behind `0 0`，未创建 PR。

本轮 7 项隔离 CPU 文件回归通过：保留原 4 项，新增多候选拒绝、真实 CLI 双 expert 分别选择和选择错误提前拒绝。修改前，多候选测试未按预期拒绝，新增接口测试也因接口尚不存在失败；修复后通过。证据 `/tmp/g2-repack-description-red.log`、`/tmp/g2-repack-description-green.log`。仅绕过包初始化，converter、argparse、文件读写使用真实代码；lint、format、语法和 diff 检查通过，不计 NPU 验收。`validation/run_g2.sh` 固定新 SHA，新建 checkout 名称带 SHA 后缀，避免与用户已有旧版本 detached worktree 冲突；同 SHA 仍优先复用。

2026-10-14 15:23 用户服务器实际执行：主仓 `/mnt/share/hajimi/sglang_quant_eval` 下创建 detached worktree `sglang/g2_validation_repack_1916bc208b`，HEAD `1916bc208baa931c3bd834d55f778c70d7d154d2`，Python 为 `/usr/local/python3.11.10/bin/python3`。回传 `/tmp/g2-repack-FYhE4Mpr/run.log` 显示 repack 完成；输出权重及元数据核对证据统一见 G2.2 目标段落。该路径属于用户服务器，不能与本机 worktree 注册表混同，亦不代表 G2.2/G2.3 运行时已执行。

2026-10-14 局部复核：G2.1、G2.2、G2.3 及 mixed gate/up 四个 worktree 的本地 HEAD 与实时 `git ls-remote origin` 相同，分别为 `24e100f1e8429bde02f3e81221c4392e9e69934b`、`6eb8e1dd1c91678284d1355cdb182a85ab4e9863`、`d6a46f90d97039e14476be47468c996f11caf603`、`2ef4c276be4b3fa6b7e4578a2d738ec00c0dd668`；工作区干净。mixed gate/up 是 G2.1 + 混精兼容 + resize 兼容，没有叠加 G2.2 / G2.3。此核对不刷新其他 PR 的 CI / review / 合入状态。

2026-10-13：在原 `junlin/modelslim-mixed-gate-up` 分支及其 `sglang/modelslim_mixed_gate_up/` worktree 中回移上游 [#43309](https://github.com/sgl-project/sglang/pull/43309) 的 `c6e822c29b9bb3d392e65a7bcb271a5d2d63613d`。实时 GitHub 查询确认该 PR 于 2026-10-12 合入；回移后的 `qwen_vl_processor.py` Git blob 与上游提交相同（`c5c9ea5c24180459e8faf16623fac26d13242a85`）。修复 Transformers 5.19 图片/视频 resize 的参数与 size 契约；标准补丁为 `llm/patches/qwen_vl_processor_transformers_519.patch`。旧代码在真实 HF resize 函数体的隔离 CPU harness 中复现相同图片 TypeError，回移后图片、批次顺序、禁用 resize 和视频时间维度四项契约检查通过，插值后端/分组叶子被隔离，不计为 NPU warmup 或生成精度通过。修复以 [`2ef4c276be4b3fa6b7e4578a2d738ec00c0dd668`](https://github.com/TallMessiWu/sglang/commit/2ef4c276be4b3fa6b7e4578a2d738ec00c0dd668) 提交并普通推送至 `origin/junlin/modelslim-mixed-gate-up`，本地与远端 SHA 相同、ahead/behind `0 0`；没有创建 PR。按用户要求，新建的 `junlin/npu-qwen-vl-resize-compat` 本地/远端分支及对应 worktree 已删除，后续修复继续在原分支推进。用户要求优先修复精度问题，预处理补丁收敛的待办见 `mxfp-refactoring-plan.md` 第 6 节。

**以下带日期的记录保留当时的状态与验证证据，不继续滚动更新其中的 PR 状态、旧编号或 HEAD。** 历史表中的旧 #32601 / #34387 通过上表映射到新 PR；“当前”“已推送”“保留”等措辞仅对记录当时有效。

## ModelSlim Dense gate/up 混合精度启动诊断与修复

服务器实际运行 `junlin/w4a8-shared-execution@24e100f1e8429bde02f3e81221c4392e9e69934b`，使用 `/mnt/share/weight/Qwen3.6-27B_mxfp-w4a8-0821`，architecture 为 `Qwen3_5ForConditionalGeneration`。用户回传的真实 `quant_model_description.json` 显示：`model.language_model.layers.0.mlp.gate_proj.weight=FLOAT`、`up_proj.weight=W4A8_MXFP`、`down_proj.weight=W4A8_MXFP`；MTP layer 0 的三片均为 FLOAT。原始 `is_layer_skipped` 的纯 CPU 最小探针复现同一个 fused-shard ValueError，真实 key/精度已排除此前“缺失 key 或前缀映射”的假设；错误发生在构造层时，不是公共 W4A8 apply/NPU 算子回归。

用户选择保留真实混合精度，不改描述 JSON 冒充权重转换。从该部署 SHA 创建独立直属 worktree `sglang/modelslim_mixed_gate_up/` / `junlin/modelslim-mixed-gate-up`，不修改 G2.1/G2.2/G2.3 或原 PR 分支。修复仅让 Qwen3.5 Dense MLP 对完整、异构的 ModelSlim gate/up 项使用两套现有 ColumnParallelLinear；各自 FLOAT / packed FP4 weight、UE8M0 scale、TP weight loader 和 post-load 独立，输出仍进入原 SiLU-and-mul/down projection。两类 Dense Qwen3.5 decoder 显式允许该结构，其他模型和同精度 gate/up 继续用合并层；原 fused precision 检查保留，缺失项不作为拆分证据。Qwen3.5 Causal/Conditional loader 对已有的独立 gate/up parameter 保留 canonical name，不误改为 fused name 或传 fused shard_id；NVFP4 fusion 跳过非合并结构。

修复已提交并普通推送：[`185fad010a94a61ab8897d72a067bac574addda7`](https://github.com/TallMessiWu/sglang/commit/185fad010a94a61ab8897d72a067bac574addda7)，英文提交信息 `:bug: fix(npu): Support mixed gate/up precision`，4 files、318 insertions、8 deletions。本地 HEAD 与 origin 分支 SHA 相同，tracking ahead/behind `0 0`，worktree 干净；没有创建 PR，没有改写已发布历史。发布日志为 `.pi/tasks/01a12eff-f049-75d4-9d0f-360c93f9a30c-4113/bd325c5ed.output`。该提交发布时尚未完成硬件验收；后续功能复测见下方 2026-10-14 记录。发布前远端只读核实：G2.1 origin head 仍为 `24e100f1e8429bde02f3e81221c4392e9e69934b`，新修复 remote branch 当时不存在；#43739 head 仍为 `ded467157750ccfd20dd2256110fad9ff104dc08`，官方 main 为 `7bd41aec126f955b8afc0ee1d03d718e4c2c562f`，未自动 rebase/merge，验证固定在部署基线。

本机验证：**7 个隔离 CPU/mock 回归通过**，覆盖完整异构与缺项、正反方向 FLOAT/W4A8、保留原 guard、同精度仍融合、SiLU×up 顺序、两种 Qwen3.5 loader 的独立与融合 weight/scale 路由，以及逻辑 TP=2 两 rank 的 Column/Row packed weight/UE8M0 scale 分片和 parallel_group。相同新测试在未修补 `24e100f1e8` 基线上因新接口/opt-in 不存在按预期失败；不把这些接口红灯单独冒充原 ValueError 的复现，原 ValueError 的最小探针证据见前述诊断。harness 加载真实 Linear/Config/Scheme/parameter/environment 模块及精确 MLP/loader 函数；只隔离 serving/hardware/collective leaves 和 activation，逻辑 TP 不算多卡通信验收。首次新增融合 loader 用例失败是 harness 未提供 `SGLANG_NPU_W4A4_NEW_PACKING`，改为加载完整生产环境模块后通过，未改生产 Linear。原 `is_layer_skipped` AST、W4A8 公共执行、scheme 和 Linear 文件不变。

语法/暂存区 compile、CI 标记 `base-a-test-cpu`、仓库固定版本 ruff CI 规则与 format、isort、codespell、diff 检查通过。生产文件全量默认 lint 的 26/35/48 条历史诊断未新增（测试导入最终按仓库 isort 分组，以实际 pre-commit 规则为准，不宣称全量默认 lint 清零）。普通 registered 入口仍在 package import 阶段缺 `psutil`，不计为通过。证据：`.pi/tasks/01a12eff-f049-75d4-9d0f-360c93f9a30c-4113/` 中 `b30648263.output`（7 个绿灯与基线红灯）、`bcf1704b5.output`（6 个含 TP 绿灯和普通入口缺依赖）、`be9f407ff.output`（最终静态检查）；harness 为 `.pi/debug/modelslim_mixed_gate_up_cpu.py`。本机仍无 torch_npu/NPU，以上检查不能代替真实 checkpoint load/post-load/warmup/文本图片、数值、性能或多卡 TP e2e 验证。

2026-10-13 服务器后续反馈：先出现 `chunk_state dtype must match initial_state`，建议显式 `SGLANG_MAMBA_SSM_DTYPE=bfloat16`（该部署基线的 SSM 默认回退为 float32，配置/环境变量可覆盖）。随后 14:29:52 的日志已进入 scheduler generation → decode NPU graph → `NPUCudaGraphBackend.replay_with_input_update()` → `graph.replay()`，报 `207005` / `EL0006` / `bind sq cq failed, retCode=0x7020023` / `Build SQ/CQ failed`。当前直接证据为 graph replay 时驱动 SQ/CQ 资源分配失败，不据通用 memory 文案断言 HBM OOM 或磁盘不足，也不将前置 image/torch.save warnings、后续 py-spy/CUDA coredump 清理信息作为根因。该次日志未附源码 SHA、CANN/driver/torch_npu 或卡资源信息，不能绑定为 `185fad010a` 的硬件通过。建议保留权重/SSM dtype/请求不变，先仅禁 decode graph 做 eager 对照；若 eager 通过，再去掉禁用项，仅捕获 decode bs=1 复测。已核实本基线参数为 `--cuda-graph-backend-decode disabled` / `--cuda-graph-bs-decode 1`；旧 `--disable-cuda-graph` 是双阶段禁用的兼容参数，不与新选择器混用。本机无法复现此 NPU 资源故障，当时等待服务器同请求 A/B 和 `npu-smi info`，未修改 SGLang graph 代码或执行进程清理。

2026-10-13 最新用户复测更正：服务器报告运行 `junlin/modelslim-mixed-gate-up@2ef4c276be4b3fa6b7e4578a2d738ec00c0dd668`，保留 `SGLANG_MAMBA_SSM_DTYPE=bfloat16`。先用 `EXTRA_ARGS=--disable-cuda-graph` 启动，`./curl.sh` 输出正常；随后 `unset EXTRA_ARGS`，用户明确确认已停止旧服务并重新执行 `./qwen3.5_dense_offline_w4a8.sh 0`，再次请求也无乱码。本地核实该启动脚本仅从 EXTRA_ARGS 添加禁用参数，未另行硬编码禁 graph。因此此前退化目前未复现，graph 缺陷及环境故障均未证实；不继续凭旧假设调整 update/replay。尚无当前服务 resolved graph 配置/实际 replay 日志、原拿破仑长文本重复及 batch/padding 的完整验收证据。

2026-10-14 用户确认已测试、当前正常，并要求记录验收。结合上段明确的部署 SHA、BF16 SSM、去掉禁 graph 参数及停止重启操作，将该 checkpoint 的离线功能验证记为**通过（用户真机复测）**；G2.1 具体勾选及剩余在线/数值项目统一见 [目标清单](mxfp-refactoring-plan.md#refactoring-status)。环境问题是用户对此前异常的推测，具体根因未证实；没有新增生产 graph 修复，不再要求为当前已正常的服务机械交换 update/replay。未收到的数值、性能与 batch/TP/EP 明细不补写为通过。当时只同步证据、未提交文档；后续用户已授权本轮文档核对后统一提交推送，保留 SGLang、kernel 和其他已有改动。

2026-10-14 在线功能补验：用户运行 `llm/qwen3.5_dense_online_w4a8.sh`，贴出 10:23:04 的启动日志并确认最终请求结果正常。视觉 blocks 0–26 的 `mlp.linear_fc2` 为 K=4304，命中 `Mxfp4W4A8Config.get_quant_method()` 的非 32 对齐回退；本地核实该保护来自本 fork 的 `7aa87ddfab7b2e62c63a7c40d04c1c0fb6bd512f`，作者 Junlin Wu，提交日期 2026-07-24，已有对应配置回归，G2.1 保留该策略。日志及功能反馈不能推及全模型全部 W4A8，也不能仅凭旧提交断言当前 torch_npu/CANN 仍不支持原形状。本次在线反馈未重新附服务源码 SHA、完整请求矩阵或数值结果；此前部署报告为 `2ef4c276be`。验收状态统一更新目标清单为“在线/离线基本功能通过，NPU 数值对照待补”，本轮文档发布按用户最新授权执行。

## 2026-10-12：G2.2 / G2.3 与 Dense W4A8 启动脚本发布

用户本轮明确授权提交和推送。三个提交均使用 Gitmoji 英文信息、普通 push；逐项 `git ls-remote` 与本地 HEAD 一致，tracking ahead/behind 均为 `0 0`。没有强推、rebase、修改原 PR 分支、创建 PR 或更新 kernel gitlink。

| 仓库 / 交付 | 实现提交 | remote branch | 提交范围 |
| --- | --- | --- | --- |
| SGLang G2.2 | `6eb8e1dd1c91678284d1355cdb182a85ab4e9863` | `origin/junlin/diffusion-mxfp8-shared-execution` | 2 个文件；离线 MXFP8 adapter 与 CPU/mock tests |
| SGLang G2.3 | `d6a46f90d97039e14476be47468c996f11caf603` | `origin/junlin/ascend-mxfp-capabilities` | 12 个文件；局部能力、实际校验接入、支持文档和 CPU/mock tests |
| 主仓 Dense W4A8 启动脚本 | `e5ea39a5aa5f1a3bd50766d430c0793f8e40e5b6` | `origin/main` | `llm/qwen3.5_dense_{offline,online}_w4a8.sh` 与 `llm/README.md` |

启动脚本离线默认使用 `/mnt/share/weight/Qwen3.6-27B_mxfp-w4a8-0821`，由 `quant_model_description.json` 自动识别 ModelSlim；在线要求明确设置原始 BF16/FP16 的 `MODEL_PATH`，使用 `mxfp_w4a8`，拒绝 ModelSlim 目录。Shell 语法与 12 个 mock 用例通过，未执行真实 NPU 清理或模型启动。

提交前重新跑 A 3 tests、B 14 tests、27 组完整旧源代码/新入口契约对照、两分支 compile/AST/ruff/isort/codespell/CI 注册及脚本 mock；全部通过。完整 serving 入口缺 `psutil` 的限制未消除，也未补 NPU/e2e 数值、图片、TP/EP、精度或性能结果。计划第 2/3/6 节已同步实现状态与清单，仅勾选 CPU/静态已满足部分，硬件验收仍未完成。

发布前实时核实 #32745 / #43739 / #32602 head 未变；官方 main 为 `e775fa7253c107aece9b725a30f57beda96b32c2`。A/B 保持 `62ff466ac0` 的最小独立基线，未自动移植新 main；旧基线的验证不能算新 main 验收。既有 10 个 SGLang worktree 仍干净且 HEAD 未变，两条新 worktree 提交后干净。主仓文档与 `.pi/`、kernel 既有 gitlink 差异按原约束保留，不混入脚本提交。

证据（主仓相对路径）：

- 提交前契约/脚本复查：`.pi/tasks/01a12eff-f049-75d4-9d0f-360c93f9a30c-4113/b57ab2b41.output`
- 暂存内容 lint/static：`.pi/tasks/01a12eff-f049-75d4-9d0f-360c93f9a30c-4113/b1604fcd5.output`
- G2.2 普通 push 与 SHA / ahead-behind 核对：`.pi/tasks/01a12eff-f049-75d4-9d0f-360c93f9a30c-4113/bf00e995e.output`
- G2.3 普通 push 与核对：`.pi/tasks/01a12eff-f049-75d4-9d0f-360c93f9a30c-4113/b407aa0ce.output`
- 主仓脚本普通 push 与核对：`.pi/tasks/01a12eff-f049-75d4-9d0f-360c93f9a30c-4113/b173cb580.output`

## 2026-10-12：Diffusion MXFP8 公共执行与 Ascend MXFP 能力试点（仅本地，未提交）

建分支前通过 `git worktree list --porcelain`、本地状态和 `gh` 核实：#32745 head 为 `cf48b477dbbe0cc47aaceff89708ba84da7dfdca`，#43739 为 `ded467157750ccfd20dd2256110fad9ff104dc08`，#32602 为 `425b0a9abb4f49bab3159a8571c72485db2eb983`。从当时实时官方 main `62ff466ac0c2cdd5521acf7d4c673fa7630eb083` 创建两条独立分支，没有叠入这些功能 PR 的栈：

| 项目 | 分支 | worktree | 基线 / HEAD | 发布状态 |
| --- | --- | --- | --- | --- |
| A：Diffusion 离线 MXFP8 | `junlin/diffusion-mxfp8-shared-execution` | `sglang/diffusion_mxfp8_shared_execution/` | `62ff466ac0c2cdd5521acf7d4c673fa7630eb083` | 仅本地未提交改动；未推送、未建 PR |
| B：Ascend MXFP 局部能力 | `junlin/ascend-mxfp-capabilities` | `sglang/ascend_mxfp_capabilities/` | `62ff466ac0c2cdd5521acf7d4c673fa7630eb083` | 仅本地未提交改动；未推送、未建 PR |

既有 Dense W4A8 抽取分支 `junlin/w4a8-shared-execution` 仍为 `24e100f1e8429bde02f3e81221c4392e9e69934b`，worktree 干净。本轮没有再次抽取其执行函数，也没有更新 #43739 或其他原 PR 分支。

交付前再次用 `gh` / `git ls-remote` 核实：#32745、#43739、#32602 均仍为 Open，head 与上述 SHA 一致；A/B 的 origin 分支尚不存在，既有 Dense W4A8 远端仍为 `24e100f1e8`。origin/upstream 分别为 `TallMessiWu/sglang` / `sgl-project/sglang`。执行期间官方 main 已前进至 `28621739fc6e8084b59f0b02178413f993e35afb`；两项验证固定在启动时基线 `62ff466ac0`，未静默 rebase/merge，也不将旧基线结论声称为新 main 的验证。GitHub compare 的 5 个后续提交、33 个文件中，没有触及本轮涉及的 SRT NPU quantization、SRT/Diffusion quantization 或 `sglang/kernels` 路径；该漂移核查只读，不替代新 main 的完整测试。

### A：适配器负责 checkpoint，执行委托公共 Method

- 仅改 `python/sglang/multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py`，新增 `test/registered/unit/layers/quantization/test_diffusion_modelslim_mxfp8.py`；该 worktree 的 SRT 公共 `linear_method_npu.py` 与基线逐字节相同。
- 最新 main 的 Diffusion 在线 MXFP8 已选择 SRT `NPUMXFP8LinearMethod`；离线现在也委托该 Method，删除重复 activation quant / matmul / dtype / shape 代码。没有新建公共量化框架，也没有让执行层判断 Diffusion checkpoint 来源。
- 原始 checkpoint 参数创建和 loader 保留：`weight` 为 `float8_e4m3fn [N,K]`，scale 参数仍名为 `weight_scale`、dtype 为 `uint8`，并保留真实 parameter 类及 weight_loader；不是在加载前改为 `weight_scale_inv`。scale 按 `ceil(K/32)` 分配，保留 partial block。
- 离线 post-load 只补必要适配：奇数 scale 数用 0 字节 pad 成偶数，再由既有 SRT Method 配对并 transpose 成 `[ceil(K/64),N,2]`，weight 为 strided `[K,N]`；不权重量化、cast/pack 或新增 NZ 转换。公共 Method 将 scale 参数折入 `weight_scale_inv`，这是加载完成后的执行布局。
- 保留 FP16/BF16 输入与输出 dtype、其他输入 dtype 转 BF16、`reshape` 展平/恢复前导维度、FP32 bias、UE8M0 dtype 标记、`group_sizes=[1,1,32]`。适配器清空公共 Method 的 bias cache，保持原离线每次调用转 FP32 的语义，包括 post-load 后更新 bias 的调用者。
- **CPU/mock：3 tests passed**。post-load 覆盖 K=32/64/96/4304，含 135 个真实 scale block 配成 136 字节的边界；27 组相同 prepared weight/scale/input 的入口契约覆盖 2D/3D/4D、FP16/BF16/FP32 和 bias 有无/类型；另验非连续输入、bias 更新及 post-load 无 NPU quant/pack 调用。
- 另加载基线完整旧适配器源代码作前后对照：27 组 K=64 的 raw checkpoint 经各自真实 post-load 后，activation quant / matmul 参数、weight 与 scale 字节/stride、bias dtype/更新、输出 dtype/shape 均相同。通过的是 mock 算子调用契约，不是量化 matmul 的硬件数值等价或精度验证。

### B：局部描述实际驱动检查、既有 fallback、文档和矩阵

- 核心位于 `python/sglang/srt/hardware_backend/npu/quantization/mxfp_capabilities.py`。复用真实 `KernelSpec` / `CapabilityRequirement` / `FormatSignature`，只作为局部数据描述；不向全局 registry 注册，不按模型名分叉，不新增 SoC 查询或构建期 provider 选择。此 main 基线没有 `device_op.py`，本轮没有补造跨代际 device operator 框架。
- 17 条本 checkout 的路径描述区分 runtime、Dense/MoE、W8A8/W4A8/W4A4、online/ModelSlim、单级/双级 scale、checkpoint dtype、执行 dtype/layout、NPU/算子/backend 要求、已存在 fallback/代码阻塞及硬件验证状态。没有把其他 PR 分支的在线 W4 MoE 实现冒充为 main 已支持。
- **实际接入**：SRT / Diffusion ModelSlim selector 给选定 Scheme 附加能力；Method 在分配前校验 partition-local K，在 layout/算子前校验原始 checkpoint dtype/shape/scale；在线 Dense post-load 校验 NPU/算子/dtype；在线与离线 MoE runner 建立前校验 backend、SiLU/clamp 和所选 fused 路径的算子依赖。
- 现有 online W4A8/W4A4 MoE full-precision fallback 的返回方法不变，warning 原因改由局部描述提供；W4A8 有 clamp 时保留已有 unfused 路径，不强迫该路径具备 fused gmm1 算子。拒绝已知不兼容的 `ascend_fuseep` INT8/float-scale 契约。DeepEP normal 的 BF16 **通信**不标为 BF16 **计算**回退。
- **明确限制**：本独立 main 基线的离线 loader 仍是 floor 分配与无 padding 配对，局部 K 要求 64 对齐；不把该 loader 限制说成 MXFP8 算法不支持 partial block。B 没有夹带 A 的修复；今后合入 A/其他 partial-block 适配时需同步放宽对应能力行和测试，不能不检查就宣称两分支直接叠加后的能力矩阵成立。
- Dense online W4A4 使用双级 L0 FLOAT32 / block=512、L1 UE8M0 / block=32；MoE offline W4A4 和 packed 单级格式单列。Diffusion `W4A4_MXFP4_DUALSCALE` 的构造器实际引用未定义 `is_dual_scale_key`：直接构造测试复现 AttributeError，selector 现在提前给出明确代码阻塞错误。未修该构造器、未重打包权重，也没有把此代码问题归因为缺少硬件验证。
- 文档 `docs/docs/hardware-platforms/ascend-npus/optimization/mxfp-capabilities.mdx` 的表由同一数据生成，并从现有 Ascend quantization 文档链接；CPU 测试核对表和所有 kernel target。修正文档中 raw scale dtype、双级 L0 dtype 与 W4A8 dispatch 的陈旧描述。
- **CPU/mock：14 tests passed**，24 组 runtime/kind/scheme/source 组合及 dtype/shape/scale、FP4 enum/UE8M0、算子缺失、selector / Dense / MoE allocation 和 runner 的提前拒绝、既有 fallback、文档一致性。使用 uv 本地缓存的真实 msgspec 0.22.0 与真实 KernelSpec/Config/Method/Scheme/parameter 模块；隔离 serving 初始化和 NPU / full-precision leaf 依赖，没有安装或改动系统依赖。
- 静态核对：15 个执行/接口方法 AST 不变；`moe_methods.py`、Ascend runner 的 fused/clamp selector 与 NPU utils 的 NZ/ND fallback 文件逐字节不变；无新增运行时 SoC 分支、全局注册或静默精度回退。

### 验证和保护边界

两分支 ruff lint/format、isort、codespell、Python compile/AST、registered test 检查、无 bare pytest / package 内注册检查、空白/冲突扫描及 `git diff --check` 均通过；B 测试类名修正后又单独复查 lint/format/isort/codespell/diff。所有记录均为 CPU/mock 或静态检查，不是完整 serving 集成或 NPU e2e。另以原生 registered 入口（仅加入缓存中的真实 msgspec/orjson）复查，A/B 都在 serving 初始化阶段因缺 `psutil` 而阻塞；此 import smoke 不计为测试通过，也没有为此安装系统依赖。

复核原有 10 个 SGLang worktree 的 HEAD/干净状态未变；四个子模块 index gitlink / checkout HEAD 均未改动，kernel 仍为 index `500ea7bb0e19017ba9feb399acb0cc27bcdb46e4`、checkout `4a4f838e8fadfb5d11c4cba722ff0af72d16f8aa`。主仓 `.pi/`、已有 kernel gitlink 差异和其他用户状态保留；只向本文追加本轮记录。不提交、推送或建 PR。

待真实 NPU：记录 SoC、torch/torch_npu/CANN、kernel wheel provider、attention backend、TP/EP 与 dispatch 模式；先做权重/scale/layout 和 bias probe，再分别跑 online/offline load→warmup→graph/eager→文本/图片请求及精度/性能。算子注册只证明接口存在，不能证明某代硬件或某 TP/EP 组合可执行；此轮没有任何新的硬件、模型质量或性能通过结论。

## 2026-10-12：W4A8 Linear 公共执行重构（独立分支，已推送）

通过 `gh pr view 43739` 和 fork ref API 实时核实：[#43739](https://github.com/sgl-project/sglang/pull/43739) 为 Open Draft，head 分支为 `TallMessiWu/sglang:junlin_qwen3.5_moe_w4a8`，SHA 为 `ded467157750ccfd20dd2256110fad9ff104dc08`；与原 worktree 和本地 origin ref 一致。先执行 `git worktree list --porcelain`，确认原有 9 个 SGLang checkout 均干净，新分支/目录尚不存在，再从该精确 head 创建：

| 分支 | worktree | 基线 / 当前 HEAD | 用途 |
| --- | --- | --- | --- |
| `junlin/w4a8-shared-execution` | `sglang/w4a8_shared_execution/` | 基线 `ded467157750ccfd20dd2256110fad9ff104dc08`（#43739 head）；当前 `24e100f1e8429bde02f3e81221c4392e9e69934b` | 按 [#31783](https://github.com/sgl-project/sglang/issues/31783) 的 Scheme / Kernel 分离方向，仅复用在线/离线 W4A8 Linear 执行；无 PR，不更新原 PR 分支 |

### 改动前契约（以该基线实际代码为准）

- `NPUMXFP4W4A8LinearMethod.apply()` 与 `NPUMXFP4W4A8OfflineLinearMethod.apply()` 除注释外 AST 完全相同：FP16/BF16 输入保持 dtype，其他 dtype 转 BF16；`reshape` 展平；动态 MXFP8 activation quant；bias 转 BF16，向量扩为 `[1,N]`；FP4 `npu_quant_matmul`；恢复输入前导维度。
- 共同执行入参为已准备的 FRACTAL_NZ transpose 权重 `[K/2,N]` 与配对 UE8M0 scale `[K/64,N,2]`。动态 activation scale 原样传递，`scale_dtype` / `pertoken_scale_dtype` 为 UE8M0，`x2_dtype` 来自 FP4 dtype helper，`group_sizes=[0,0,32]`。
- 生命周期差异保留：在线创建原 dtype `[N,K]` 权重，post-load 做 FP4 quant（`round_mode="round"`）、NZ cast 后 transpose；离线 ModelSlim 创建 packed `uint8 [N,K/2]` 权重与 `uint8 [N,K/32]` scale，post-load 仅 NZ cast、scale reshape 和 transpose，不二次量化/打包。
- 非 32 对齐输入的在线 config BF16 fallback 留在原位置；MoE 已共享执行，本轮不修改 runner、融合/clamp、DeepEP normal=BF16 / low-latency=MXFP8。W4A4、Diffusion 和跨代际框架均不在范围内。

### 重构与验证

- 原调用关系：在线 method / 离线 scheme 的 kernel 各自执行一份完整的 `apply()`。新关系：两个 `apply()` 都委托同文件的 `npu_w4a8_mxfp4_linear(x, weight, weight_scale, bias)`，共享 dtype 处理、展平、activation quant、bias 准备、matmul 入参和输出 shape 恢复。公共函数只接收已准备的 weight/scale，不识别 checkpoint 来源，不处理权重量化或布局。
- AST 对比确认公共执行体与两份原执行体一致（仅将 `layer.weight` / `layer.weight_scale` 替换为参数）；整个生产模块除此公共函数和两处委托外 AST 不变，在线创建/量化与离线 post-load 分别保留，config、MoE、W4A4、MXFP8 文件均未修改。
- 扩展原有 CPU 注册测试 `test/registered/unit/layers/quantization/test_npu_mxfp4_w4a8_linear.py`。重构前 7 tests passed，重构后 8 tests passed：36 组相同 prepared weight/scale/input 的在线/离线契约，覆盖 2D/3D/4D、FP16/BF16/FP32、无 bias、FP32/FP16 向量与 BF16 二维 bias，另有两入口非连续输入和公共执行委托检查。离线真实 post-load 方法的 CPU/mock probe 确认只 NZ cast + transpose，不调用 quant/pack。
- Lint 与静态检查通过：使用本地 pre-commit 缓存中配置对应 revision 的 ruff（0.15.1）、isort（7.0.0 源码）、codespell（v2.4.1 源码）做非修改检查；ruff lint/format、isort、codespell、`check_no_bare_pytest_main.py`、全仓 `check_registered_tests.py`、`check_no_registered_tests_in_package.py`、Python compile、空白/冲突标记检查和新 worktree / 主仓文档 `git diff --check` 均通过。
- 本机 `torch 2.14.0+cpu`，无 torch_npu/NPU。普通注册测试入口在 import 阶段缺 `orjson`；上述测试加载完整目标生产模块、真实 base_config/config/量化工具与测试文件，隔离 serving 包初始化及重型依赖，使用 CPU tensor/mock NPU ops。不是完整 serving 集成、真实 NZ/算子数值验证或模型 e2e。
- 待 NPU 补验：真实 NZ 权重与 UE8M0 scale 下，以相同 prepared tensor 比较两入口（2D/多维、dtype、bias），与重构前基线比输出；分别跑在线 BF16/FP16 load→quant→warmup 和离线 packed checkpoint load→warmup，再验文本/图片请求与 graph/eager FIA。不同量化来源的权重不要求互相数值相等；精度与性能尚无新结论。

实现阶段先保留本地改动；随后按用户追加要求，仅将新 worktree 中两个 SGLang 文件提交为 `24e100f1e8429bde02f3e81221c4392e9e69934b`（`:recycle: refactor(npu): Share W4A8 execution`），普通 push 到 `origin/junlin/w4a8-shared-execution`。实时 `git ls-remote` 核对远端 SHA 与本地一致，ahead/behind 为 `0 0`，新 worktree 干净。未创建 PR，未更新 #43739 原分支；主仓已有文档改动、`.pi/` 与全部子模块指针保留，主仓文档不随 SGLang 提交。原有 9 个 SGLang worktree 的 HEAD/干净状态和 4 个子模块 gitlink/checkout HEAD 均已复核未变。

## 2026-10-12：重新创建两个关闭的 Draft PR

按用户要求用 GitHub CLI 创建替代 PR，沿用原来的 fork 分支、标题、技术正文、`base=main`、Draft 和禁止维护者修改分支的设置。正文保留原 PR 链接以追溯讨论，删除已失效的重开说明与旧 CI 自动状态块；新 PR 的 CI 状态由上游工作流生成。本轮未修改代码或推送新提交。

| 原 PR（保持 Closed） | 当前 PR | 分支 / worktree | 当前 HEAD | 创建后核验 |
| --- | --- | --- | --- | --- |
| [#32601](https://github.com/sgl-project/sglang/pull/32601) | [#43739](https://github.com/sgl-project/sglang/pull/43739) · W4A8 MXFP MoE | `junlin_qwen3.5_moe_w4a8` · `sglang/qwen3.5_moe_w4a8/` | `ded467157750ccfd20dd2256110fad9ff104dc08` | Open，Draft，MERGEABLE |
| [#34387](https://github.com/sgl-project/sglang/pull/34387) | [#43740](https://github.com/sgl-project/sglang/pull/43740) · Ascend 950 FIA mixed split | `junlin_a5_fia_mixed_split` · `sglang/a5_fia_mixed_split/` | `01bb8dc6856140c44eb3367577630e7ee0934767` | Open，Draft，MERGEABLE |

当前依赖编号：`kernel #638 →（wheel/API）#32745 → #43739 → #32602`；`#32745 → #43740`。#32745 和 #32602 的 PR 正文引用同步更新到新编号。旧 PR 的讨论、review 和 CI 历史保留在原编号下，没有迁移为新 PR 的验证结果。

两个新 PR 的 head 均与对应远端分支和本地 worktree 一致。原标签均由上游自动标记；#43739 根据当前 diff 另获 `jit-kernel` 标签。未重新执行代码测试或 NPU 验证，验证边界沿用正文中注明的 2026-10-10 检查结果。

## 2026-10-10：按依赖栈同步七个 Ascend PR

本节为 2026-10-10 的历史快照；#32601 / #34387 的当前替代编号见上节。本轮以 SGLang 官方 `main` `c55790d38b`（执行期间从 `ff19159e16` 前进，已补同步整个依赖栈）、kernel 官方 `main` `36e8a8e399` 为基线，全部使用普通 merge 和 push，未改写发布历史，所有 PR 的 base 仍为 `main`。

| PR | 当前 fork 分支 HEAD | 分支 / worktree | GitHub 状态 |
| --- | --- | --- | --- |
| [kernel #638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) | `4a4f838e8f` | `codex/a5-gemma-rmsnorm-csrc` · `sgl-kernel-npu/` | Open，非 Draft，MERGEABLE |
| [SGLang #32745](https://github.com/sgl-project/sglang/pull/32745) | `cf48b477db` | `junlin_qwen3.5_dense_w8a8` · `sglang/qwen3.5_dense_w8a8/` | Open，非 Draft，MERGEABLE |
| [SGLang #40419](https://github.com/sgl-project/sglang/pull/40419) | `756dac5a66` | `junlin_npu_mamba_ssm_dtype` · `sglang/npu_mamba_ssm_dtype/` | Open，非 Draft，MERGEABLE |
| [kernel #824](https://github.com/sgl-project/sgl-kernel-npu/pull/824) | `1c6fad69a1` | `cache-loc-assign-bounds` · `sgl-kernel-npu-worktrees/cache-loc-assign-bounds/` | Open，**非 Draft**，MERGEABLE |
| [SGLang #32601](https://github.com/sgl-project/sglang/pull/32601) | `ded4671577` | `junlin_qwen3.5_moe_w4a8` · `sglang/qwen3.5_moe_w4a8/` | **Closed，Draft；重开被 GitHub 拒绝** |
| [SGLang #32602](https://github.com/sgl-project/sglang/pull/32602) | `425b0a9abb` | `junlin_qwen3.5_moe_w4a4` · `sglang/qwen3.5_moe_w4a4/` | Open，Draft，MERGEABLE |
| [SGLang #34387](https://github.com/sgl-project/sglang/pull/34387) | `01bb8dc685` | `junlin_a5_fia_mixed_split` · `sglang/a5_fia_mixed_split/` | **Closed，Draft；重开被 GitHub 拒绝** |

七个目标 checkout 均干净，`HEAD == origin/<branch>`，ahead/behind 均为 `0 0`，官方 main 均是 HEAD 的祖先。#32601 包含最新 #32745，#32602 包含最新 #32601，#34387 包含最新 #32745。#638 与 #824 各自独立合入 kernel main，未互相叠加。

**重开阻塞已通过 GitHub CLI 再核实**：两个 PR 均由 `github-actions` 在 2026-10-03 09:25（北京时间）关闭，原因都是 Draft 已 38 天未更新；见 [#32601 关闭说明](https://github.com/sgl-project/sglang/pull/32601#issuecomment-5964087029)、[#34387 关闭说明](https://github.com/sgl-project/sglang/pull/34387#issuecomment-5964086714) 和 [关闭任务](https://github.com/sgl-project/sglang/actions/runs/37085765141)。登录用户确为作者 `TallMessiWu`，CLI token 包含 `repo` scope；上游权限为 READ（`triage=false`、`push=false`），fork 权限为 ADMIN。两个 PR 的 `viewerCanUpdate=true`、`viewerCanReopen=false`；同一账号亲自关闭的 #32266 / #32908 则为 `viewerCanReopen=true`（仅查询，未重开）。这支持“关闭来源与当前账号的重开资格有关”的判断，但 GitHub 的通用错误没有提供内部拒绝原因，不能仅凭该布尔值断言已证明是权限不足。

本轮 `gh pr reopen #32601` 仍返回 `Could not open the pull request`；两个 PR 的 REST `state=open` 均返回 422、`errors=[]`。#34387 有一次 CLI 打印 `Reopened`，但后续独立查询仍为 CLOSED、没有 ReopenedEvent；直接 GraphQL mutation 也出现无 errors 却返回 CLOSED 的情况，因此不能认定恢复成功。已排除 PR 合并/锁定、仓库归档、head/base 分支丢失及本轮改写历史：两个关闭时 head 都仍是最新分支的祖先。当前可行的下一步是请上游具备 Triage 或更高权限的维护者重开原 PR；如果维护者也失败，应把这些矛盾响应交给 GitHub Support，而不是继续推断为作者权限问题。仓库当前 workflow 没有作者评论 `/reopen` 即自动恢复的入口。

分支已推送；closed PR 的 `headRefOid` 仍停在关闭时的 `7d71e1cdf4` / `c61c0f16a2`，但 `headRef.target.oid` 已分别指向新提交。其旧 CI 不能当作新分支的 CI。成功重开后需核对 OPEN、Draft、最新 head、事件记录和合并状态。当前代码均包含本轮 main，不存在待解的合并冲突。

依赖顺序：`kernel #638 →（wheel/API）SGLang #32745 → #32601 → #32602`；`#32745 → #34387`。#40419、kernel #824 独立。#36426 只作组合验证依赖，没有混入正式分支。#808 已合并，不重开；#32041、#32266 等不在本轮重开范围内。

本轮冲突处理与验证：

- **#638 / #32745**：保留构建期 provider 选择和稳定 Gemma API。kernel 26 个 build-target/provider 测试通过；SGLang 8 个隔离 Gemma API/registry 测试通过（3 个 SRT 用例未运行），K=32/64/96/4304 的真实 post-load 与 scale 表达式 CPU probe 通过。
- **#40419**：主线的 GDN backend 和 prefill 文件保留原样，最终仅剩 `memory_pool.py` 的 14 行 dtype 修复。执行真实构造器与 NPU conv-layout helper 的 8 组隔离 CPU probe，覆盖 NPU/非 NPU、投机/非投机、FP32/BF16 配置；persistent/intermediate SSM dtype 一致。新 kernel #808 已拒绝非 BF16，旧注释同步更新。
- **#824**：保留按 `outCacheLoc.numel()` 确定 view 的修复；回归脚本补齐 `cache_loc_assign` 和 `cache_loc_update`，块末端及普通分配都覆盖 int32/int64。静态、格式和 shell 语法检查通过；本轮没有运行 NPU 算子用例。
- **#32601**：保留 main 的 scale helper、融合/clamp 策略、DeepEP normal=BF16 / low-latency=MXFP8。在线 `[E,N,K/64,2]` scale 先还原为三维块序列，再交给 helper，避免漏转置；新增 online/offline 布局回归，隔离执行通过，bias 与 fusion/clamp 检查通过。
- **#32602**：W4A8 Method 与父分支一致；保留专家单级 W4A4、Dense 双级 W4A4及 BF16 fallback 的区别。融合条件接入 main 的统一 gate，W4A4 在 ascend_tp 融合、DeepEP 保持未融合，明确拒绝不支持的 `swiglu_limit`。布局、BF16 dispatch、融合/clamp 和 Dense L0 scale 检查通过。
- **#34387**：保留主线 DCP、稀疏 KV offload、调度和图执行字段；图拒绝逻辑接入 `prefer_eager_mixed_prefill`，同时参与调度阶段 DP 投票和 forward 判定。迁移并去重 Gemma 测试。70 个 attention 用例以实际 Python 定义、CPU tensor 和 mock NPU 调用隔离执行通过；另有 8 组 TP/DP mixed/pure 图策略检查通过。

所有分支适用的 pre-commit hooks 和 PR 增量 diff 检查通过。完整 SRT 单测收集受本机缺少 serving 依赖限制；AST 提取/隔离 CPU 测试只验证控制流、形状与接口，**不是完整模块集成或 NPU e2e**。没有重新编译 910/950 wheel，也没有新 load/warmup/文本/图片/精度/性能结果，Draft 保持原样。

更新期间中间 head 的 CI 归因（日志对应最后补同步 main 之前的提交；最终 head 需重新计算）：

- #32745（`f3adade9c7`）：extra gate 缺 `run-ci-extra`；AMD 两个实跑作业在导入阶段报 `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`，属于镜像依赖冲突，未走到 NPU 逻辑。其余部分检查仍在运行。[AMD 日志](https://github.com/sgl-project/sglang/actions/runs/38039390011/job/114176493574)
- #40419（`4cf051726a`）：PR test gate 缺 `run-ci`，属于触发条件阻塞。[gate 日志](https://github.com/sgl-project/sglang/actions/runs/38039629066/job/114177111497)
- #32602（`43f51e7919`）：gate 明确打印 `PR is draft. Blocking CI.`，按要求保留 Draft。[gate 日志](https://github.com/sgl-project/sglang/actions/runs/38040104960/job/114178470258)
- kernel #638 / #824：当前仍有排队/运行中的检查；尚不能称为全部通过。
- #32601 / #34387：仍 Closed，未产生新分支 head 的 PR CI。

最终 head 再核对：五个 Open PR 均为 MERGEABLE，两个 closed PR 仍为 UNKNOWN；七个 GitHub fork ref 与本地 SHA 一致。 [#32745 `cf48b477db`](https://github.com/sgl-project/sglang/actions/runs/38040873274/job/114180691942)：AMD gate 命中用户级 120 分钟冷却（前次通过 gate 的触发时间为 2026-10-10 08:52:26 UTC），不是代码测试失败。 [#40419 `756dac5a66`](https://github.com/sgl-project/sglang/actions/runs/38040875626/job/114180698769)：gate 缺 `run-ci`。 [#32602 `425b0a9abb`](https://github.com/sgl-project/sglang/actions/runs/38040905114/job/114180781404)：Draft gate 拦截。 kernel #638/#824 的 CI 仍为 pending；SGLang 其余检查尚未全部完成。

七个 PR 的依赖及验证说明已通过 `gh pr edit --body-file` 更新；#40419 标题同步收敛为 BF16 speculative SSM allocation。保留旧性能数据的 PR 已标注这些结果早于本轮刷新，不能作为新 head 的验证。

主仓原有文档改动与 `.pi/` 保留；主仓 gitlink 未暂存或提交；测试组合分支 `main-638-824`、`junlin_qwen3.5_dense_w8a8_pr36426` 未更新。

---

> **2026-10-09 kernel 编译测试分支**：新建并推送 `main-638-824`，以当前官方 `main` 的 `84ce0315e4` 为基线，依次 squash 合入 #638（源 head `72efb124cb`）和 #824（源 head `2685ca1afc`）。分支严格为 main + `cefb0f9d`（#638）+ `e629e45a`（#824），无 merge commit；本地目录为 `sgl-kernel-npu-worktrees/main-638-824/`。26 个 CPU build-target/provider 测试通过，Python AST、shell 语法与 diff 检查通过；尚未编译 wheel 或做 NPU 硬件验证。Ascend 950 kernel wheel 使用 `bash build.sh -a kernels 950`。

> **2026-10-09 sgl-kernel-npu 分支清理**：按用户要求，开发 fork 与本地分支仅保留 `main`、`codex/a5-gemma-rmsnorm-csrc`（[#638](https://github.com/sgl-project/sgl-kernel-npu/pull/638)）和 `cache-loc-assign-bounds`（[#824](https://github.com/sgl-project/sgl-kernel-npu/pull/824)）。其余 10 个 fork 分支、对应的 5 个 worktree，以及本地临时分支 `pr651` / `pr742` 已删除，未创建备份。#808 已合并、#823 已关闭，其分支与 worktree 均已删除。SGLang 分支、本轮之前的文档改动和主仓 gitlink 保持原样。

| 当前 kernel 分支 | HEAD | 本地目录 |
| --- | --- | --- |
| `main` | `84ce0315e4` | 无独立 worktree |
| `codex/a5-gemma-rmsnorm-csrc`（#638，Open） | `72efb124cb` | `sgl-kernel-npu/` |
| `cache-loc-assign-bounds`（#824，Open Draft） | `2685ca1afc` | `sgl-kernel-npu-worktrees/cache-loc-assign-bounds/` |
| `main-638-824`（编译测试，无 PR） | `e629e45a` | `sgl-kernel-npu-worktrees/main-638-824/` |

2026-10-09：kernel 本地 `main` 与 fork `origin/main` 从 `3479f4d99c` 快进 87 个提交至官方 `upstream/main` 的 `84ce0315e4`，通过普通 push 同步；实时核对三者 SHA 一致、ahead/behind `0 0`。

以下历次核对与实验记录保留用于说明技术背景；其中已删除分支的构建、合并和对照操作不再适用，当前 kernel 分支与目录以上表为准。

> **2026-10-08 更新（仅 #32745 / #638）**：通过各自 WSL checkout 本地 merge、commit、普通 push。#32745 的 head 为 `54b0865665`（merge `a2d59e1d4b` + CI 入口兼容修复），合入 SGLang `upstream/main` `74308cf1d6`；#638 的 head 为 `72efb124cb`，合入 kernel `upstream/main` `84ce0315e4`。两者 GitHub 均为 `OPEN / MERGEABLE / BLOCKED / REVIEW_REQUIRED`，本地与 origin SHA 一致、ahead/behind `0 0`，checkout 干净。其他 PR 状态本次未复核。

> 最近核对：2026-09-14（Asia/Shanghai）；#32745 与 #638 两行于 2026-09-17 合并上游后重新核对；sgl-kernel-npu #808 于 2026-09-17 新建后核对；SGLang #40419 与 kernel 分支 `gdn-state-dtype-check` 于 2026-09-20 新建后核对，#32745 的 head 同日回退到 `4f35f3ef7e`；同日稍后按用户要求把 bf16 断言并入 #808、把 #40419 叠到 `junlin_qwen3.5_dense_w8a8_pr36426`，并重做两个 kernel 构建分支的 merge；同日 #32745 与该测试分支合入 `upstream/main` `c1a1eb5f66`（CONFLICTING → MERGEABLE）。GitHub 状态来自实时 PR 查询，本地状态来自 `git worktree list --porcelain`、`git branch --show-current` 和 `git rev-parse HEAD`。这些信息会漂移，操作前必须重新查询。2026-09-24 按用户要求复核 sgl-kernel-npu 侧 #808/#823 两对共四个分支：GitHub 两个 PR 均为 OPEN + Draft + MERGEABLE（本次已算出 mergeable，不再是 UNKNOWN），本地 head 与 GitHub head 一致，四个分支与 `origin` 零分叉、worktree 干净；**SGLang 侧 PR 状态本轮未复核**。同时更正了「两个实验分支只差 195 行」这条已过期的结论（见下文）。2026-09-24 稍后：按用户决定 **#808 保持整份不拆分**，已取消 Draft 转为 ready for review（`isDraft=false`，`BLOCKED` 仅因缺 approve；该分支无 CI checks，也未指定 reviewer）；同日补上 maintainer 要求的单算子性能用例 `tests/python/sgl_kernel_npu/test_recurrent_gated_delta_rule_vs_triton.py`（先 `ffa0542` 提出，同日把三轮修正 squash 成单提交，随后两次 amend：去掉测试文件里固化的实测数据表、精简 docstring 与注释 410→370 行 / docstring 62→33 行 / `#` 注释 25→15 行），现 head `22ec11275c`，分支共 4 提交。A/B 实测表写在 PR 描述里、不固化在测试文件中，因为数字每跑一轮都在动）并把判据实测结果写入下文。**另注意：该分支 2026-09-24 经 squash 与两次 amend 共三次 force-push 重写，之前 pull 过的 checkout 必须 `git reset --hard origin/ascend950-recurrent-gated-delta-rule`，不要 `git pull`。** 2026-09-28：CI `Lint` 在 `22ec11275c` 上失败（`black-jupyter` 改 `tests/python/sgl_kernel_npu/test_recurrent_gated_delta_rule_vs_triton.py` 的 `seed_pool` 表达式、`clang-format` 合并 `op_host/recurrent_gated_delta_rule.cpp` 的 `mix_qkv` 断言为一行），按 hook 输出逐字节修好后以 `bf59bb9cd5` 普通 push（未改写历史）；本地对该 PR 全部 7 个改动文件跑 `pre-commit run --files` 全绿。新 head 的两个 workflow run 仍停在 `action_required`（fork PR 需仓库侧批准），本账号对 `sgl-project/sgl-kernel-npu` 只有 `pull` 权限、无法点批准，lint 结果需等维护者放行后才能复核。同日稍后再核对时 #808 已从 `MERGEABLE` 变成 `CONFLICTING / DIRTY`：上游 `main` 从 `004bc36` 前进到 `3f64164`（7 笔：#791/#809/#812/#818/#819/#825/#826），与 #808 在 `csrc/pytorch_extensions.cpp` 和 `include/sgl_kenel_npu_ops.h` 冲突。两处冲突形状完全相同：#826 把 `sparse_flash_attention` 的 schema/impl/声明移出 `SGL_KERNEL_ENABLE_A3_ONLY_OPS`，#808 把 `recurrent_gated_delta_rule` 的同一组三处移到同一个插入点，git 无法自动合并。解法是两者都保留、都放在 `#ifdef` 之外（丢掉 ours 侧那个重复的 `#ifdef` 行），顺序为 `recurrent_gated_delta_rule` 在前、`sparse_flash_attention` 在后；`csrc/CMakeLists.txt` 自动合并、无需人工干预。merge 提交 `c524e45eff`（`--no-edit`，父提交 `bf59bb9cd5` + `3f64164`），已推 `origin`，GitHub 恢复 `MERGEABLE`。验证：`git diff upstream/main` 现为 7 文件，注册文件只差 rgdr 那三处搬移（各 17 行对等），`sparse_flash_attention` 仍无条件；用预处理器栈跟踪确认 rgdr 的 schema/impl/声明与 CMake 的 host/kernel 源全部落在 `A3_ONLY` 与 `A5_ONLY` 两个门之外、`m.def`/`m.impl` 各只出现一次、`#if*`/`#endif` 与括号均平衡；全树 `pre-commit run --all-files` 19 个 hook 全 Passed。2026-09-30（Asia/Shanghai）按用户要求复核 sgl-kernel-npu 侧作者为 `TallMessiWu` 的全部 PR（`--author TallMessiWu --state all`，只有 4 个）：**#808 已于 2026-09-29 11:07:18Z 由 `sglang-npu-bot` 合入 `main`**，合并形式是 squash —— GitHub 的 `mergeCommit` 为 `9e0891b24a0e51bc67cbef6cde1eab96bb03f4fb`，该提交只有一个父提交 `8b70dd1`（#828），我们分支上的 `c524e45eff` **不是** `upstream/main` 的祖先；squash 后 7 个文件的 blob 与我们分支 head 逐文件 `git diff --quiet` 全部 IDENTICAL，`arch35` 头 141 行原样在 `upstream/main` 中，即内容无损落地、只是历史被压缩（#808 的 6 个提交在 fork 上仍在，PR 页仍可见）。合并当天的位置：`3f64164`（#826）→ `f6c78ce`（#830，修 fused sigmoid gating recurrent decode 的 ssm state layout）→ `8b70dd1`（#828，KDA target verify A5）→ `9e0891b`（#808）→ `a37f707`（#815）。`upstream/main` 现为 `a37f7070da`，`#638` 落后 13 个提交；`norm/gemma_rmsnorm.py` 与 `build_tools/target_provider.py` 在 `upstream/main` 中依旧不存在，#638 仍是未覆盖。另发现仓库里还有一个此前未记入本文件的 PR：**`cache-loc-assign-bounds` → [#824](https://github.com/sgl-project/sgl-kernel-npu/pull/824)（Open Draft，MERGEABLE / BLOCKED `REVIEW_REQUIRED`，head `2685ca1afc`，2026-09-23 创建，本地 worktree `sgl-kernel-npu-worktrees/cache-loc-assign-bounds/` 与 GitHub head 一致）**，本轮一并补入下表。4 个 PR 的 GitHub head 全部与本地 checkout 一致；SGLang 侧本轮未复核。

> **注意上游这 7 笔同时引入了新的 `SGL_KERNEL_ENABLE_A5_ONLY_OPS`**（Ascend950 专属算子的对称开关，`csrc/CMakeLists.txt:43`、`csrc/pytorch_extensions.cpp:235/393`、`include/sgl_kenel_npu_ops.h:215`），CMakeLists 的注释明确要求双 arch 算子走 `SGL_KERNEL_ARCH` 而不是这两个按代际的门，rgdr 因此留在门外。CANN/torch_npu 不在本机，无法做编译或 NPU smoke，本轮只有静态与 lint 验证；新 head 的两个 workflow run 同样停在 `action_required`。

## 远程地址

| 仓库 | `origin`（开发 fork） | `upstream`（官方仓） |
| --- | --- | --- |
| SGLang | <https://github.com/TallMessiWu/sglang.git> | <https://github.com/sgl-project/sglang.git> |
| sgl-kernel-npu | <https://github.com/TallMessiWu/sgl-kernel-npu.git> | <https://github.com/sgl-project/sgl-kernel-npu.git> |

主仓 `.gitmodules` 的 `sgl-kernel-npu` URL 指向开发 fork；子模块内保留 `upstream` 作为官方同步来源。

## 历史 PR 状态快照（2026-09 至 2026-10-08；不再维护）

原表名为“当前 open PR（author: `TallMessiWu`）”。其中 PR 号、HEAD、检查状态和分支保留情况均为旧快照；现在的合入情况只查[统一跟踪表](mxfp-refactoring-plan.md#pr-merge-status)。

| 仓库 / PR | 状态 | head 分支 | GitHub / 本地 HEAD | merge-base（upstream/main） | 本地 worktree |
| --- | --- | --- | --- | --- | --- |
| SGLang [#32745](https://github.com/sgl-project/sglang/pull/32745) — Qwen3.5 dense serving on Ascend 950 | Open，MERGEABLE / BLOCKED（`REVIEW_REQUIRED`） | `junlin_qwen3.5_dense_w8a8` | `54b0865665` | `74308cf1d6`（2026-10-08） | `sglang/qwen3.5_dense_w8a8/`（主 clone） |
| SGLang [#40419](https://github.com/sgl-project/sglang/pull/40419) — 修 Qwen3.5 GDN 在 Ascend 上的 prefill 与投机解码 | Open，MERGEABLE | `junlin_npu_mamba_ssm_dtype` | `313bc46917`（3 提交：bf16 SSM state + gating 补维 + prefill 改调 AscendC chunk 算子；两笔都 cherry-pick 到 `junlin_qwen3.5_dense_w8a8_pr36426`） | `2d216a11f8`（2026-09-20） | `sglang/npu_mamba_ssm_dtype/` |
| SGLang [#32601](https://github.com/sgl-project/sglang/pull/32601) — Qwen3.5 MoE W4A8 MXFP | Open Draft，MERGEABLE | `junlin_qwen3.5_moe_w4a8` | `7d71e1cdf4` | `1fa32d50e1`（2026-08-25） | `sglang/qwen3.5_moe_w4a8/` |
| SGLang [#32602](https://github.com/sgl-project/sglang/pull/32602) — Qwen3.5 MoE W4A4 MXFP4 | Open Draft，MERGEABLE | `junlin_qwen3.5_moe_w4a4` | `65653414a3` | `1fa32d50e1`（2026-08-25） | `sglang/qwen3.5_moe_w4a4/` |
| SGLang [#34387](https://github.com/sgl-project/sglang/pull/34387) — A5 mixed chunked-prefill FIA split | Open Draft，**CONFLICTING / DIRTY** | `junlin_a5_fia_mixed_split` | `c61c0f16a2` | `1fa32d50e1`（2026-08-25） | `sglang/a5_fia_mixed_split/` |
| sgl-kernel-npu [#638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) — portable Gemma RMSNorm API | Open，MERGEABLE / BLOCKED（`REVIEW_REQUIRED`） | `codex/a5-gemma-rmsnorm-csrc` | `72efb124cb` | `84ce0315e4`（2026-10-08） | `sgl-kernel-npu/` |
| sgl-kernel-npu [#824](https://github.com/sgl-project/sgl-kernel-npu/pull/824) — cache_loc 算子的 `out_cache_loc` view 按张量本身定长 | Open Draft，MERGEABLE / BLOCKED（`REVIEW_REQUIRED`，无 review） | `cache-loc-assign-bounds` | `2685ca1afc` | `004bc36`（2026-09-23） | `sgl-kernel-npu-worktrees/cache-loc-assign-bounds/` |

核对时 6 个 GitHub head SHA 均与本地 checkout 完全一致，6 个 SGLang worktree 与 kernel checkout 均无未提交改动。#808 创建后 GitHub head 与其 worktree 的 `783b44c911` 一致，worktree 无未提交改动。

两处相对 2026-08-25 的变化：

- **#34387 已变成 `CONFLICTING`**，上游在 8-25 之后动过它碰的文件，需要重新基于 `upstream/main` 合并。其余四个 SGLang PR 仍是 `MERGEABLE`。
- **#638 的 `ping1jing2` review 已于 2026-08-06 变成 `DISMISSED`**，之后无新评论（最后更新 2026-08-20）。五个 SGLang PR 全部卡在 `REVIEW_REQUIRED`，没有 approve。

参考基线（2026-09-17 fetch 的 `upstream/main`）：SGLang `4fb9b5b5ba`、sgl-kernel-npu `67a38fe215`（同日二次 fetch，含 #802）。#32745 与 #638 已合入该基线（落后 0，GitHub head 与本地一致，仍是 MERGEABLE / BLOCKED）；#32601、#32602、#34387 仍以 `1fa32d50e1` 为 merge-base，各落后 1206 commits（#32266 同样落后，已于 2026-09-18 关闭）。

**#32745 / #638 同步最新上游（2026-10-08）**：#32745 仅在旧路径 `test/registered/kernels/ops/layernorm/test_kernels_namespace.py` 出现 modify/delete 冲突，上游 #41243 已将测试移到 `test/registered/unit/kernels/test_kernels_namespace.py`。删除旧文件、保留新文件的全部上游用例，只把 NPU `_GEMMA_RMSNORM` 默认 provider 期望改为 `sgl_kernel_npu`；merge 时相对上游为 10 文件 / +351 / −21，随后为 Gemma pytest 测试补上 CI 要求的 `__main__` 入口（`54b0865665`，+4 行），最终为 10 文件 / +355 / −21。#638 无冲突，相对上游仍为 14 文件 / +818 / −120。两个 merge 均保留原 head 与 upstream/main 为双亲。验证：各自最终 PR diff 的全部适用 pre-commit hooks 通过，kernel 构建/provider CPU 测试 26 passed；SGLang 用真实 torch 与 kernels、隔离顶层 serving 依赖运行 registry/provider 测试 14 passed / 3 deselected（SRT 测试未跑），Python AST 检查通过。新 CI 在合并 head 上暴露了缺少 `__main__` 入口导致所有 suite 收集失败的问题；已本地复现并修复，用真实收集器验证全仓 2310 个文件 / 2826 条注册记录通过，新增入口按 CI 的 `-f` 调用方式运行 8 passed / 3 deselected，文件 lint 通过。这些检查不构成 NPU e2e 验证。

新 head 已触发 CI：#638 [构建](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/37739621665)、[Lint](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/37739621664)、[算子测试](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/37739621795)、[DeepEP](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/37739621699) 中 Lint 已成功，其余运行或排队，950 的 `test-deepep-a5 (9.1.0)` 已开始；原 head 上的 DeepEP 失败为 HCCL 初始化/socket timeout，发生在参考 `dist.all_reduce`，尚不能据此认定 Gemma provider 回归。#32745 入口修复后的 head `54b0865665` 已重新触发 CI，[Lint](https://github.com/sgl-project/sglang/actions/runs/37740161388) 在运行；[NPU CI](https://github.com/sgl-project/sglang/actions/runs/37740161676) 被 `pr-gate` 的 120 分钟冷却规则阻断，日志记录上次放行时间为 `2026-10-08T06:47:51Z`，按该快照北京时间 16:48 后可再尝试，或由高权限维护者触发；[AMD Extra gate](https://github.com/sgl-project/sglang/actions/runs/37740161674) 因缺少 `run-ci-extra` 标签失败，测试未执行。本账号对上游只有 `pull` 权限（`push`/`triage` 均为 false），需要维护者加该标签。上述为本次查询快照，CI 最终结果需重新核对。

## 历史上游覆盖核对（按下述基线，不代表当前 main）

逐项对照过 `upstream/main`（2026-09-30 复核），**除已合入的 #808 外，没有任何一个 open PR 的核心功能被完整覆盖**，可以继续推进。已经被上游吃掉的是 #32601/#32602 的离线部分。

| PR | 覆盖 | 说明 |
| --- | --- | --- |
| #32745 | 未覆盖 | `GemmaRMSNorm.forward_npu` 与 `kernels/ops/layernorm/__init__.py::GemmaRMSNormOp.forward_torch_npu` 仍直接调 `torch_npu.npu_gemma_rms_norm`；`KernelBackend` 里 `sgl_kernel_npu` 仍只是 `TODO(RFC #29630)` 注释。仅有的缓解是 `SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM`（2026-02 为 Skywork-Gemma2 加的 native 回退，有性能代价）和 `residual is not None` 时的 `add_gemma_rms_norm` triton 路径；`residual=None` 的裸 Gemma RMSNorm 在 A5 上依旧落到未注册算子 |
| #32601 | 部分 | 已在 upstream：`NPUW4A8MXFP4MoEMethod`（离线，#30318）、Linear 侧 `NPUMXFP4W4A8LinearMethod`。仍缺：`npu_mxfp4.py` 的 `FusedMoE` 分支（还是 warning + `UnquantizedFusedMoEMethod`）、`NPUW4A8MXFP4OnlineMoEMethod`、K%32 非对齐 Linear 回退 |
| #32602 | 部分 | 已在 upstream：`NPUW4A4MXFP4MoEMethod`（离线，#30319）、`NPUDualLevelMXFP4LinearMethod`、`npu/moe/matmul.py` 里的 `npu_grouped_matmul_swiglu_quant_v2` 封装。仍缺：`npu_mxfp4_w4a4.py` 的 `FusedMoE` 分支、`NPUW4A4MXFP4OnlineMoEMethod`、W4A4 的 `apply_fused_gmm1_swiglu`（上游 `apply` 走非融合四算子）、runner 的 `fuses_gmm1_swiglu` 泛化、`HiddenStatesDynamicQuant` 的 lazy op 绑定、K%32 回退 |
| #34387 | 未覆盖 | upstream `forward_mixed` 仍是单次 FIA 调用；`hardware_backend/npu/` 下没有 `device_op.py`；`environ.py` 里没有 `SGLANG_NPU_FIA_MIXED_SPLIT` |
| #638 | 未覆盖 | upstream kernel 仓没有 `norm/gemma_rmsnorm.py`，`setup.py` 没有构建期 provider staging。`build.sh` 虽已有 `Ascend950PR_9599` 的 kernels target，但不做 Gemma provider 选择。新增的 `norm/split_qkv_rmsnorm_rope.py::split_qkvgate_gemma_rmsnorm_rope` 是融合 QKV+RoPE 的另一条路径，不替代独立的 GemmaRMSNorm 调用；#733 的 `split_qkv_rmsnorm_mrope.py` 不含 Gemma 路径。上游 release workflow 已按 #803 为 950 单独出包（`./build.sh -a kernels Ascend950PR_9599`），合入 #638 后该包会 stage ACLNN provider |
| #808 | **已合入** | 2026-09-29 以 `9e0891b24a` 落地：host/kernel/schema/声明全部移出 `SGL_KERNEL_ENABLE_A3_ONLY_OPS`，新增 `op_kernel/arch35/recurrent_gated_delta_rule_regbase.h`（141 行）并把 4 条 bf16 dtype 断言写进 `op_host/recurrent_gated_delta_rule.cpp`。SGLang 侧仍需 #40419（`MambaPool` 投机路径建 bf16 state）+ 二维 beta 补维才能不吃断言 |

## 跨仓配对与历史验证记录

本节保留配套 API、构建和历史实验的依据；当前待合入前置与已合入项以[合入依赖说明](mxfp-refactoring-plan.md#merge-dependencies)为准。

**#638 与 #32745 是强配对，必须一起合。** #32745 的 `from sgl_kernel_npu.norm.gemma_rmsnorm import npu_gemma_rms_norm` 带 `except ImportError` 回退到 `torch_npu.npu_gemma_rms_norm`；#638 未合、wheel 里没有该模块时，#32745 单独合入在 A5 上等于没改，反之亦然。provider 由 `SGL_KERNEL_NPU_BUILD_TARGET` 在构建期选择：`target_providers/Ascend910/norm/gemma_rmsnorm.py` 或 `target_providers/Ascend950/norm/gemma_rmsnorm.py` 被 staging 成 `norm/gemma_rmsnorm.py`。

**#734 已并入 #638（2026-09-17）**：`wangyao-i` 的 [#734](https://github.com/sgl-project/sgl-kernel-npu/pull/734)（open，分支 `feat/pr638-continue`）基于 #638 的 `99421ab5b6`，只多一个提交 `1def0a4f14`，把 Gemma 专用的 `GEMMA_PROVIDERS` staging 泛化为 `build_tools/target_provider.py` + `target_providers/<target>/` 目录，并要求构建 wheel 时必须设置 `SGL_KERNEL_NPU_BUILD_TARGET`。该提交已 `cherry-pick -x` 到 #638 本地分支（`8df27879fe`，作者保留为 `wangyao-i`），唯一冲突在 `python/sgl_kernel_npu/README.md`，保留了 `Ascend950PR_9599` 的编译目标说明。并入后 #734 的改动全部包含在 #638 中。原提交漏了在 `find_namespace_packages` 中排除 `build_tools`，wheel 会多装一个顶层 `build_tools` 包；随后补 `8cf80e44a6` 修复：`setup.py` 同时排除 `build_tools`，原先断言 exclude 字符串的测试改为断言 wheel 顶层包只有 `sgl_kernel_npu`（该测试在修复前失败）。两个提交均已推送到 #638。

#32745 的 `installation.mdx` 说明已于 2026-09-17 改为（`02bc394020`）：A5 用户安装 `950` release 包，或用与 release 相同的 `bash build.sh -a kernels Ascend950PR_9599` 从源码构建。

**A5 编译目标（2026-09-17 已修，#638 `d884b4c94c`）**：合入的上游 #724/#750 让 `csrc/CMakeLists.txt` 按 `^ascend950` 选择 `arch35` 并启用 A5 专属算子（`kv_compress_epilog`、`swiglu_group_quant`），而 #638 原先把通用 `950` 和 `npu-smi` 检测到的 A5 映射到兼容目标 `Ascend910_9382`，构建出的 A5 包会用 `arch22` 编译 compressor、缺少 A5 专属算子，和官方 950 包不一致；上游 SGLang #37373 的 DSV4 在 A5 上会直接调用 `torch.ops.npu.kv_compress_epilog`。现在 kernels 目标下这两种情况都编译为 `Ascend950PR_9599`（与官方 950 release 包一致）；显式传入的 `Ascend950PR_*`/`Ascend950DT_*` 仍原样透传（测试用例为 `Ascend950PR_958b`）；不带 `-a` 的完整构建在 A5 上直接报错，因为 DeepEP 与 kernels 共用一次 CMake 配置，而 DeepEP 固定使用 A3 的 SoC 值。自动检测不区分 A5 具体型号，非 `9599` 型号需显式传入。Gemma provider 不受影响。本仓 `docs/sgl-kernel-npu-build.md` 与 `sgl-kernel-npu-dev` skill 已同步（主仓 `7152f9f`）。只做了 CPU 验证（build-target 测试 18 passed、SoC 解析场景模拟），未在 A5 上实际构建。

**`causal_conv1d` 在 Ascend 950 上的注册（2026-09-17，已由上游 #802 解决）**：改用原生目标 `Ascend950PR_9599` 后，A5 上跑 Qwen3.5 BF16 在 decode 图捕获时报 `torch.ops.npu.causal_conv1d` 不存在。原因是上游 #632 为了让 A5 先编过，把一批 host 算子统一划成 A3 专属，并没有逐个验证；`causal_conv1d` 的设备 kernel 一直在编，只关掉了 host 端和注册。对照 vllm-ascend `csrc/build_aclnn.sh` 的按 SoC 清单，`causal_conv1d`、`recurrent_gated_delta_rule`、`sparse_attention_score` 都是 A3/Ascend 950 通用；其余 A3 专属项（`lightning_indexer`、`sparse_flash_attention`、`sparse_attn_sharedkv`、`mla_preprocess`、`batch_matmul_transpose`、稀疏 KV offload）与 vllm-ascend 一致。#638 曾用 `7afbf61538` 单独放开 `causal_conv1d`；同日上游 [#802](https://github.com/sgl-project/sgl-kernel-npu/pull/802)（`zhaozx-cn`，support kimi k3 on A5）合入了同样的改动，#638 因此变成 CONFLICTING。按用户要求删掉 `7afbf61538`，再次合并 `upstream/main` `67a38fe215`（merge `500ea7bb0e`，无冲突，`--force-with-lease` 推送），PR 描述也删掉了 `causal_conv1d`。现在 #638 的 `csrc/`、`include/` 与上游完全一致，GitHub 恢复 MERGEABLE / BLOCKED；CPU 测试 26 passed。Ascend 950 上仍需用该分支重新构建（原生目标约 45 分钟）后跑 `tests/python/sgl_kernel_npu/test_conv1d_prefill.py` 和 `llm/qwen3.5_dense_bf16.sh`。

**`recurrent_gated_delta_rule` 的 Ascend 950 适配（2026-09-17 开 Draft [#808](https://github.com/sgl-project/sgl-kernel-npu/pull/808)）**：用于在 Ascend 950 上给 Qwen3.5 开 MTP（NEXTN target verify 调用该算子，SGLang #20918 引入）。分支 `ascend950-recurrent-gated-delta-rule` 基于 `upstream/main` `67a38fe215`，单提交 `783b44c911`，worktree 在 `sgl-kernel-npu-worktrees/ascend950-recurrent-gated-delta-rule/`（见下文 kernel worktree 结构）。做法与上游 #802 的 `chunk_kda_fwd`、`causal_conv1d` 相同：host、kernel、schema、impl 与头文件声明移出 `SGL_KERNEL_ENABLE_A3_ONLY_OPS`，schema 不变；kernel 保持单一源文件和原有 pipeline（stage buffer、MTE2/V/MTE3 事件、`mix_qkv` 拆分、intermediate state 初始化、speculative token 之间 FP32 state 延续），只在 `__CCE_AICORE__ == 310` 下把 arch22 专属的 repeat-stride `Mul`/`MulAddDst`、`Brcb`、mask 寄存器 `ReduceSum`、`Sum`/`Rsqrt` L2 norm 换成 `op_kernel/arch35/recurrent_gated_delta_rule_regbase.h` 的 MicroAPI 实现（`RowDotRegbase`、`RankOneUpdateRegbase`、`L2NormalizeRowsRegbase`，循环写法参考 vllm-ascend #9224、#9382，L2 norm 用 arch35 RMSNorm 同款序列）。arch35 不再切 host UB 预算外的 `qTempInUb`/`kTempInUb`/`qSumLocal`/`kSumLocal`；host 在 arch35 上用 `GetCoreNumAiv()` 作为 block 数，910 仍是 `GetCoreNum()`。按用户决定 `MAX_MTP` 保持 8（#11236 的 16 另开 PR），以 Draft 提出。只做了静态与 CPU 验证：clang-format 18.1.8、codespell、`git diff --check`；只展开新增条件后 910 视角的 kernel/host 源码与 `main` 逐字节一致；numpy 模拟 arch35 寄存器循环（64 lane、`UpdateMask` 尾块、BRC/首元素 load-store、masked store、多 block 分片、intermediate/recurrent 初始化、`num_accepted_tokens`）对照测试文件的 golden，8 组形状（含 dk=100/200、dv=72/136 和多 vStep 分块）float64 最大绝对误差 2.2e-16，只证明下标与循环边界。**未做**：950 上 `bash build.sh -a kernels Ascend950PR_9599` 构建、`hasattr(torch.ops.npu, "recurrent_gated_delta_rule")`、`python3 tests/python/sgl_kernel_npu/test_recurrent_gated_delta_rule.py`（脚本数值不一致也退出 0，须看 `failed: N`）、Qwen3.5 NEXTN e2e（依赖 #638 + #32745），以及 910B/910C 回归。#651/#742 的 BF16 往返改动落在两个 arch 共用的 `Compute` 代码里，任一方先合入后 rebase 即可对齐。

**只放开编译的对照分支（2026-09-23 建，已推 `origin`）**：同事反馈在 Ascend 950 上「只把这个算子从 A3-only 放开、kernel 不动就能跑」。为验证，从 #808 的 base `004bc36` 拉两个分支做 A/B（按用户要求不 fetch 上游，base 与 #808 完全一致）：

- `ascend950-rgdr-ungate-probe` → [#823](https://github.com/sgl-project/sgl-kernel-npu/pull/823)（已 **Closed** 2026-09-24，关闭时附了实测结论的评论；head `26c905a`，分支与 worktree 保留）：只改 `csrc/CMakeLists.txt`、`csrc/pytorch_extensions.cpp`、`include/sgl_kenel_npu_ops.h` 三个注册文件，把 host 源、kernel 源、schema、impl 与声明移出 `SGL_KERNEL_ENABLE_A3_ONLY_OPS`。这三个文件与 #808 head 逐字节一致；kernel 与 host 一行未动，仍是 arch22 向量块、`GetCoreNum()` 和完整 UB 切分，**也不含** bf16 dtype 断言。2026-09-24 实测 `git diff 823 808`：**3 文件 / +195 / −0**，即新增 `op_kernel/arch35/recurrent_gated_delta_rule_regbase.h` +141、`op_kernel/recurrent_gated_delta_rule_kernel.cpp` +35（`__CCE_AICORE__ == 310` 分支）、`op_host/recurrent_gated_delta_rule.cpp` +19（4 条 bf16 断言 + `#ifdef SGL_KERNEL_ARCH_35` 下 `GetCoreNumAiv()`）。所以 #823 是 #808 在**树层面**的严格子集，两臂可直接对照。
- `ascend950-mtp-experiment-ungate`（worktree 同名，head `476b1b9`）：上一条 + #638 `500ea7bb0e` + #742（含 #651）`185d8a7`，对齐 `ascend950-mtp-experiment` 的组合。#742 与 base 里 #747 在 `mamba/causal_conv1d.py`、`test_conv1d_update.py`、`test_mamba_state_update.py` 上冲突，直接取 `ascend950-mtp-experiment` 的已解决版本（这三个文件 #808 没碰，取完与该分支逐字节一致）。~~与 `ascend950-mtp-experiment` 的全部差异正好是 #808 的 kernel/host 195 行。~~ **2026-09-24 更正**：`476b1b9` 11:16 建，`f30a2d9` 11:23 才把 `cache-loc-assign-bounds` `2685ca1` 合进来，两者差异**不止 195 行**——实测 `git diff ascend950-mtp-experiment-ungate ascend950-mtp-experiment` 为 **6 文件 / +286 / −5**，两组：(1) #808 的 arch35 三文件 195 行；(2) cache-loc 修复——`csrc/cache_location_assign/op_host/cache_loc_assign.cpp` 18 行（改用 `outCacheLoc.numel()` 而非 `batchSize * MAX_STEP`）、新增 `tests/python/sgl_kernel_npu/test_cache_loc_bounds.py` 76 行、`scripts/run_kernel_tests.sh` +2。**所以直接拿这两个 worktree 做 wheel A/B 是混淆的**：比的是「#808 arch35 + cache-loc」vs「两者都没有」。干净 A/B 用两个 PR 分支（#808 vs #823），或先给 ungate 补上 `2685ca1` 再比。

判据：950 上 `bash build.sh -a kernels Ascend950PR_9599` 是否编过，以及 `test_recurrent_gated_delta_rule.py` 的 `failed: N`（脚本数值不一致也退出 0）。两个 wheel 的精度与性能 A/B 用 `llm/diagnostics/recurrent_gated_delta_rule_check.py`：`--bench N --dump <file>` 各跑一次，再 `--compare A B` 逐元素比输出并列耗时（2026-09-23 加，CPU 上只验过 `--dry-run` 与 `--compare`，NPU 路径未跑）。**读结论前必须先确认构建目标**：CMake 会打印 `SGL_KERNEL_ARCH=... SGL_KERNEL_ENABLE_A3_ONLY_OPS=... (SOC=...)`，只有 `arch35` 才说明问题；若是兼容目标 `Ascend910_9382`，那是 arch22 且 `A3_ONLY_OPS` 默认 ON，算子本来就编进去了，「放开就能跑」不成立。若原生 950 上放开即可用，#808 的 arch35 MicroAPI 部分应从「能否运行的前提」降级为性能优化，并拆成独立 PR（最小放开 + bf16 断言先合）。

**2026-09-24 判据落地（两个 wheel 各跑 `--matrix` + 跨 wheel `--compare`）**：bf16（连续/strided 两种布局 × 4 组形状，到 MTP=8）两边全部 `超容差元素=0`，对 golden 的最大误差**逐格相同**（out `9.500~9.763e-04`，state `1.953e-03~3.850e-03`），其中 state `3.835e-03` 距容差绝对下限 `2⁻⁸=3.906e-03` 只剩 **1.8%** 余量。两边彼此的差是 bf16 单 ULP（`1.953e-03`，连续与 strided 都复现同样 6 项），属归约顺序不同而非算法差异；`--matrix` **分辨不出**这 1 ULP（误差峰值由两边共有的 bf16 量化主导），只有 `--compare` 能。**真正的分水岭是断言**：#808 对 fp32 state 直接 `RuntimeError: recurrent_state must be bfloat16`；#823 无断言，静默算出 1e35/NaN（MTP=8 时 `14886654/18874368` 个元素坏，即已踩过的那个「输出通顺但逐渐失忆」故障模式回来）。单算子耗时 #808 快 1.4~4.6×（连续 `1.43/2.17/3.33/4.37`，strided `1.59/2.09/3.21/4.60`；同 wheel 重跑的噪声基线 ≤14%，故全部在带外），属微基准，e2e 收益未测。**决定：不拆分，走整份 #808。**#823 已于 2026-09-24 关闭（关闭评论里带上了上面的实测结论），分支与 worktree 保留、远端分支未删；未收尾项：`ascend950-mtp-experiment-ungate` 那份 wheel 的构建目标仍未确认（`hasattr(torch.ops.npu,'kv_compress_epilog')` 或构建日志的 `SGL_KERNEL_ARCH=`）——若为原生 950，则说明 arch22 向量块 API 在 arch35 上仍能编能跑、只是慢，这是一个可写进 PR 的附带结论。

**Ascend 950 上跑 Qwen3.5 MTP 的分支组合（2026-09-20 核对，同日合上游后更新；kernel 行 HEAD 于 2026-09-24 复核，SGLang 行本轮未复核）**：

| 仓 | 目录 | 分支 | HEAD | 叠了什么 |
| --- | --- | --- | --- | --- |
| SGLang | `sglang/qwen3.5_dense_w8a8_pr36426/` | `junlin_qwen3.5_dense_w8a8_pr36426` | `1eca006f8d` | #32745 + 上游 #36426 + #40419（cherry-pick），base `c1a1eb5f66` |
| sgl-kernel-npu | `sgl-kernel-npu-worktrees/ascend950-mtp-experiment/` | `ascend950-mtp-experiment` | `f30a2d9e58` | base `004bc36`（已含 #802、#804、**#747**）+ #638 + #808 + #742（含 #651）+ `cache-loc-assign-bounds` `2685ca1` |

历史构建组合（上述两个分支已于 2026-10-09 删除，不再作为构建入口）：当时 wheel 使用 `ascend950-mtp-experiment` 而非 `ascend950-mtp-integration`，原因是后者的 `move_intermediate_cache` 仍是 `h_block_size=2`，在 950 上会 `ub overflow, requires 2097152 bits while 2031616 bits available`；#651（含在 #742 里）把它改成 1。（2026-09-28 备注：该分支仍以 `004bc36` 为底，没有这次的上游 `3f64164`；上游新 7 笔与本组合的算子无关，重做该分支时若把 #808 的新 head `c524e45eff` 合进来，会同样撞上 `pytorch_extensions.cpp` / `sgl_kenel_npu_ops.h` 的 `sparse_flash_attention` 冲突，解法同上。）`llm/qwen3.5_dense_bf16.sh` 的 MTP 开关不再设 `SGLANG_MAMBA_SSM_DTYPE`，改由 #40419 在 `MambaPool` 里自动覆盖，启动日志出现 "not supported by the NPU speculative verify kernels" 即生效。

**Qwen3.5 MTP 在 Ascend 950 上复读的根因（2026-09-20 定位并修复）**：开 NEXTN 后输出流畅但几十步内塌成复读，接受率从 ~2.5 一路爬到满值 4.00；关掉 MTP 一切正常。根因是 SSM state 的 **dtype 位宽不匹配**，不是 layout。`torch.ops.npu.recurrent_gated_delta_rule` 只有 `RGDR<bfloat16_t, bfloat16_t>` 一份实例化，host 的 UB 预算按每元素 2 字节算（`coeff = (2 + 2) * aDk + 4`），且 `EXEC_KERNEL_CMD` 前没有任何按 dtype 的分发；而 `MambaPool` 的 `ssm_dtype` 默认是 `torch.float32`（`configs/mamba_utils.py`，可由 `mamba_ssm_dtype` 或 `SGLANG_MAMBA_SSM_DTYPE` 覆盖）。位宽对不上不会报错：kernel 用 2 字节步长走 4 字节 buffer，每隔一个元素落在 fp32 的高半字，而 fp32 的高 16 位正是它自己的 bf16 截断 —— 数值全程留在合理量程内，state 却是错位的，所以表现是文字通顺地逐渐失忆，而不是 NaN。只有 verify 路径会踩到：prefill/decode 走 triton，dtype 从张量上读。定位靠的是在真机 verify 里同时跑算子和 torch 参考的探针（`llm/patches/patch_gdn_verify_probe.py`）拍到 state 是 `torch.float32`，再用 `llm/diagnostics/recurrent_gated_delta_rule_check.py --matrix` 只换 dtype 复现：bf16 最大误差 9.8e-04，fp32 返回 1e35 量级。两处修复：SGLang [#40419](https://github.com/sgl-project/sglang/pull/40419) 让 `MambaPool` 在 NPU 投机路径直接建 bf16；算子 host 补 `mix_qkv`/`beta`/两个 state 张量的 dtype 断言，让这种不匹配不可能再静默 —— 按用户要求并入 #808 的第二个提交 `6734df4e54`（#808 是 Draft，可直接拿来测），原独立分支 `gdn-state-dtype-check` 作废。和 #808 无关，910B/910C 同样会中。过程中被证伪的假设记录在案：state 转置（#693 思路）、conv 算子的 spec 分支、两个 mamba state triton kernel、state 提交的 off-by-one，全部排除。

**#32745 合入 `upstream/main` `c1a1eb5f66`（2026-09-20，CONFLICTING → MERGEABLE）**：落后 166 个提交，只有 `npu_cudagraph_backend.py` 冲突。上游 [#39589](https://github.com/sgl-project/sglang/pull/39589) 独立修了同一个竞态 —— `replay_with_input_update` 原本在后台线程里 rebind、主线程直接 `graph.replay()`，`thread.join()` 在 replay 之后，rebind 还在飞的时候 replay 已经到驱动的 BindSqCq，`rtModelExecute` 失败（retCode 0x7020023，报成 Insufficient_Resources(EL0006)）。我们的修法是放回调用线程同步做；上游改成复用一个 device-bound worker（`ThreadPoolExecutor(max_workers=1, initializer=self._device_module.set_device, initargs=(self._device_id,))`）并 `update_future.result()` 后再 replay，顺序保证相同，而且 worker 的 device 绑定在 initializer 里一次做好、不用每次 replay 建线程。上游版本更完整，所以整个文件取上游，`74d99befdb` 的改动作废，#32745 的有效 diff 从 11 个文件降到 10 个。其余 10 个文件自动合并无冲突。测试分支 `junlin_qwen3.5_dense_w8a8_pr36426` 再合 #32745 时零冲突（#36426 的 4 个文件与上游新提交不重叠），#40419 的 14 行原样带过。**未做**：本机没有装好 sglang 的 Python 环境，`test_modelslim_mxfp8.py` / `test_npu_gemma_rmsnorm.py` / `test_kernels_namespace.py` 这轮没跑，只做了 AST 解析与冲突标记检查；按仓库约定大规模 merge 后要在 NPU 机器上先重跑 baseline 再叠新改动。

**SSM state layout 的跨仓契约：SGLang #39589 必须配 sgl-kernel-npu #747（2026-09-20 踩到）**：`Mamba2StateShape.temporal` 是 `(num_heads, head_dim=V, state_size=K)`，也就是 `(HV, V, K)`。历史上 NPU 投机路径靠 `memory_pool.py` 里的 `temporal_state.transpose(-1, -2)` 把逻辑视图翻成 `(HV, K, V)` 去迁就 FLA triton kernel 的 `b_h[BK, BV]`，同时物理排布留在 `(HV, V, K)` 给 AscendC 算子按物理内存读 —— 两个消费者各看各的，正好都对。kernel 仓 [#747](https://github.com/sgl-project/sgl-kernel-npu/pull/747)（`67be199`，2026-09-17 合入）把 triton kernel 统一翻成 `b_h[BV, BK]`（源码注释 `# V row K col`、`# V dim stride = K`），SGLang [#39589](https://github.com/sgl-project/sglang/pull/39589)（`8ac39c66d8`，2026-09-18 合入）随之删掉那个转置。**两者必须成对**：wheel 停在 #747 之前而 SGLang 已过 #39589，prefill（triton）与 verify（算子）对同一块内存的解释就差一个转置，表现是 MTP 一开口就乱码截断（不是之前那种先通顺再复读）。只影响投机路径 —— 非投机时那个转置本来就不生效，triton 是唯一消费者，自洽所以两种 wheel 都能跑。判据：关掉 MTP 正常、开 MTP 立刻乱码。修法是把 kernel 分支合到含 #747 的 `upstream/main` 后重建 wheel；`67a38fe..004bc36` 只有 2 个提交，#747 动的三个注册文件（`csrc/CMakeLists.txt`、`csrc/pytorch_extensions.cpp`、`include/sgl_kenel_npu_ops.h`）与 #808 自动合并无冲突，合完 `fla/` 与上游逐字节一致、#651 的 `h_block_size=1` 保留。

**#747 的第二处断裂：gating 输出降维（2026-09-20）**：除了 SSM state layout，[#747](https://github.com/sgl-project/sgl-kernel-npu/pull/747) 还把 `fused_gdn_gating_npu` 的输出从 `torch.empty(1, batch, num_heads)` 改成 `torch.empty(batch, num_heads)`，为的是对上同一个 PR 新增的 AscendC 算子 `chunk_gated_delta_rule`（`beta` 是 `(T, Nv)` 二维）。但 SGLang 侧至今没跟上：`layers/attention/linear/kernels/gdn_triton.py` 仍然 `from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu`（上游 `efa7be2091` 核对过），而那条 triton 路径断言 `len(beta.shape) == 3`。于是**带 #747 的 wheel 配任何版本的 SGLang，GDN prefill 都直接 AssertionError**，和开不开 MTP 无关。修法放在 SGLang [#40419](https://github.com/sgl-project/sglang/pull/40419) 的第二个提交：调 `kernel_dispatcher.extend` 前按 `g.dim() == 2` 补回首维，旧 wheel 不受影响。verify 路径不用改，它走 `fused_gdn_gating_kernel_without_sigmoid`（`torch.empty_like(a)`，#747 没动）且本来就有 `unsqueeze(0)`。长期的正路是让 SGLang 改调 #747 新增的 `fla.chunk_gated_delta_rule_npu`（原生吃二维 beta，但签名用 `actual_seq_lengths` 而非 `cu_seqlens`、返回值少一个 `h`），那是上游该补的配对 PR。运行时补丁见 `llm/patches/archive/patch_gdn_gating_shape.py`。

**#747 的第三处断裂：prefill 的 state 布局（2026-09-20，Qwen3.5 MTP 在 950 上最终跑通）**：#747 把 pool、decode triton、verify 算子统一成 `(nv, dv, dk)`，**但没有动 prefill 走的 `sgl_kernel_npu.fla.chunk`**，它仍返回 `(nv, dk, dv)`。SGLang 的 `ssm_states[cache_indices] = last_recurrent_state` 因此把转置后的内容写进 pool —— `dk == dv == 128` 时形状相同，不报错。用 `dk=64, dv=32` 实测（`llm/diagnostics/gdn_prefill_layout_check.py`，模型自身的形状永远分辨不出来）：triton 返回 `(1,4,64,32)`、#747 新增的 AscendC 算子返回 `(1,4,32,64)`，两者 attention 输出一致（128 token 时差 3.6e-02，bf16 累积量级）。修法是让 prefill 改调 `torch.ops.npu.chunk_gated_delta_rule`（`llm/patches/archive/patch_gdn_prefill_ascendc.py`，已提交进测试分支 `62644c2253`）。**这一处加上前两处（bf16 state、gating 降维）之后，950 上的 Qwen3.5 MTP 才真正跑通。**已按上游质量补完并并入 #40419：算子的 chunk 网格是 `sum_b ceil(len_b/64)`，与 GDN 在 `_init_track_ssm_indices` 里建的那套逐字相同，写的又正是"每块的进入态"，所以 `chunk_state` 直接当 `h` 返回，page 跟踪照常工作，而且行布局是 pool 的 `(nv, dv, dk)`（triton 那份 `h` 和它整条路径一样是转置的）。入参改成一次 `reshape` 直接喂给 `l2norm_fwd`，比先 `.contiguous()` 少一次拷贝；剩下的开销是算 chunk 数时每层一次 host 同步。离线比对里还有一处没解释清楚：两个实现的 state 转置后仍对不上（相对差 1.0），但 attention 输出一致，暂未追。

**集成分支 `ascend950-mtp-integration`（2026-09-18 新建，无 PR，已推到 `origin`，head `da04db0846`）**：#808 与 #638 文件零重叠（#638 只动 `build.sh`、`python/`、`tests/`；#808 只动 `csrc/`、`include/`），所以 #808 基于 `upstream/main` 独立开 PR，两者合并无冲突。Qwen3.5 MTP 的 e2e 需要两者同时生效（Gemma RMSNorm provider + recurrent 算子），因此把 #638 的 `500ea7bb0e` 与 #808 的 `783b44c911` 用 `merge-tree` + `commit-tree` 合成一个 merge 提交作为构建分支，本地没有对应 worktree。#808 或 #638 有新提交后重新合成即可。单独验证 #808（构建 + 算子测试）不需要这个分支：`upstream/main` 的 `build.sh` 已支持 `-a kernels Ascend950PR_9599`，也没有 #638 引入的 `SGL_KERNEL_NPU_BUILD_TARGET` 要求。

**待做（独立 PR）**：`sparse_attention_score` 的 Ascend 950 同步（vllm-ascend #14412、#14918、#15371；上游另有 open 的 #722）。

**命名与历史重写（2026-09-17）**：按用户要求，#638 与 #32745 分支上由我们写的 “A5” 表述（提交信息、代码注释、帮助文字、README、测试名）改为正式名称 Ascend 950 / 950，用 `git filter-branch` 改写原提交后 `--force-with-lease` 推送；上游原有的 A5 措辞（如 `Detected A5:`、`(A2/A3/A5)`、deepep 帮助行）和 `wangyao-i` 写的内容未动。改写前后提交结构、作者与时间一致，最终树只差 A5 替换。旧 SHA 已不在 fork 上：主仓本地未推送的 gitlink 提交 `2decf29`、`303f2b0`、`582204c` 仍指向重写前的 `5b0ad9d`、`5749d79`、`84e144b`，`dfc8f8c` 指向随后删掉的 `7afbf61538`，最新的 `814cb31` 指向 `500ea7bb0e`。#32266 分支历史里仍含 #32745 的旧提交（如 `bf0904c27a`）。

## 分支上下文

2026-09-17 把 #32745 与 #638 合并到最新 `upstream/main`，两边都无冲突，merge 提交分别为 `ef127a4f36` 和 `d9768f1d3e`。#32745 另补 `a88e917b64`：上游 `test_kernels_namespace.py` 仍期望 NPU 上的 `_GEMMA_RMSNORM` 走 `torch_npu`，与本 PR 的 `sgl_kernel_npu` provider 冲突；此前 CI 卡在 `pr-gate`，`base-a-test-cpu` 从未运行，所以没有暴露。随后又补 `02bc394020` 修正 A5 安装说明；#638 另补 `d884b4c94c` 修正 A5 编译目标，并 cherry-pick #734 的 `8df27879fe` 与修复 `8cf80e44a6`（均见上文）。合并后只做了本地 CPU 验证：kernel 两个 build-target 测试套件 16 passed；SGLang layernorm 三个 CPU 测试文件 59 passed / 1 skipped；上游 lint 链（ruff-format 0.15.1、isort 7.0.0、codespell）通过。未做 NPU 硬件验证。

2026-08-25 把 #32266、#32601、#32602、#34387 重新基于 `upstream/main` 合并。合并后又叠了两笔：

- #32601 恢复 `--quantization mxfp_w4a8` 的 Linear 量化。分支尖端的 `6bcf2af6a2`（`:alembic:` 精度实验）把 `LinearBase` 全部路由到 `UnquantizedLinearMethod`，相对 main 是回退；现在回到 `7aa87ddfab` 的形态（Linear 保留 `NPUMXFP4W4A8LinearMethod` + K%32 对齐兜底），MoE 分支保留。要再做 expert-only 精度隔离，用 `ignored_layers` 排除非专家层，不要改 dispatch。
- #32602 把 gmm1 + swiglu + fp4 requant 重新融进 `npu_grouped_matmul_swiglu_quant_v2`（上游 #30319 落地的是非融合四算子版）。DeepEP 保持上游非融合路径，不像旧版那样 raise。

**待 A5 验证**：融合 gmm1 返回的 per-token scale 是否已配好对。`NPUW4A4MXFP4MoEMethod.apply()` 里那条 `pertoken_scale is not None` 分支在融合之前不可达，现在改成只在 scale 为 2 维时才 reshape（依据是 MXFP8 的 w2 gmm 对融合 scale 原样透传）。跑通 gmm2 即可确认。

主仓 `sgl-kernel-npu` gitlink 仍记录 PR #638 的旧 head `500ea7bb0e`（主仓 `814cb31`）；2026-10-08 子模块 checkout 已前进到 `72efb124cb`，因此主仓 `git status` 显示 `M sgl-kernel-npu`。本次未更新或暂存 gitlink，只在用户明确要记录新版本时才更新。该分支位于开发 fork 的 `codex/a5-gemma-rmsnorm-csrc`，不是 `main`。

## SGLang worktree 结构

```text
sglang/
├── qwen3.5_dense_w8a8/   # 主 clone；持有共享 .git；PR #32745
├── qwen3.5_dense_w8a8_pr36426/  # 派生 worktree；#32745 + 上游 #36426 的测试分支；无 PR
├── a5_fia_mixed_split/   # 派生 worktree；PR #34387（当前 CONFLICTING）
├── ascend_moe_lora/      # 派生 worktree；junlin-ascend-moe-lora @ bc9a4d0753；无 PR
├── qwen3.5_moe_w8a8/     # 派生 worktree；PR #32266 已关闭，分支与目录保留
├── qwen3.5_moe_w4a8/     # 派生 worktree；PR #32601
├── qwen3.5_moe_w4a4/     # 派生 worktree；PR #32602
├── npu_mamba_ssm_dtype/  # 派生 worktree；PR #40419（NPU 投机 bf16 SSM state）
└── upstream-main/        # 派生 worktree；本地分支 upstream-main @ 0dd66def7c；只读参考基线，不开 PR
```

`ascend_moe_lora`（`junlin-ascend-moe-lora`，merge-base `f2c84de022` / 2026-08-14，ahead 1）尚未开 PR。`upstream-main` 是跟随上游的只读基线 worktree，用于对照上游实现；它不是功能分支，落后 `upstream/main` 时直接快进即可，不要在里面改代码。

**`junlin_qwen3.5_dense_w8a8_pr36426`（2026-09-17 新建，2026-09-18 重建，无 PR，已推到 `origin`）**：给 `llm/qwen3.5_dense_offline_w8a8.sh`（`Qwen3.5-27B-mxw8a8`）提前合入上游 [#36426](https://github.com/sgl-project/sglang/pull/36426)（issue [#36423](https://github.com/sgl-project/sglang/issues/36423) 的修复，`zhujianwei-ops`，open，head `8786461159`），供 NPU 机器测试。

2026-09-20 起这个分支多了一笔自有提交 `15459663f2`（#40419 的 bf16 SSM state 修复，cherry-pick 自 `junlin_npu_mamba_ssm_dtype`），因为 Ascend 950 上跑 MTP 就用这个分支，而 #40419 还没合入上游；#40419 合入后把这笔删掉。在此之前它**不带任何自有改动**，精确等于 #32745 + #36426：`5aae20a88a` 是 #32745 的 `4f35f3ef7e` 与 #36426 head 的 `--no-ff` merge（无冲突），`git diff junlin_qwen3.5_dense_w8a8 HEAD` 与 #36426 自身的 diff 逐文件逐行一致（4 个文件、+54/-10）。重建之前它还带着 partial scale 与 graph rebind 两笔，现在这两笔已在 #32745 里，留在这边是重复。本地 CPU 回归 13 passed（`test_modelslim_mxfp8.py` 2 + `test_npu_gemma_rmsnorm.py` 11）。

历史验证结论仍然有效：合入 #36426 之前，MXFP8 checkpoint 的 GDN `in_proj_qkvz` 退回 `UnquantizedLinearMethod`，float8 权重不带 scale 直接进 bf16 参数，输出乱码；合入后用户在 Ascend 950 上确认告警清零、回答正常。#36426 合入上游后删除该分支与 worktree。

**#32745 的范围已扩大（2026-09-18）**：标题从 “Fix Qwen3.5 GemmaRMSNorm on Ascend 950” 改为 “Fix Qwen3.5 dense serving on Ascend 950”，因为分支上现在有三笔互相独立的修复：Gemma RMSNorm provider（配 kernel #638）、NPU graph rebind 竞态（`74d99befdb`）、MXFP8 partial scale（`4f35f3ef7e`，`cherry-pick -x` 自 #32266 的 `fc9cd5bad6`）。另有 `668222c7f0` 把 `test_npu_gemma_rmsnorm.py` 从 `test/registered/kernels/` 挪到 `test/registered/unit/npu/`：上游在 `4fb9b5b5ba` 之后给 `check_registered_tests.py` 加了 `_KERNEL_ROOT` 规则，`kernels/` 下的新增文件必须注册 `*-kernel-*` suite，而 workflow 里没有 CPU 的 kernel suite；本地分支的 checker 是旧版，只有 CI（PR 与最新 main 的 merge 结果）能发现。**#32266 落后 1206 commits，它的 `test_npu_gemma_rmsnorm.py` 也在 `kernels/` 下，rebase 时会踩同一个坑。**

注意 #32745 合入后 Qwen3.5 dense 的 **MXFP8 离线量化仍跑不通**，还缺上游 [#36426](https://github.com/sgl-project/sglang/pull/36426) 的 GDN 名字解析与视觉塔量化配置透传；#32745 提供的是 Gemma RMSNorm、graph replay 与 partial scale 三块。MXFP8 partial scale 的实测依据：`model.visual.blocks.0.mlp.linear_fc2.weight` 为 `[1152, 4304]`、`weight_scale` 为 `[1152, 135]`（`ceil(4304/32)`，奇数），同 block 的 `linear_fc1` 是 `K=1152` / 36 列，不受影响。

同一分支上又叠了 `5c4120b79c`（**与量化无关**；2026-09-18 按用户要求 `cherry-pick` 进 #32745，成为 `74d99befdb`，PR 正文同日加了 “Additional fix: NPU graph rebind ordering” 小节，不再单开 PR。代价是它跟着 #32745 一起等 kernel #638，随时可以从 #32745 上 reset 掉再单提）：`NPUCudaGraphBackend.replay_with_input_update` 原本在后台线程 rebind seq_lens、主线程同时 `graph.replay()`，而且 `thread.join()` 排在 replay 之后，没有任何机制保证 rebind 先于它要喂的那次执行落地。两者在 driver 里竞态，replay 走到 BindSqCq 时 rebind 还在飞行中，`rtModelExecute` 间歇失败。改动把 rebind 放回调用线程，先 rebind 再 replay，净 -3 行，不加环境变量开关（按用户要求）。

现象与定位：decode graph 第一次 replay 报 `bind sq cq failed, model_id=54, retCode=0x7020023` / `Insufficient_Resources(EL0006)`，措辞像显存不足，实际与内存无关；同一台机器同一脚本时好时坏，BF16 与 ModelSlim W8A8 MXFP8 都复现，换卡复现，卡上无残留进程。同一条 SGLang 堆栈在 [LinyuanLi0046/Ascend-SGLang-Skills#10](https://github.com/LinyuanLi0046/Ascend-SGLang-Skills/issues/10)（2026-08-04）被报过，那边 retCode 是 `0x7020004` / `EL0003`，plog 底层是 `trs_sqcq.c: Stream not inited or stream_mem not match`，Ascend 侧给的修法同样是先 rebind 再 replay。这段线程代码自 #23906（2026-06-09）起就在，上游至今未改。

验证：2026-09-18 用户在 Ascend 上做了 A/B 对照 —— 打补丁后连续多次跑 `llm/qwen3.5_dense_offline_w8a8.sh` 不再复现；把并发 rebind 放回去（当时还带 `SGLANG_NPU_GRAPH_UPDATE_IN_THREAD=1` 开关）立刻复现。因果链闭合。另有 CPU 时序验证（fake graph/device module：rebind 先于 replay 完成，且不再起线程）与上游 lint 链（ruff-format 0.15.1 format/check、isort 7.0.0、codespell）通过。开 PR 前需向用户确认机器的正式芯片名，PR 正文用正式名称（Ascend 950 / 910C / 910B，不写 A5/A3/A2）。

#36426 合入上游后删除该分支与 worktree。

kernel 仓功能分支的 worktree 放在主仓 `sgl-kernel-npu-worktrees/<name>/`，由子模块 `sgl-kernel-npu/` 派生。2026-10-09 清理后的目录如下：

```text
sgl-kernel-npu/                                   # 子模块；#638
sgl-kernel-npu-worktrees/
├── cache-loc-assign-bounds/                      # 派生 worktree；#824
└── main-638-824/                                # 编译测试；main + 两个 squash commit
```

`main` 无独立 worktree。所有 MTP 集成/实验分支、#808/#823 分支和独立 dtype-check 分支及其 worktree 已删除；本地临时分支 `pr651` / `pr742` 也已删除，未创建备份。

强制规则：**每个需要修改 SGLang 代码的分支，都必须在 `sglang/` 下有独立 worktree。** 不得在现有目录中切换功能分支，也不得直接在外部临时 worktree 修改。

共享仓库曾在系统临时目录注册过一个 `junlin_codeowners` worktree，其目录已消失，注册记录已于 2026-08-25 用 `git worktree prune` 清除。分支 `junlin_codeowners`（`78b216ea63`）本身保留；要复用它必须在 `sglang/` 下新建 worktree，不得再在仓库外目录修改。

## 分支依赖

当前合入栈、API / wheel 依赖、组合验证依赖和并行范围统一维护在[重构计划第 5.2 节](mxfp-refactoring-plan.md#merge-dependencies)，本节不再保留另一张易过期的依赖图。旧 #32601 / #34387 已分别改用 #43739 / #43740；具体分支未因 PR 换号而改变。

`junlin_qwen3.5_dense_w8a8_pr36426` 仍是组合验证用途，不能因测试分支包含 #36426 就把它混入正式 PR 栈。提交祖先关系在操作前用 `git merge-base --is-ancestor <parent-head> <child-head>` 核实；API / wheel 依赖不能用 Git 祖先关系代替验证。

## 历史合并 / 关闭记录

以下保留事件发生时的说明；分支 / worktree 是否仍存在以上方当前定位及实时注册表为准，当前计划的 PR 状态统一查[合入跟踪表](mxfp-refactoring-plan.md#pr-merge-status)。

- [#20922](https://github.com/sgl-project/sglang/pull/20922)（Wan2.2 Diffusion MXFP8）2026-05-07 合并。
- [#22338](https://github.com/sgl-project/sglang/pull/22338)（Wan2.2 Diffusion MXFP4）2026-05-19 合并。
- [#24918](https://github.com/sgl-project/sglang/pull/24918)、[#25904](https://github.com/sgl-project/sglang/pull/25904)（Diffusion 量化文档）2026-05 合并。
- [#22352](https://github.com/sgl-project/sglang/pull/22352)（Qwen3 Dense W8A8 MXFP8）2026-06-16 合并；[#28505](https://github.com/sgl-project/sglang/pull/28505)（委托给 kernel + `torch.ops.npu`）2026-06-17 合并。
- [#23650](https://github.com/sgl-project/sglang/pull/23650)（Qwen3 Dense W4A8）2026-07-06、[#23795](https://github.com/sgl-project/sglang/pull/23795)（Qwen3 Dense W4A4）2026-07-17 合并。
- [#30768](https://github.com/sgl-project/sglang/pull/30768)（Qwen3 MoE W8A8）与 [#32013](https://github.com/sgl-project/sglang/pull/32013)（ModelSlim packed MXFP4 loader）2026-07-29 合并；[#31917](https://github.com/sgl-project/sglang/pull/31917)（CODEOWNERS）同日合并。
- [#34829](https://github.com/sgl-project/sglang/pull/34829)（清理 NPU 量化注释）2026-08-20 合并；对应 worktree 已移除，分支 `junlin-remove-vllm-ascend-comments` 仍在本地与 fork 远程。
- [#36768](https://github.com/sgl-project/sglang/pull/36768)（量化注释改用 vendor-neutral 措辞）2026-08-29 合并，head `junlin-remove-vendor-refs-comments`；无对应 worktree。
- sgl-kernel-npu [#808](https://github.com/sgl-project/sgl-kernel-npu/pull/808)（recurrent_gated_delta_rule on Ascend 950）2026-09-29 由 `sglang-npu-bot` 合入 `main`，squash 成 `9e0891b24a`；7 个改动文件与分支 head `c524e45eff` 逐文件一致（`c524e45eff` 本身不是 `upstream/main` 祖先）。分支 `ascend950-recurrent-gated-delta-rule` 与 worktree `sgl-kernel-npu-worktrees/ascend950-recurrent-gated-delta-rule/` 保留未清理；后续改动请另开分支，不要在这条已合并的历史上继续加提交。
- [#32266](https://github.com/sgl-project/sglang/pull/32266)（Qwen3.5 MoE W8A8 MXFP8 / 视觉塔适配）2026-09-18 关闭：MXFP8 partial scale 已 `cherry-pick` 进 #32745（`4f35f3ef7e`），GDN packed mapping 与视觉塔 quant config 由上游 #36426 以不同实现覆盖，分支 `junlin_qwen3.5_moe_w8a8` 与 worktree `sglang/qwen3.5_moe_w8a8/` 保留。
- 已关闭未合并：[#32155](https://github.com/sgl-project/sglang/pull/32155)（被 #32266 取代）、[#32776](https://github.com/sgl-project/sglang/pull/32776)（typed device capabilities）、[#32908](https://github.com/sgl-project/sglang/pull/32908)（capability-based Gemma dispatch，被 #32745 + #638 的构建期方案取代）。
- 活跃 PR 正文或 diff 若仍显示已合并的前置 PR，表示功能分支尚未完成基于最新 `main` 的栈清理；不要再把它们记为 open dependency。
- [#30318](https://github.com/sgl-project/sglang/pull/30318)（2026-08-16）与 [#30319](https://github.com/sgl-project/sglang/pull/30319)（2026-08-18，均由 `LinyuanLi0046` 提交）已把**离线 ModelSlim `W4A8_MXFP` / `W4A4_MXFP4` MoE** 路径合入 main，对应 `ModelSlimW4A8MXFP4MoE` / `NPUW4A8MXFP4MoEMethod` 与 `ModelSlimW4A4MXFP4MoE` / `NPUW4A4MXFP4MoEMethod`。#32601 与 #32602 原本各自实现了一套同名不同类的离线 scheme，已于 2026-08-25 删除，只保留在线量化入口（`--quantization mxfp_w4a8` / `mxfp4` 的 experts 分支），并把在线分支加进上游的 kernel 类。新增 MoE 量化方案前先查 `moe_quant_schemes` 表，上游条目在前，重复注册会变成死代码。

## 操作前核对

```bash
# 真实 worktree 注册表
git -C sglang/qwen3.5_dense_w8a8 worktree list --porcelain

# 当前目录的分支、HEAD、远程
git -C sglang/<worktree> status --short --branch
git -C sglang/<worktree> remote -v
git -C sglang/<worktree> rev-parse HEAD

# kernel 仓
git -C sgl-kernel-npu status --short --branch
git -C sgl-kernel-npu remote -v
git -C sgl-kernel-npu worktree list --porcelain

# open PR 实时状态
gh pr list --repo sgl-project/sglang --author TallMessiWu --state open \
  --json number,title,state,isDraft,headRefOid,mergeable,mergeStateStatus,reviewDecision
gh pr list --repo sgl-project/sgl-kernel-npu --author TallMessiWu --state open \
  --json number,title,state,isDraft,headRefOid,mergeable,mergeStateStatus,reviewDecision
```

新增 SGLang 功能分支时，从主 clone 创建独立目录：

```bash
git -C sglang/qwen3.5_dense_w8a8 worktree add ../<worktree-name> -b <branch> upstream/main
```

如果分支已经存在，去掉 `-b` 并传现有分支名。创建前先确认该分支没有被其他 worktree 占用。
