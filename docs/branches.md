# 活跃分支、PR 与 worktree

> 最近核对：2026-09-14（Asia/Shanghai）；#32745 与 #638 两行于 2026-09-17 合并上游后重新核对；sgl-kernel-npu #808 于 2026-09-17 新建后核对；SGLang #40419 与 kernel 分支 `gdn-state-dtype-check` 于 2026-09-20 新建后核对，#32745 的 head 同日回退到 `4f35f3ef7e`；同日稍后按用户要求把 bf16 断言并入 #808、把 #40419 叠到 `junlin_qwen3.5_dense_w8a8_pr36426`，并重做两个 kernel 构建分支的 merge；同日 #32745 与该测试分支合入 `upstream/main` `c1a1eb5f66`（CONFLICTING → MERGEABLE）。GitHub 状态来自实时 PR 查询，本地状态来自 `git worktree list --porcelain`、`git branch --show-current` 和 `git rev-parse HEAD`。这些信息会漂移，操作前必须重新查询。

## 远程地址

| 仓库 | `origin`（开发 fork） | `upstream`（官方仓） |
| --- | --- | --- |
| SGLang | <https://github.com/TallMessiWu/sglang.git> | <https://github.com/sgl-project/sglang.git> |
| sgl-kernel-npu | <https://github.com/TallMessiWu/sgl-kernel-npu.git> | <https://github.com/sgl-project/sgl-kernel-npu.git> |

主仓 `.gitmodules` 的 `sgl-kernel-npu` URL 指向开发 fork；子模块内保留 `upstream` 作为官方同步来源。

## 当前 open PR（author: `TallMessiWu`）

| 仓库 / PR | 状态 | head 分支 | GitHub / 本地 HEAD | merge-base（upstream/main） | 本地 worktree |
| --- | --- | --- | --- | --- | --- |
| SGLang [#32745](https://github.com/sgl-project/sglang/pull/32745) — Qwen3.5 dense serving on Ascend 950 | Open，MERGEABLE / BLOCKED（`REVIEW_REQUIRED`） | `junlin_qwen3.5_dense_w8a8` | `c34d054145` | `c1a1eb5f66`（2026-09-20） | `sglang/qwen3.5_dense_w8a8/`（主 clone） |
| SGLang [#40419](https://github.com/sgl-project/sglang/pull/40419) — 修 Qwen3.5 GDN 在 Ascend 上的 prefill 与投机解码 | Open，MERGEABLE | `junlin_npu_mamba_ssm_dtype` | `313bc46917`（3 提交：bf16 SSM state + gating 补维 + prefill 改调 AscendC chunk 算子；两笔都 cherry-pick 到 `junlin_qwen3.5_dense_w8a8_pr36426`） | `2d216a11f8`（2026-09-20） | `sglang/npu_mamba_ssm_dtype/` |
| SGLang [#32601](https://github.com/sgl-project/sglang/pull/32601) — Qwen3.5 MoE W4A8 MXFP | Open Draft，MERGEABLE | `junlin_qwen3.5_moe_w4a8` | `7d71e1cdf4` | `1fa32d50e1`（2026-08-25） | `sglang/qwen3.5_moe_w4a8/` |
| SGLang [#32602](https://github.com/sgl-project/sglang/pull/32602) — Qwen3.5 MoE W4A4 MXFP4 | Open Draft，MERGEABLE | `junlin_qwen3.5_moe_w4a4` | `65653414a3` | `1fa32d50e1`（2026-08-25） | `sglang/qwen3.5_moe_w4a4/` |
| SGLang [#34387](https://github.com/sgl-project/sglang/pull/34387) — A5 mixed chunked-prefill FIA split | Open Draft，**CONFLICTING / DIRTY** | `junlin_a5_fia_mixed_split` | `c61c0f16a2` | `1fa32d50e1`（2026-08-25） | `sglang/a5_fia_mixed_split/` |
| sgl-kernel-npu [#638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) — portable Gemma RMSNorm API | Open，MERGEABLE / BLOCKED（`REVIEW_REQUIRED`） | `codex/a5-gemma-rmsnorm-csrc` | `500ea7bb0e` | `67a38fe215`（2026-09-17） | `sgl-kernel-npu/` |
| sgl-kernel-npu [#808](https://github.com/sgl-project/sgl-kernel-npu/pull/808) — recurrent_gated_delta_rule on Ascend 950 | Open Draft，MERGEABLE / BLOCKED | `ascend950-recurrent-gated-delta-rule` | `33b689ce48`（950 适配 + bf16 state 断言 + 合上游） | `004bc36`（2026-09-20） | `sgl-kernel-npu-worktrees/ascend950-recurrent-gated-delta-rule/` |

核对时 6 个 GitHub head SHA 均与本地 checkout 完全一致，6 个 SGLang worktree 与 kernel checkout 均无未提交改动。#808 创建后 GitHub head 与其 worktree 的 `783b44c911` 一致，worktree 无未提交改动。

两处相对 2026-08-25 的变化：

- **#34387 已变成 `CONFLICTING`**，上游在 8-25 之后动过它碰的文件，需要重新基于 `upstream/main` 合并。其余四个 SGLang PR 仍是 `MERGEABLE`。
- **#638 的 `ping1jing2` review 已于 2026-08-06 变成 `DISMISSED`**，之后无新评论（最后更新 2026-08-20）。五个 SGLang PR 全部卡在 `REVIEW_REQUIRED`，没有 approve。

参考基线（2026-09-17 fetch 的 `upstream/main`）：SGLang `4fb9b5b5ba`、sgl-kernel-npu `67a38fe215`（同日二次 fetch，含 #802）。#32745 与 #638 已合入该基线（落后 0，GitHub head 与本地一致，仍是 MERGEABLE / BLOCKED）；#32601、#32602、#34387 仍以 `1fa32d50e1` 为 merge-base，各落后 1206 commits（#32266 同样落后，已于 2026-09-18 关闭）。

## 上游覆盖情况（open PR 是否已被 main 吃掉）

逐项对照过 `upstream/main`，**没有任何一个 open PR 的核心功能被完整覆盖**，可以继续推进。已经被上游吃掉的是 #32601/#32602 的离线部分。

| PR | 覆盖 | 说明 |
| --- | --- | --- |
| #32745 | 未覆盖 | `GemmaRMSNorm.forward_npu` 与 `kernels/ops/layernorm/__init__.py::GemmaRMSNormOp.forward_torch_npu` 仍直接调 `torch_npu.npu_gemma_rms_norm`；`KernelBackend` 里 `sgl_kernel_npu` 仍只是 `TODO(RFC #29630)` 注释。仅有的缓解是 `SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM`（2026-02 为 Skywork-Gemma2 加的 native 回退，有性能代价）和 `residual is not None` 时的 `add_gemma_rms_norm` triton 路径；`residual=None` 的裸 Gemma RMSNorm 在 A5 上依旧落到未注册算子 |
| #32601 | 部分 | 已在 upstream：`NPUW4A8MXFP4MoEMethod`（离线，#30318）、Linear 侧 `NPUMXFP4W4A8LinearMethod`。仍缺：`npu_mxfp4.py` 的 `FusedMoE` 分支（还是 warning + `UnquantizedFusedMoEMethod`）、`NPUW4A8MXFP4OnlineMoEMethod`、K%32 非对齐 Linear 回退 |
| #32602 | 部分 | 已在 upstream：`NPUW4A4MXFP4MoEMethod`（离线，#30319）、`NPUDualLevelMXFP4LinearMethod`、`npu/moe/matmul.py` 里的 `npu_grouped_matmul_swiglu_quant_v2` 封装。仍缺：`npu_mxfp4_w4a4.py` 的 `FusedMoE` 分支、`NPUW4A4MXFP4OnlineMoEMethod`、W4A4 的 `apply_fused_gmm1_swiglu`（上游 `apply` 走非融合四算子）、runner 的 `fuses_gmm1_swiglu` 泛化、`HiddenStatesDynamicQuant` 的 lazy op 绑定、K%32 回退 |
| #34387 | 未覆盖 | upstream `forward_mixed` 仍是单次 FIA 调用；`hardware_backend/npu/` 下没有 `device_op.py`；`environ.py` 里没有 `SGLANG_NPU_FIA_MIXED_SPLIT` |
| #638 | 未覆盖 | upstream kernel 仓没有 `norm/gemma_rmsnorm.py`，`setup.py` 没有构建期 provider staging。`build.sh` 虽已有 `Ascend950PR_9599` 的 kernels target，但不做 Gemma provider 选择。新增的 `norm/split_qkv_rmsnorm_rope.py::split_qkvgate_gemma_rmsnorm_rope` 是融合 QKV+RoPE 的另一条路径，不替代独立的 GemmaRMSNorm 调用；#733 的 `split_qkv_rmsnorm_mrope.py` 不含 Gemma 路径。上游 release workflow 已按 #803 为 950 单独出包（`./build.sh -a kernels Ascend950PR_9599`），合入 #638 后该包会 stage ACLNN provider |
| #808 | 未覆盖 | upstream `recurrent_gated_delta_rule` 的 host、kernel、schema 与声明仍在 `SGL_KERNEL_ENABLE_A3_ONLY_OPS` 内，kernel 没有 `arch35` 分支；上游 open 的 #651、#742 只改 arch22 的 state 往返语义，#747 不碰本算子 |

## 跨仓配对

**#638 与 #32745 是强配对，必须一起合。** #32745 的 `from sgl_kernel_npu.norm.gemma_rmsnorm import npu_gemma_rms_norm` 带 `except ImportError` 回退到 `torch_npu.npu_gemma_rms_norm`；#638 未合、wheel 里没有该模块时，#32745 单独合入在 A5 上等于没改，反之亦然。provider 由 `SGL_KERNEL_NPU_BUILD_TARGET` 在构建期选择：`target_providers/Ascend910/norm/gemma_rmsnorm.py` 或 `target_providers/Ascend950/norm/gemma_rmsnorm.py` 被 staging 成 `norm/gemma_rmsnorm.py`。

**#734 已并入 #638（2026-09-17）**：`wangyao-i` 的 [#734](https://github.com/sgl-project/sgl-kernel-npu/pull/734)（open，分支 `feat/pr638-continue`）基于 #638 的 `99421ab5b6`，只多一个提交 `1def0a4f14`，把 Gemma 专用的 `GEMMA_PROVIDERS` staging 泛化为 `build_tools/target_provider.py` + `target_providers/<target>/` 目录，并要求构建 wheel 时必须设置 `SGL_KERNEL_NPU_BUILD_TARGET`。该提交已 `cherry-pick -x` 到 #638 本地分支（`8df27879fe`，作者保留为 `wangyao-i`），唯一冲突在 `python/sgl_kernel_npu/README.md`，保留了 `Ascend950PR_9599` 的编译目标说明。并入后 #734 的改动全部包含在 #638 中。原提交漏了在 `find_namespace_packages` 中排除 `build_tools`，wheel 会多装一个顶层 `build_tools` 包；随后补 `8cf80e44a6` 修复：`setup.py` 同时排除 `build_tools`，原先断言 exclude 字符串的测试改为断言 wheel 顶层包只有 `sgl_kernel_npu`（该测试在修复前失败）。两个提交均已推送到 #638。

#32745 的 `installation.mdx` 说明已于 2026-09-17 改为（`02bc394020`）：A5 用户安装 `950` release 包，或用与 release 相同的 `bash build.sh -a kernels Ascend950PR_9599` 从源码构建。

**A5 编译目标（2026-09-17 已修，#638 `d884b4c94c`）**：合入的上游 #724/#750 让 `csrc/CMakeLists.txt` 按 `^ascend950` 选择 `arch35` 并启用 A5 专属算子（`kv_compress_epilog`、`swiglu_group_quant`），而 #638 原先把通用 `950` 和 `npu-smi` 检测到的 A5 映射到兼容目标 `Ascend910_9382`，构建出的 A5 包会用 `arch22` 编译 compressor、缺少 A5 专属算子，和官方 950 包不一致；上游 SGLang #37373 的 DSV4 在 A5 上会直接调用 `torch.ops.npu.kv_compress_epilog`。现在 kernels 目标下这两种情况都编译为 `Ascend950PR_9599`（与官方 950 release 包一致）；显式传入的 `Ascend950PR_*`/`Ascend950DT_*` 仍原样透传（测试用例为 `Ascend950PR_958b`）；不带 `-a` 的完整构建在 A5 上直接报错，因为 DeepEP 与 kernels 共用一次 CMake 配置，而 DeepEP 固定使用 A3 的 SoC 值。自动检测不区分 A5 具体型号，非 `9599` 型号需显式传入。Gemma provider 不受影响。本仓 `docs/sgl-kernel-npu-build.md` 与 `sgl-kernel-npu-dev` skill 已同步（主仓 `7152f9f`）。只做了 CPU 验证（build-target 测试 18 passed、SoC 解析场景模拟），未在 A5 上实际构建。

**`causal_conv1d` 在 Ascend 950 上的注册（2026-09-17，已由上游 #802 解决）**：改用原生目标 `Ascend950PR_9599` 后，A5 上跑 Qwen3.5 BF16 在 decode 图捕获时报 `torch.ops.npu.causal_conv1d` 不存在。原因是上游 #632 为了让 A5 先编过，把一批 host 算子统一划成 A3 专属，并没有逐个验证；`causal_conv1d` 的设备 kernel 一直在编，只关掉了 host 端和注册。对照 vllm-ascend `csrc/build_aclnn.sh` 的按 SoC 清单，`causal_conv1d`、`recurrent_gated_delta_rule`、`sparse_attention_score` 都是 A3/Ascend 950 通用；其余 A3 专属项（`lightning_indexer`、`sparse_flash_attention`、`sparse_attn_sharedkv`、`mla_preprocess`、`batch_matmul_transpose`、稀疏 KV offload）与 vllm-ascend 一致。#638 曾用 `7afbf61538` 单独放开 `causal_conv1d`；同日上游 [#802](https://github.com/sgl-project/sgl-kernel-npu/pull/802)（`zhaozx-cn`，support kimi k3 on A5）合入了同样的改动，#638 因此变成 CONFLICTING。按用户要求删掉 `7afbf61538`，再次合并 `upstream/main` `67a38fe215`（merge `500ea7bb0e`，无冲突，`--force-with-lease` 推送），PR 描述也删掉了 `causal_conv1d`。现在 #638 的 `csrc/`、`include/` 与上游完全一致，GitHub 恢复 MERGEABLE / BLOCKED；CPU 测试 26 passed。Ascend 950 上仍需用该分支重新构建（原生目标约 45 分钟）后跑 `tests/python/sgl_kernel_npu/test_conv1d_prefill.py` 和 `llm/qwen3.5_dense_bf16.sh`。

**`recurrent_gated_delta_rule` 的 Ascend 950 适配（2026-09-17 开 Draft [#808](https://github.com/sgl-project/sgl-kernel-npu/pull/808)）**：用于在 Ascend 950 上给 Qwen3.5 开 MTP（NEXTN target verify 调用该算子，SGLang #20918 引入）。分支 `ascend950-recurrent-gated-delta-rule` 基于 `upstream/main` `67a38fe215`，单提交 `783b44c911`，worktree 在 `sgl-kernel-npu-worktrees/ascend950-recurrent-gated-delta-rule/`（见下文 kernel worktree 结构）。做法与上游 #802 的 `chunk_kda_fwd`、`causal_conv1d` 相同：host、kernel、schema、impl 与头文件声明移出 `SGL_KERNEL_ENABLE_A3_ONLY_OPS`，schema 不变；kernel 保持单一源文件和原有 pipeline（stage buffer、MTE2/V/MTE3 事件、`mix_qkv` 拆分、intermediate state 初始化、speculative token 之间 FP32 state 延续），只在 `__CCE_AICORE__ == 310` 下把 arch22 专属的 repeat-stride `Mul`/`MulAddDst`、`Brcb`、mask 寄存器 `ReduceSum`、`Sum`/`Rsqrt` L2 norm 换成 `op_kernel/arch35/recurrent_gated_delta_rule_regbase.h` 的 MicroAPI 实现（`RowDotRegbase`、`RankOneUpdateRegbase`、`L2NormalizeRowsRegbase`，循环写法参考 vllm-ascend #9224、#9382，L2 norm 用 arch35 RMSNorm 同款序列）。arch35 不再切 host UB 预算外的 `qTempInUb`/`kTempInUb`/`qSumLocal`/`kSumLocal`；host 在 arch35 上用 `GetCoreNumAiv()` 作为 block 数，910 仍是 `GetCoreNum()`。按用户决定 `MAX_MTP` 保持 8（#11236 的 16 另开 PR），以 Draft 提出。只做了静态与 CPU 验证：clang-format 18.1.8、codespell、`git diff --check`；只展开新增条件后 910 视角的 kernel/host 源码与 `main` 逐字节一致；numpy 模拟 arch35 寄存器循环（64 lane、`UpdateMask` 尾块、BRC/首元素 load-store、masked store、多 block 分片、intermediate/recurrent 初始化、`num_accepted_tokens`）对照测试文件的 golden，8 组形状（含 dk=100/200、dv=72/136 和多 vStep 分块）float64 最大绝对误差 2.2e-16，只证明下标与循环边界。**未做**：950 上 `bash build.sh -a kernels Ascend950PR_9599` 构建、`hasattr(torch.ops.npu, "recurrent_gated_delta_rule")`、`python3 tests/python/sgl_kernel_npu/test_recurrent_gated_delta_rule.py`（脚本数值不一致也退出 0，须看 `failed: N`）、Qwen3.5 NEXTN e2e（依赖 #638 + #32745），以及 910B/910C 回归。#651/#742 的 BF16 往返改动落在两个 arch 共用的 `Compute` 代码里，任一方先合入后 rebase 即可对齐。

**Ascend 950 上跑 Qwen3.5 MTP 的分支组合（2026-09-20 核对，同日合上游后更新）**：

| 仓 | 目录 | 分支 | HEAD | 叠了什么 |
| --- | --- | --- | --- | --- |
| SGLang | `sglang/qwen3.5_dense_w8a8_pr36426/` | `junlin_qwen3.5_dense_w8a8_pr36426` | `1eca006f8d` | #32745 + 上游 #36426 + #40419（cherry-pick），base `c1a1eb5f66` |
| sgl-kernel-npu | `sgl-kernel-npu-worktrees/ascend950-mtp-experiment/` | `ascend950-mtp-experiment` | `1df42b260b` | base `004bc36`（已含 #802、#804、**#747**）+ #638 + #808 + #742（含 #651） |

wheel 必须从 `ascend950-mtp-experiment` 构建，不能退回 `ascend950-mtp-integration`：后者的 `move_intermediate_cache` 仍是 `h_block_size=2`，在 950 上会 `ub overflow, requires 2097152 bits while 2031616 bits available`；#651（含在 #742 里）把它改成 1。`llm/qwen3.5_dense_bf16.sh` 的 MTP 开关不再设 `SGLANG_MAMBA_SSM_DTYPE`，改由 #40419 在 `MambaPool` 里自动覆盖，启动日志出现 "not supported by the NPU speculative verify kernels" 即生效。

**Qwen3.5 MTP 在 Ascend 950 上复读的根因（2026-09-20 定位并修复）**：开 NEXTN 后输出流畅但几十步内塌成复读，接受率从 ~2.5 一路爬到满值 4.00；关掉 MTP 一切正常。根因是 SSM state 的 **dtype 位宽不匹配**，不是 layout。`torch.ops.npu.recurrent_gated_delta_rule` 只有 `RGDR<bfloat16_t, bfloat16_t>` 一份实例化，host 的 UB 预算按每元素 2 字节算（`coeff = (2 + 2) * aDk + 4`），且 `EXEC_KERNEL_CMD` 前没有任何按 dtype 的分发；而 `MambaPool` 的 `ssm_dtype` 默认是 `torch.float32`（`configs/mamba_utils.py`，可由 `mamba_ssm_dtype` 或 `SGLANG_MAMBA_SSM_DTYPE` 覆盖）。位宽对不上不会报错：kernel 用 2 字节步长走 4 字节 buffer，每隔一个元素落在 fp32 的高半字，而 fp32 的高 16 位正是它自己的 bf16 截断 —— 数值全程留在合理量程内，state 却是错位的，所以表现是文字通顺地逐渐失忆，而不是 NaN。只有 verify 路径会踩到：prefill/decode 走 triton，dtype 从张量上读。定位靠的是在真机 verify 里同时跑算子和 torch 参考的探针（`llm/patch_gdn_verify_probe.py`）拍到 state 是 `torch.float32`，再用 `llm/recurrent_gated_delta_rule_check.py --matrix` 只换 dtype 复现：bf16 最大误差 9.8e-04，fp32 返回 1e35 量级。两处修复：SGLang [#40419](https://github.com/sgl-project/sglang/pull/40419) 让 `MambaPool` 在 NPU 投机路径直接建 bf16；算子 host 补 `mix_qkv`/`beta`/两个 state 张量的 dtype 断言，让这种不匹配不可能再静默 —— 按用户要求并入 #808 的第二个提交 `6734df4e54`（#808 是 Draft，可直接拿来测），原独立分支 `gdn-state-dtype-check` 作废。和 #808 无关，910B/910C 同样会中。过程中被证伪的假设记录在案：state 转置（#693 思路）、conv 算子的 spec 分支、两个 mamba state triton kernel、state 提交的 off-by-one，全部排除。

**#32745 合入 `upstream/main` `c1a1eb5f66`（2026-09-20，CONFLICTING → MERGEABLE）**：落后 166 个提交，只有 `npu_cudagraph_backend.py` 冲突。上游 [#39589](https://github.com/sgl-project/sglang/pull/39589) 独立修了同一个竞态 —— `replay_with_input_update` 原本在后台线程里 rebind、主线程直接 `graph.replay()`，`thread.join()` 在 replay 之后，rebind 还在飞的时候 replay 已经到驱动的 BindSqCq，`rtModelExecute` 失败（retCode 0x7020023，报成 Insufficient_Resources(EL0006)）。我们的修法是放回调用线程同步做；上游改成复用一个 device-bound worker（`ThreadPoolExecutor(max_workers=1, initializer=self._device_module.set_device, initargs=(self._device_id,))`）并 `update_future.result()` 后再 replay，顺序保证相同，而且 worker 的 device 绑定在 initializer 里一次做好、不用每次 replay 建线程。上游版本更完整，所以整个文件取上游，`74d99befdb` 的改动作废，#32745 的有效 diff 从 11 个文件降到 10 个。其余 10 个文件自动合并无冲突。测试分支 `junlin_qwen3.5_dense_w8a8_pr36426` 再合 #32745 时零冲突（#36426 的 4 个文件与上游新提交不重叠），#40419 的 14 行原样带过。**未做**：本机没有装好 sglang 的 Python 环境，`test_modelslim_mxfp8.py` / `test_npu_gemma_rmsnorm.py` / `test_kernels_namespace.py` 这轮没跑，只做了 AST 解析与冲突标记检查；按仓库约定大规模 merge 后要在 NPU 机器上先重跑 baseline 再叠新改动。

**SSM state layout 的跨仓契约：SGLang #39589 必须配 sgl-kernel-npu #747（2026-09-20 踩到）**：`Mamba2StateShape.temporal` 是 `(num_heads, head_dim=V, state_size=K)`，也就是 `(HV, V, K)`。历史上 NPU 投机路径靠 `memory_pool.py` 里的 `temporal_state.transpose(-1, -2)` 把逻辑视图翻成 `(HV, K, V)` 去迁就 FLA triton kernel 的 `b_h[BK, BV]`，同时物理排布留在 `(HV, V, K)` 给 AscendC 算子按物理内存读 —— 两个消费者各看各的，正好都对。kernel 仓 [#747](https://github.com/sgl-project/sgl-kernel-npu/pull/747)（`67be199`，2026-09-17 合入）把 triton kernel 统一翻成 `b_h[BV, BK]`（源码注释 `# V row K col`、`# V dim stride = K`），SGLang [#39589](https://github.com/sgl-project/sglang/pull/39589)（`8ac39c66d8`，2026-09-18 合入）随之删掉那个转置。**两者必须成对**：wheel 停在 #747 之前而 SGLang 已过 #39589，prefill（triton）与 verify（算子）对同一块内存的解释就差一个转置，表现是 MTP 一开口就乱码截断（不是之前那种先通顺再复读）。只影响投机路径 —— 非投机时那个转置本来就不生效，triton 是唯一消费者，自洽所以两种 wheel 都能跑。判据：关掉 MTP 正常、开 MTP 立刻乱码。修法是把 kernel 分支合到含 #747 的 `upstream/main` 后重建 wheel；`67a38fe..004bc36` 只有 2 个提交，#747 动的三个注册文件（`csrc/CMakeLists.txt`、`csrc/pytorch_extensions.cpp`、`include/sgl_kenel_npu_ops.h`）与 #808 自动合并无冲突，合完 `fla/` 与上游逐字节一致、#651 的 `h_block_size=1` 保留。

**#747 的第二处断裂：gating 输出降维（2026-09-20）**：除了 SSM state layout，[#747](https://github.com/sgl-project/sgl-kernel-npu/pull/747) 还把 `fused_gdn_gating_npu` 的输出从 `torch.empty(1, batch, num_heads)` 改成 `torch.empty(batch, num_heads)`，为的是对上同一个 PR 新增的 AscendC 算子 `chunk_gated_delta_rule`（`beta` 是 `(T, Nv)` 二维）。但 SGLang 侧至今没跟上：`layers/attention/linear/kernels/gdn_triton.py` 仍然 `from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu`（上游 `efa7be2091` 核对过），而那条 triton 路径断言 `len(beta.shape) == 3`。于是**带 #747 的 wheel 配任何版本的 SGLang，GDN prefill 都直接 AssertionError**，和开不开 MTP 无关。修法放在 SGLang [#40419](https://github.com/sgl-project/sglang/pull/40419) 的第二个提交：调 `kernel_dispatcher.extend` 前按 `g.dim() == 2` 补回首维，旧 wheel 不受影响。verify 路径不用改，它走 `fused_gdn_gating_kernel_without_sigmoid`（`torch.empty_like(a)`，#747 没动）且本来就有 `unsqueeze(0)`。长期的正路是让 SGLang 改调 #747 新增的 `fla.chunk_gated_delta_rule_npu`（原生吃二维 beta，但签名用 `actual_seq_lengths` 而非 `cu_seqlens`、返回值少一个 `h`），那是上游该补的配对 PR。运行时补丁见 `llm/patch_gdn_gating_shape.py`。

**#747 的第三处断裂：prefill 的 state 布局（2026-09-20，Qwen3.5 MTP 在 950 上最终跑通）**：#747 把 pool、decode triton、verify 算子统一成 `(nv, dv, dk)`，**但没有动 prefill 走的 `sgl_kernel_npu.fla.chunk`**，它仍返回 `(nv, dk, dv)`。SGLang 的 `ssm_states[cache_indices] = last_recurrent_state` 因此把转置后的内容写进 pool —— `dk == dv == 128` 时形状相同，不报错。用 `dk=64, dv=32` 实测（`llm/gdn_prefill_layout_check.py`，模型自身的形状永远分辨不出来）：triton 返回 `(1,4,64,32)`、#747 新增的 AscendC 算子返回 `(1,4,32,64)`，两者 attention 输出一致（128 token 时差 3.6e-02，bf16 累积量级）。修法是让 prefill 改调 `torch.ops.npu.chunk_gated_delta_rule`（`llm/patch_gdn_prefill_ascendc.py`，已提交进测试分支 `62644c2253`）。**这一处加上前两处（bf16 state、gating 降维）之后，950 上的 Qwen3.5 MTP 才真正跑通。**已按上游质量补完并并入 #40419：算子的 chunk 网格是 `sum_b ceil(len_b/64)`，与 GDN 在 `_init_track_ssm_indices` 里建的那套逐字相同，写的又正是"每块的进入态"，所以 `chunk_state` 直接当 `h` 返回，page 跟踪照常工作，而且行布局是 pool 的 `(nv, dv, dk)`（triton 那份 `h` 和它整条路径一样是转置的）。入参改成一次 `reshape` 直接喂给 `l2norm_fwd`，比先 `.contiguous()` 少一次拷贝；剩下的开销是算 chunk 数时每层一次 host 同步。离线比对里还有一处没解释清楚：两个实现的 state 转置后仍对不上（相对差 1.0），但 attention 输出一致，暂未追。

**集成分支 `ascend950-mtp-integration`（2026-09-18 新建，无 PR，已推到 `origin`，head `da04db0846`）**：#808 与 #638 文件零重叠（#638 只动 `build.sh`、`python/`、`tests/`；#808 只动 `csrc/`、`include/`），所以 #808 基于 `upstream/main` 独立开 PR，两者合并无冲突。Qwen3.5 MTP 的 e2e 需要两者同时生效（Gemma RMSNorm provider + recurrent 算子），因此把 #638 的 `500ea7bb0e` 与 #808 的 `783b44c911` 用 `merge-tree` + `commit-tree` 合成一个 merge 提交作为构建分支，本地没有对应 worktree。#808 或 #638 有新提交后重新合成即可。单独验证 #808（构建 + 算子测试）不需要这个分支：`upstream/main` 的 `build.sh` 已支持 `-a kernels Ascend950PR_9599`，也没有 #638 引入的 `SGL_KERNEL_NPU_BUILD_TARGET` 要求。

**待做（独立 PR）**：`sparse_attention_score` 的 Ascend 950 同步（vllm-ascend #14412、#14918、#15371；上游另有 open 的 #722）。

**命名与历史重写（2026-09-17）**：按用户要求，#638 与 #32745 分支上由我们写的 “A5” 表述（提交信息、代码注释、帮助文字、README、测试名）改为正式名称 Ascend 950 / 950，用 `git filter-branch` 改写原提交后 `--force-with-lease` 推送；上游原有的 A5 措辞（如 `Detected A5:`、`(A2/A3/A5)`、deepep 帮助行）和 `wangyao-i` 写的内容未动。改写前后提交结构、作者与时间一致，最终树只差 A5 替换。旧 SHA 已不在 fork 上：主仓本地未推送的 gitlink 提交 `2decf29`、`303f2b0`、`582204c` 仍指向重写前的 `5b0ad9d`、`5749d79`、`84e144b`，`dfc8f8c` 指向随后删掉的 `7afbf61538`，最新的 `814cb31` 指向 `500ea7bb0e`。#32266 分支历史里仍含 #32745 的旧提交（如 `bf0904c27a`）。

## 分支上下文

2026-09-17 把 #32745 与 #638 合并到最新 `upstream/main`，两边都无冲突，merge 提交分别为 `ef127a4f36` 和 `d9768f1d3e`。#32745 另补 `a88e917b64`：上游 `test_kernels_namespace.py` 仍期望 NPU 上的 `_GEMMA_RMSNORM` 走 `torch_npu`，与本 PR 的 `sgl_kernel_npu` provider 冲突；此前 CI 卡在 `pr-gate`，`base-a-test-cpu` 从未运行，所以没有暴露。随后又补 `02bc394020` 修正 A5 安装说明；#638 另补 `d884b4c94c` 修正 A5 编译目标，并 cherry-pick #734 的 `8df27879fe` 与修复 `8cf80e44a6`（均见上文）。合并后只做了本地 CPU 验证：kernel 两个 build-target 测试套件 16 passed；SGLang layernorm 三个 CPU 测试文件 59 passed / 1 skipped；上游 lint 链（ruff-format 0.15.1、isort 7.0.0、codespell）通过。未做 NPU 硬件验证。

2026-08-25 把 #32266、#32601、#32602、#34387 重新基于 `upstream/main` 合并。合并后又叠了两笔：

- #32601 恢复 `--quantization mxfp_w4a8` 的 Linear 量化。分支尖端的 `6bcf2af6a2`（`:alembic:` 精度实验）把 `LinearBase` 全部路由到 `UnquantizedLinearMethod`，相对 main 是回退；现在回到 `7aa87ddfab` 的形态（Linear 保留 `NPUMXFP4W4A8LinearMethod` + K%32 对齐兜底），MoE 分支保留。要再做 expert-only 精度隔离，用 `ignored_layers` 排除非专家层，不要改 dispatch。
- #32602 把 gmm1 + swiglu + fp4 requant 重新融进 `npu_grouped_matmul_swiglu_quant_v2`（上游 #30319 落地的是非融合四算子版）。DeepEP 保持上游非融合路径，不像旧版那样 raise。

**待 A5 验证**：融合 gmm1 返回的 per-token scale 是否已配好对。`NPUW4A4MXFP4MoEMethod.apply()` 里那条 `pertoken_scale is not None` 分支在融合之前不可达，现在改成只在 scale 为 2 维时才 reshape（依据是 MXFP8 的 w2 gmm 对融合 scale 原样透传）。跑通 gmm2 即可确认。

主仓 `sgl-kernel-npu` gitlink 记录的是 PR #638 的 `500ea7bb0e`（主仓 `814cb31`），与子模块 checkout 一致，`git status` 不显示 `M sgl-kernel-npu`。该 commit 位于开发 fork 的 `codex/a5-gemma-rmsnorm-csrc` 分支而非 `main`；#638 后续如有新提交或被合并压缩，gitlink 会重新落后，只在用户明确要记录新版本时才更新。

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

kernel 仓功能分支的 worktree 统一放在主仓 `sgl-kernel-npu-worktrees/<name>/`（2026-09-17 起，主仓 `.gitignore` 忽略该目录），由子模块 `sgl-kernel-npu/` 派生；`sgl-kernel-npu/` 本身继续检出 #638 的 `codex/a5-gemma-rmsnorm-csrc`，不在里面切分支：

```text
sgl-kernel-npu/                                   # 子模块；#638
sgl-kernel-npu-worktrees/
├── ascend950-recurrent-gated-delta-rule/         # 派生 worktree；#808（Draft，含 bf16 state 断言）
├── ascend950-mtp-experiment/                     # 派生 worktree；**建 wheel 用这个**；无 PR
└── gdn-state-dtype-check/                        # 已作废：提交已并入 #808，分支与目录保留
```

分支 `ascend950-mtp-integration`（#638 + #808 的 merge，head `3c51550550`）只在 `origin` 和本地 refs 里，没有 worktree；2026-09-20 起改法是把新的 #808 head 直接 `git merge` 进来（临时 worktree 用完即删），不再 `merge-tree` 重合成，这样 `ascend950-mtp-experiment` 里 #742 的冲突解决不用重做。

强制规则：**每个需要修改 SGLang 代码的分支，都必须在 `sglang/` 下有独立 worktree。** 不得在现有目录中切换功能分支，也不得直接在外部临时 worktree 修改。

共享仓库曾在系统临时目录注册过一个 `junlin_codeowners` worktree，其目录已消失，注册记录已于 2026-08-25 用 `git worktree prune` 清除。分支 `junlin_codeowners`（`78b216ea63`）本身保留；要复用它必须在 `sglang/` 下新建 worktree，不得再在仓库外目录修改。

## 分支依赖

本地提交祖先关系与 PR 栈为：

```text
#32601 / junlin_qwen3.5_moe_w4a8
  └─ #32602 / junlin_qwen3.5_moe_w4a4   # 含 #32601 的 merge 提交，不含其后的 lazy-op 修复

#32745 / junlin_qwen3.5_dense_w8a8   # 独立修复，与 sgl-kernel-npu #638 配对
  ├─ #34387 / junlin_a5_fia_mixed_split    # merge 了 #32745，未 rebase 前 diff 含其提交
  └─ 无 PR / junlin_qwen3.5_dense_w8a8_pr36426  # merge 上游 #36426 + cherry-pick #32266 的 fc9cd5bad6，仅测试用
```

#32601 与 #32602 都以 `main` 为 base，可以独立 review。

## 已合并 / 已关闭

- [#20922](https://github.com/sgl-project/sglang/pull/20922)（Wan2.2 Diffusion MXFP8）2026-05-07 合并。
- [#22338](https://github.com/sgl-project/sglang/pull/22338)（Wan2.2 Diffusion MXFP4）2026-05-19 合并。
- [#24918](https://github.com/sgl-project/sglang/pull/24918)、[#25904](https://github.com/sgl-project/sglang/pull/25904)（Diffusion 量化文档）2026-05 合并。
- [#22352](https://github.com/sgl-project/sglang/pull/22352)（Qwen3 Dense W8A8 MXFP8）2026-06-16 合并；[#28505](https://github.com/sgl-project/sglang/pull/28505)（委托给 kernel + `torch.ops.npu`）2026-06-17 合并。
- [#23650](https://github.com/sgl-project/sglang/pull/23650)（Qwen3 Dense W4A8）2026-07-06、[#23795](https://github.com/sgl-project/sglang/pull/23795)（Qwen3 Dense W4A4）2026-07-17 合并。
- [#30768](https://github.com/sgl-project/sglang/pull/30768)（Qwen3 MoE W8A8）与 [#32013](https://github.com/sgl-project/sglang/pull/32013)（ModelSlim packed MXFP4 loader）2026-07-29 合并；[#31917](https://github.com/sgl-project/sglang/pull/31917)（CODEOWNERS）同日合并。
- [#34829](https://github.com/sgl-project/sglang/pull/34829)（清理 NPU 量化注释）2026-08-20 合并；对应 worktree 已移除，分支 `junlin-remove-vllm-ascend-comments` 仍在本地与 fork 远程。
- [#36768](https://github.com/sgl-project/sglang/pull/36768)（量化注释改用 vendor-neutral 措辞）2026-08-29 合并，head `junlin-remove-vendor-refs-comments`；无对应 worktree。
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
