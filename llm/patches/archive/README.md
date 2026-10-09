# 历史补丁归档

这些补丁已退出日常使用。保留文件用于还原旧部署、查阅当时的 A/B 实验，或复现旧版本问题；当前服务优先使用配套 SGLang 分支和 kernel wheel。

| 文件 | 归档依据 | 保留范围 |
| --- | --- | --- |
| `patch_gdn_gating_shape.py` | gating 补维已在本地 `npu_mamba_ssm_dtype` 和 `qwen3.5_dense_w8a8_pr36426` 的源码中实现 | 旧部署的 shim 还原 |
| `patch_gdn_prefill_ascendc.py` | 本地 NPU MTP 分支已有 AscendC prefill、L2 norm 和 chunk_state 适配 | 旧部署还原、旧版本复现 |
| `patch_npu_spec_state_transpose.py` | 只适配 kernel #747 之前的 wheel；新 wheel 配对时叠加会反向错配 | 旧 wheel 兼容与还原 |
| `patch_gdn_verify_state_major.py` | verify 转置假设已被排除；当时复读的根因是 SSM state dtype 位宽不匹配 | 历史实验、还原 |
| `patch_mamba_triton_fallback.py` | mamba state Triton kernel 的根因假设已被排除 | 历史 A/B、还原 |
| `patch_cache_loc_update_capacity.py` | 在 SGLang 硬编码 `batch * 16` 的 workaround 已被仓库明确撤回；正式修复在 kernel 侧 | 容量对照、旧部署还原；需要插桩时使用上级目录的 cache_loc probe |
| `a5_fia_mixed_split_debug_logging.patch` | 对当前 `a5_fia_mixed_split` checkout 执行 `git apply --check` 失败，原始 patch 已漂移 | 查阅插桩位置；复用前须对目标源码重新生成 |

归档依据来自本地源码对照和 [已知故障机制](../../../docs/known-pitfalls.md)、[分支记录](../../../docs/branches.md)。目录整理没有重新判定远程 PR 是否合入，也没有验证远端部署的 wheel 版本。

旧部署如仍有补丁，应先用原脚本的 `--show`（如支持）核对目标与备份，再执行 `--restore`。这些脚本按当前 Python 环境定位目标文件；整体备份还原可能覆盖后来叠加的改动，需先查看目标 diff。

当不再需要还原旧部署或复现这些版本时，这一整个归档目录可以清理；Git 历史保留原始内容。
