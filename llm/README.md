# LLM 启动、评测与排查

一级目录只保留 26 个直接使用的入口：21 个服务启动脚本（`qwen*.sh`、`a5_fia_mixed_split_serve.sh`）、`curl.sh`、`npu-cleaner.sh`，以及 AIS Bench 的 `gpqa.sh`、`gsm8k.sh`、`mmmu.sh`。辅助脚本按用途放进子目录。

| 目录 | 用途 | 保留原因 |
| --- | --- | --- |
| `ais_bench/` | 公共评测入口、模型配置生成器 | 三个评测入口共用；旧版 AIS Bench 仍需要生成配置 |
| `benchmarks/` | FIA mixed split 的服务压测、算子 benchmark | 服务性能与算子性能需要分别验证 |
| `diagnostics/` | 环境检查、plog 定位、算子正确性与压力检查 | 跨仓契约和并发故障仍需要独立验证 |
| `patches/` | 可回滚的诊断插桩与 A/B 工具 | 排查工具；按需使用，启动脚本不会自动应用 |
| `patches/archive/` | 已替代、已排除或仅用于旧版本的补丁 | 退出日常使用，保留旧部署的还原入口与历史复现 |
| `ascend_moe_lora/` | MoE LoRA 验证流程 | 独立的启动与验证场景，见其 README |
| `test_images/` | 多模态测试图片及生成器 | `curl.sh` 使用的测试素材 |
| `llm_quant_descriptions/` | 量化描述样例 | 量化格式的参考数据 |

从仓库根目录运行：

```bash
./llm/qwen3.5_dense_bf16.sh 0
./llm/curl.sh
VLLM_PORT=6969 ./llm/gsm8k.sh --batch-size 16
VLLM_PORT=6969 ./llm/gpqa.sh
VLLM_PORT=6969 ./llm/mmmu.sh
```

AIS Bench 的公共逻辑在 `ais_bench/run_ais_bench.sh`。生成的配置仍写入 `llm/.ais_bench_configs/`；FIA benchmark 与 profiler 结果仍写入 `llm/fia_bench/`。这些本地产物已被 Git 忽略。

## 需要保留的诊断工具

| 脚本 | 验证内容 |
| --- | --- |
| `diagnostics/mtp_env_doctor.py` | 实际 SGLang checkout、wheel 算子注册、state dtype/layout、残留补丁 |
| `diagnostics/npu_fault_kernel.sh` | 从已有 Ascend plog 定位 fault kernel，无需重跑服务 |
| `diagnostics/recurrent_gated_delta_rule_check.py` | GDN verify 的数值、dtype 矩阵、wheel 精度与耗时 A/B |
| `diagnostics/causal_conv1d_verify_check.py` | Conv MTP verify 输出与 state 更新 |
| `diagnostics/gdn_prefill_layout_check.py` | 用 `dk != dv` 区分 prefill 的 K-major/V-major 布局 |
| `diagnostics/gdn_prefill_batch_stress.py` | 多序列、变长、chunk_state 和真实头数的并发压力 |
| `diagnostics/cache_loc_update_oob_check.py` | cache_loc 的 roomy/tight 容量对照与越界复现 |

```bash
python3 llm/diagnostics/mtp_env_doctor.py
./llm/diagnostics/npu_fault_kernel.sh <scheduler-PID>
python3 llm/diagnostics/recurrent_gated_delta_rule_check.py --dry-run
./llm/benchmarks/a5_fia_mixed_split_bench.sh compare
```

## 需要保留的诊断补丁

| 脚本 | 用途与边界 |
| --- | --- |
| `patches/patch_mtp_phase_sync.py` | 各 MTP 阶段插入同步，定位异步 fault；会影响性能 |
| `patches/patch_cache_loc_probe.py` | cache_loc 缓冲区几何观察与对照；`guard` 只用于定位 |
| `patches/patch_gdn_verify_bounds.py` | 算子执行前检查 verify state 索引容量 |
| `patches/patch_gdn_verify_probe.py` | 同一组真机输入下比较 GDN 算子与 torch 参考 |
| `patches/patch_gdn_verify_torch.py` | Conv/GDN 的 torch A/B；同时提供上一个 probe 导入的参考实现，不能单独删除 |

先用脚本的 `--help` 查看参数；支持 `--show` 的脚本可先检查目标文件。诊断结束后用对应脚本的 `--restore` 还原并重启服务。它们依赖目标源码版本，移动目录不会让旧补丁自动适配新代码。

## 可退出日常使用的历史补丁

7 个历史文件已移到 [patches/archive/](patches/archive/README.md)，具体取代关系、已排除的假设和旧 wheel 限制见该目录说明。归档文件不再作为正常启动的前置步骤；旧部署还有补丁或备份时，保留它们便于 `--restore`。

整理验证通过：一级目录白名单、移动后的权限与引用、Shell/Python 语法、CPU 参考逻辑；使用模拟的新旧 AIS Bench 验证了三个入口各自的 host/port 与 URL 两条路径，共 12 个调用用例，覆盖切换工作目录与附加参数透传。FIA benchmark 仍定位原来的结果目录。NPU 算子的正确性、模型 warmup、精度与性能仍需要真机验证。
