# LLM 启动、评测与排查

一级目录只保留 29 个直接使用的入口：23 个服务启动脚本（`qwen*.sh`、`a5_fia_mixed_split_serve.sh`）、`curl.sh`、`npu-cleaner.sh`、`bench_serving.sh`，以及 AIS Bench 的 `gpqa.sh`、`gsm8k.sh`、`mmmu.sh`。辅助脚本按用途放进子目录。

| 目录 | 用途 | 保留原因 |
| --- | --- | --- |
| `common/` | NPU 启动前清理与设备参数处理 | 所有带清理步骤的启动脚本共用，统一确认与取消行为 |
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
VLLM_PORT=6969 ./llm/bench_serving.sh
VLLM_PORT=6969 ./llm/gsm8k.sh --batch-size 16
VLLM_PORT=6969 ./llm/gpqa.sh
VLLM_PORT=6969 ./llm/mmmu.sh
```

## ShareGPT 服务性能测试

先启动模型服务，再运行 `./llm/bench_serving.sh`。菜单以模型名为主、实际路径放在淡色括号内；**↑/↓ 选择，Enter 开始，Esc/Q 退出**，Ctrl+C 也会恢复终端状态。选项直接读取同目录 `*_bf16.sh` 的固定 `--model-path`，不会执行这些启动脚本，也不会清理 NPU 或切换服务端模型。请选择与当前服务一致的 tokenizer；同模型 BF16/在线量化对照应选同一项。

默认保留原压测配置：`--backend sglang`、`127.0.0.1:${VLLM_PORT:-6969}`、ShareGPT 数据集 `/mnt/share/dataset/ShareGPT_V3_unfiltered_cleaned_split.json`、并发 32、500 条请求。

```bash
VLLM_PORT=6969 ./llm/bench_serving.sh
MAX_CONCURRENCY=16 NUM_PROMPTS=100 VLLM_PORT=7979 ./llm/bench_serving.sh
./llm/bench_serving.sh --help
```

`SERVER_HOST`、`DATASET_PATH` 可覆盖地址和数据集；`NO_COLOR=1` 关闭强调/淡化效果。结果默认写入**执行时当前目录**的 `artifacts/bench-serving/<时间戳>-<随机后缀>/`，包含 `bench.log`、`results.jsonl`，临时文件与 tokenizer 缓存也位于该目录下；`BENCH_ROOT` 可改结果根目录。压测进程失败时脚本保留非零退出码。脚本需要交互终端，不依赖额外 TUI 工具。

隔离菜单回归（真实 PTY、模拟 benchmark，不访问模型服务）：

```bash
python3 -m unittest discover -s llm/tests -p 'test_bench_serving.py' -v
```

## 卡号、TP 与启动前清理

`qwen*.sh`、`a5_fia_mixed_split_serve.sh` 和 `ascend_moe_lora/serve.sh` 共用设备解析规则：显式卡号的**数量**决定 TP；不传卡号保留该脚本写定的默认 TP，`TP_SIZE` 环境变量不再覆盖它。

以 8 卡环境为例：

| 卡号参数 | TP | `ASCEND_RT_VISIBLE_DEVICES` |
| --- | --- | --- |
| 不传 | 脚本默认值；默认 TP2 时仍用 TP2 | `0,1,2,3,4,5,6,7` |
| `3` | 1 | `3` |
| `3 5` | 2 | `3,5` |
| `all` | 8 | `0,1,2,3,4,5,6,7` |

不传卡号或传 `all` 时，子进程移除继承的 `ASCEND_RT_VISIBLE_DEVICES`，通过 `torch_npu.npu.device_count()` 查询当前环境可访问的 NPU，并显式设置升序完整列表；默认 TP1/TP2 分别使用前 1/2 张卡。查询失败或没有 NPU 时中止启动。`all` 不能与卡号混用，重复或无效卡号也会报错。模型自身的 TP 限制仍由 SGLang 校验，脚本不自动降低 TP。

`qwen*.sh` 和 `a5_fia_mixed_split_serve.sh` 共用清理规则：

- 指定卡号时清理所选卡；`all` 清理全部卡；不指定卡号时不清理。
- `npu-smi info` 只要查到所选卡上有任何 PID，就显示 `PID / START_TIME / COMMAND / 工作目录`，启动时间通过 `ps` 获取，工作目录通过 `pwdx` 获取，供你判断，随后 **Enter 确认清理，P 保留进程并继续，R 换到空闲卡，Esc 取消整个启动**；确认前不会终止任何进程。
- **R / r** 会重新查询当前可访问卡数和进程占用，保留原选卡中的空闲卡及其位置，按卡号升序替换被占用的卡，显示前后卡号并直接继续启动；**不杀任何已有进程、不减少卡数或 TP**。例如指定 `0 2`，若 0、1 被占用且 2、3 空闲，则改用 `3 2`，仍为 TP2。
- 空闲卡不足时显示“需要 X 张卡，当前仅 Y 张空闲”及空闲卡号，保持原卡号并留在菜单，不启动、不清理；可再次按 R 重查，或选择 Enter/P/Esc。查询失败也留在菜单。`all` 按 R 时仍需满足全部卡数，不会自动降为剩余空闲卡数。
- 换卡只在启动器的冲突菜单提供；单独执行 `npu-cleaner.sh` 仍仅支持清理/保留/取消。空闲的判据是查询时 `npu-smi info` 未报告 PID，不是显存容量、健康或互联拓扑保证；不预留卡，其他任务仍可能在查询后抢占。
- “工作目录”是进程**当前工作目录**，不一定是最初启动目录。权限不足或进程已退出时该列显示“不可获取”；未安装 `pwdx` 时显示“不可获取（未安装 pwdx）”，仍必须确认；`-f/--force` 下也会展示表格。
- 不判断进程是否“自己的”：同 UID、不同 UID、`sudo` 或无法通过 `ps` 查询归属，都不能跳过确认。只有所选卡上没有 PID 时才直接继续。
- `-f` / `--force` 是有 PID 时唯一跳过确认的方式，但不会提升系统权限；清理失败仍中止启动。
- 没有交互终端且所选卡上有 PID 时默认取消，只有显式 `-f` 才继续。
- `-f` 不计入卡号数量，也不会进入 `ASCEND_RT_VISIBLE_DEVICES`；`-f all` 使用全部卡并跳过清理确认。仅传 `-f` 而无卡号不会清理任何卡，TP 仍取脚本默认值。

```bash
./llm/npu-cleaner.sh 0 1                     # 单独清理；目标卡有 PID 就确认
./llm/qwen3.5_dense_bf16.sh 0 1              # 有冲突时按 R 换卡；Esc 取消启动
./llm/qwen3.5_dense_bf16.sh -f 0 1           # 显式跳过确认
./llm/qwen3.5_dense_bf16.sh all              # TP=全部 NPU 数量，有进程则确认
./llm/a5_fia_mixed_split_serve.sh on -f 0    # 模式参数在前
./llm/a5_fia_mixed_split_serve.sh off all    # off/on 均支持 all
MODEL_PATH=/weights/Qwen3-30B-A3B ./llm/ascend_moe_lora/serve.sh baseline graph all
```

`diffusion/*.sh` 也共用上述选卡与清理规则，但卡数传给 **`--num-gpus`**，不强制设置 LLM 的 `--tp`，具体并行策略由 SGLang 决定。不传卡号保留默认 `--num-gpus 1`、全部卡可见且跳过清理；`all` 使用全部可访问卡。请保留仓库中的 `llm/common/npu-launch.sh` 和 `llm/npu-cleaner.sh`，不要单独复制 Diffusion 脚本。

```bash
bash diffusion/ti2v_online_mxfp8.sh 3 5      # 两张卡，--num-gpus 2；有进程时确认
bash diffusion/t2v_bf16.sh all              # 自动检测卡数并使用全部卡
bash diffusion/i2v_offline_mxfp8.sh -f 0     # 显式跳过确认并清理卡 0
```

MoE LoRA 脚本仍保持不清理进程的行为。

隔离回归测试（模拟 NPU 查询、进程信息与 kill，不会终止真实进程）：

```bash
python3 -m unittest discover -s llm/tests -p 'test_npu_cleaner.py' -v
```

## BF16 与在线量化权重

所有 `qwen*_online_*.sh` 的默认模型路径均以同前缀的 `*_bf16.sh` 为基准；在线量化必须使用同一份原始 BF16/FP16 权重，不能指向离线预量化 checkpoint。沿用 `qwen3.5_*.sh` 的入口命名，实际模型以加载目录的 config 为准。

| 脚本前缀 | BF16 与在线量化共用的默认模型目录 |
| --- | --- |
| `qwen3.5_dense` | `/mnt/share/weights/Qwen3.5-27B` |
| `qwen3.5_moe` | `/mnt/share/weight/Qwen3.6-35B-A3B` |
| `qwen3_dense` | `/mnt/share/weight/Qwen3-32B-ori/Qwen3-32B` |
| `qwen3_moe` | `/mnt/share/weights/Qwen3-30B-A3B` |

`qwen3.5_dense_online_w4a8.sh` 保留 `MODEL_PATH` 环境变量覆盖，其余上述 BF16/在线脚本使用固定路径。做对照时须检查显式覆盖，避免覆盖后偏离 BF16 基准。隔离启动回归同时检查 LLM 和 Diffusion 在线入口与对应 BF16 入口的默认路径一致性。

### Dense W4A8 在线 / 离线

`qwen3.5_dense_offline_w4a8.sh` 仍默认使用 `/mnt/share/weight/Qwen3.6-27B_mxfp-w4a8-0821`，支持 `MODEL_PATH` 覆盖；本次不改离线权重。离线格式由 `quant_model_description.json` 判断。当前在线默认是 Qwen3.5、离线默认是 Qwen3.6，**不能直接将这两个默认配置作为同模型量化对照**；需显式选择同版本的原始与预量化 checkpoint。切换在线/离线模式时也要检查已有 `MODEL_PATH`。

```bash
# 离线：读取 quant_model_description.json 自动选择 ModelSlim，不传 --quantization。
./llm/qwen3.5_dense_offline_w4a8.sh 0

# 在线：不设置 MODEL_PATH 时默认使用上表的原始 27B 权重，也可显式覆盖。
MODEL_PATH=/实际的BF16或FP16权重目录 ./llm/qwen3.5_dense_online_w4a8.sh 0

# 两个卡号使用 TP2；可切换端口、上下文和 eager 模式。
VLLM_PORT=6970 CONTEXT_LENGTH=8192 EXTRA_ARGS="--disable-cuda-graph" \
    ./llm/qwen3.5_dense_offline_w4a8.sh 0 1
```

在线脚本显式使用 `--quantization mxfp_w4a8`，并拒绝带 `quant_model_description.json` 的 ModelSlim 目录；离线脚本要求该文件存在，避免把未量化权重误当成离线 W4A8。两者均启用 FIA，默认 TP=1、端口 6969、上下文 131072；`MODEL_PATH` 可覆盖。`EXTRA_ARGS` 与既有脚本一样按空白分词。

当前 Qwen3.6-27B 部署验证使用 `SGLANG_MAMBA_SSM_DTYPE=bfloat16`，这是该部署条件，不将它设为所有模型的通用默认。恢复默认 graph 配置需去掉 `EXTRA_ARGS` 中的禁用参数并重启服务；单独在终端 `unset` 不会改变运行中服务。在线视觉 fc2 的 K=4304 回退提示及混精加载边界见 [已知陷阱](../docs/known-pitfalls.md#w4a8-alignment-fallback)，功能与数值验收见 [G2.1 清单](../docs/mxfp-refactoring-plan.md#refactoring-status)。

传入卡号会先调用 `npu-cleaner.sh` 清理所选卡上的进程；不传卡号则不清理，保留默认 TP 并设置全部卡可见。做在线/离线对照时须使用同一模型版本的原始和预量化 checkpoint，并确认服务器实际运行的是目标 SGLang 代码；启动脚本本身不证明 NPU 数值、warmup 或图片请求通过。

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
| `diagnostics/modelslim_w4a8_check.py` | 离线 W4A8 Linear 与 FP4/UE8M0 软件参考对照；不覆盖整模型或完整 graph，也不替代两入口的重构等价对照 |

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
