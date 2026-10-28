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

`SERVER_HOST`、`DATASET_PATH` 可覆盖地址和数据集；`NO_COLOR=1` 关闭强调/淡化效果。`FLUSH_CACHE=1` 在每次压测前清空服务 prefix cache，默认 `0` 保持原行为；用于 A/B 时两组取值应一致。结果默认写入**执行时当前目录**的 `artifacts/bench-serving/<时间戳>-<随机后缀>/`，包含 `bench.log`、`results.jsonl`，临时文件与 tokenizer 缓存也位于该目录下；`BENCH_ROOT` 可改结果根目录。压测进程失败时脚本保留非零退出码。脚本需要交互终端，不依赖额外 TUI 工具。

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
| `diagnostics/mxfp4_perf_probe.py` | 在线 W4A8 的 GDN 投影、MoE 路由/GMM 单 rank 微基准；区分激活量化与计算，比较融合前后 |
| `diagnostics/npu_profile_serving.py` | 复用原生 benchmark 采集正在运行的 BF16/W4A8 服务 trace，记录实际服务配置；不启动或修改服务 |
| `diagnostics/summarize_npu_profile.py` | 在服务器本地汇总 NPU Text 导出，只需复制短摘要；也可处理已经采集的目录 |

```bash
python3 llm/diagnostics/mtp_env_doctor.py
./llm/diagnostics/npu_fault_kernel.sh <scheduler-PID>
python3 llm/diagnostics/recurrent_gated_delta_rule_check.py --dry-run
./llm/benchmarks/a5_fia_mixed_split_bench.sh compare
```

## W4A8 性能瓶颈诊断

先做算子级 A/B，再用真实服务 profile 确认占比。两者都只是内部诊断，**不能把微基准或开启 profiler 的吞吐当作正式 benchmark**；脚本不会更新 PR、修改 SGLang、切分支、安装依赖或设置 PYTHONPATH。

### 1. 单 rank 算子探针

在与服务相同、已包含 GDN 共享输入改动的 Python/SGLang 环境运行。探针要求在线 W4A8 方法提供 `prepare_input`，并记录真实导入的源码来源。`--model-path` **只读取 config.json，不读取 checkpoint 权重**，从实际 Qwen GDN/MoE 配置推导形状；合成 BF16 输入/权重，FP4 权重使用安装环境里的生产方法准备，转换在计时之外。当前面向普通 SwiGLU、EP=1、attention TP 与 MoE TP 相同的配置；不模拟 DP/EP、LoRA 或自定义激活。

```bash
MODEL_PATH=/mnt/share/weight/Qwen3.6-35B-A3B  # 改为本次测试的实际模型
python3 llm/diagnostics/mxfp4_perf_probe.py --model-path "$MODEL_PATH" --tp 2 --dry-run

# 示例物理卡 6：必须空闲；不清理进程，不要与服务在同一卡上同时跑。
# --tp 2 仅生成 TP2 下一个 rank 的形状，实际只用一张卡，无 TP 通信。
ASCEND_RT_VISIBLE_DEVICES=6 python3 llm/diagnostics/mxfp4_perf_probe.py \
    --model-path "$MODEL_PATH" --tp 2 --device 0 --mode eager

# 固定输入/路由的 NPU graph 微基准，不等于完整 serving graph。
# 先缩小到 32/128 tokens；比较每张图内 1/16/64 次调用，检查 replay 开销是否被摊薄。
ASCEND_RT_VISIBLE_DEVICES=6 python3 llm/diagnostics/mxfp4_perf_probe.py \
    --model-path "$MODEL_PATH" --tp 2 --device 0 --mode graph \
    --tokens 32 128 --graph-unroll 1 16 64
```

默认 `--tokens 1 8 32 128 512`、`--warmup 10 --iterations 50 --repeats 3`；MoE 同时测均匀和热点路由，并保存每个 expert 的 token 数。**并发 32 不意味着每个 expert 的 M=32**，所以不能只测一个大矩阵。可用 `--sections gdn` 或 `--sections moe` 缩小范围；不会为了省显存擅自减少 expert 数。`--max-weight-gib 4` 限制单 rank BF16 专家权重大小，不是峰值显存上限；BF16/FP4 双份权重、转换临时张量、图池与 workspace 还需额外空间。

主要看同一 token 数/路由分布下的这些对照：

| 疑点 | 对照 case | 如何解释 |
| --- | --- | --- |
| 小矩阵 W4 是否更慢 | `qkvz_bf16` / `qkvz_w4_matmul_only`；`ba_bf16` / `ba_w4_matmul_only` | 去掉激活量化后的算子对比；再看不同 M 的变化 |
| 激活量化成本 | `*_w4_matmul_only` / `*_w4_total`；`activation_quant` | 比较完整调用与预量化调用，不把孤立耗时相加当模型总耗时 |
| GDN 输入复用是否有收益 | `pair_w4_separate` / `pair_w4_shared` / `pair_bf16` | 前两者都包含量化；`pair_w4_qkvz_bf16_ba` 是仅 BA 保留 BF16 的消融 |
| MoE 融合路由是否有收益 | `moe_rank_w4_unfused_route` / `moe_rank_w4_fused_route` / `moe_rank_bf16` | 都包含路由、GMM1/激活、GMM2、combine；不含 router logits 和跨卡通信 |
| GMM1/GMM2 是否拖慢 | `gmm1_bf16_swiglu` / `gmm1_w4_fused`；`gmm2_bf16` / `gmm2_w4` | 使用已路由/已量化的输入；GMM1 已含激活，不是纯矩阵乘 |

结果默认在当前目录 `artifacts/mxfp4-perf/<唯一运行目录>/results.jsonl`。保存 config hash、实际导入的量化源码与探针路径/hash/git SHA、torch/torch_npu/设备信息，以及影响同步/队列的环境变量白名单（只记录，不修改）。Event 的 stream 时间也可能包含 host 发射空隙，**不是纯 kernel 时间**。重复运行同一层是热缓存微基准，不能直接预测整模型收益。

Graph 计时版本 `timing_version=2` 默认逐个测试 `--graph-unroll 1 16 64`：在一张图内捕获 U 次调用，每轮 replay `--iterations` 次，`stream_ms` / `wall_ms` 及中位数按 **iterations × U** 归一化为每次调用；`replay_stream_ms` / `replay_wall_ms` 保留每次整图 replay 的原始均值。每行有 `calls_per_replay`、`replays_per_round`，不可把不同 U 的行混为重复样本。`host_submit_ms_per_replay` 是主机发射循环均值，可能包含阻塞等待，不是纯 CPU 或纯 kernel 成本。Eager 不使用图内重复，始终 U=1。

每组 graph 增加 `graph_control_add`（4 元素加法）显示短图的固定开销下限，**不将它从其他时间里直接相减**。若 U=16→64 时每次调用耗时仍明显下降，说明尚未稳定，不据此决定生产优化；图内重复也可能改变缓存、内存复用和调度。U 越大图池/捕获成本可能越高，OOM 保留错误，不自动缩小 U。旧版一张图只捕获一次操作；若各小操作时间都挤在同一平台，只能作为 replay 调用现象，不能据此判断算子持平。

`graph_replay_check` 在计时外修改输入并重放两次，检查简单加法图确实更新输出；它不代表全部 GDN/MoE 图数值验收。各组 `sanity` 明确标记 `evaluation_mode=eager_before_timing`，只是捕获前合成输出的有限值/接线检查，不是 replay 输出或模型精度验收。

可用 `--output-dir ./artifacts/my-probe` 指定未存在的目录；脚本拒绝覆盖，失败退出非零并保留部分 JSONL/error.log，graph 失败不会偷偷退到 eager。临时目录和可配置缓存也放在该运行目录下。

### 1a. 普通 TP 路由三路对照

`--sections moe-route` 只测同一组输入、路由和权重下的四条完整链路：BF16 参照，以及 W4A8 的 A/B/C；另保留 graph 的小加法 control。它不重跑 GDN 和所有孤立 GMM，不改已安装的 SGLang，也不改变服务默认策略。

| 路径 | case | 顺序 |
| --- | --- | --- |
| A | `moe_rank_w4_fused_route` | BF16 输入 → 路由融合 MXFP8 量化 → 专家计算 → combine |
| B | `moe_rank_w4_unfused_route` | BF16 输入 → 路由展开 → MXFP8 量化 → 专家计算 → combine |
| C | `moe_rank_w4_prequantized_route` | 原始 token 先做 MXFP8 量化 → FP8 与 scale 一起路由 → 专家计算 → combine |

三条 W4 链路都保留 FP4 权重、GMM1/SwiGLU/requant 融合和 GMM2 COUNT 契约，**C 的输入量化在每次计时调用内**。C 直接调用 `npu_moe_init_routing_v2(quant_mode=-1, scale=..., x_dtype=float8_e4m3fn)`；现有生产 BF16 routing wrapper 不传入并保留这组 scale，不能拿它直接替代。安装环境不支持该接口时保留报错并退出，不自动升级或回退。

```bash
MODEL_PATH=/mnt/share/weight/Qwen3.6-35B-A3B
# 物理卡 6 必须空闲；TP2 仅表示单 rank 形状，实际只占一张卡。
ASCEND_RT_VISIBLE_DEVICES=6 python3 llm/diagnostics/mxfp4_perf_probe.py \
    --model-path "$MODEL_PATH" --tp 2 --device 0 \
    --sections moe-route --tokens 32 128 --routing uniform skewed \
    --mode graph --graph-unroll 16 64
```

先运行这一条并回传打印出的 `results.jsonl`。若有错误，同时回传同目录 `error.log`。需要 eager 对照时才用相同参数改为 `--mode eager`；无需默认重跑全量探针。

计时外，C 会按 `row_idx_type=0` 的源→目标映射逐字节检查 FP8 payload、scale，并核对专家分组。`sanity` 还记录 C/B、C/A 完整输出误差，这些跨路径误差仅报告，不自动宣布数值等价。graph 模式另输出 `moe_route_graph_replay_check`：对每个指定 U、三条 W4 链路分别改变输入和 expert IDs 两次，对照各自 eager 输出（BF16 `rtol=0.016, atol=1e-5`），结束后恢复原 fixture 再计时。它覆盖合成链路的重放，不代表模型精度验收。

**本次探针将 MoE finalize 从旧默认 `drop_pad_mode=0` 对齐为生产普通 TP 使用的 `2`**，在 runtime 行记录 `finalize_drop_pad_mode`。加上 section/token 选择影响 RNG 推进，新旧批次的绝对耗时不能直接归因于 C；比较本次同 fixture 的 A/B/C，检查 U16→64 是否稳定。

### 1b. 同版本 W4A8 服务路由 A/B

目标仍为已有 `junlin_qwen3.5_moe_w4a8`，或其 `junlin_qwen3.5_moe_w4a8_pr36426` 组合分支；由用户在服务器对应 checkout 手动更新并确认当前安装指向它。两组必须使用同一个新 SHA。`SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT` 在权重加载时解析，默认 `1` 为当前融合路径 A，`0` 为先路由再量化的 B，需要重启服务。它覆盖普通 TP 的 prefill/decode，不按 batch 自动切换；FP4 权重、GMM1/SwiGLU/requant 融合、GMM2、GDN 共享量化和 DeepEP 模式配置保持原契约。它是对照开关，尚未声明 B 在服务端更快。

```bash
# A：先启动服务；显式设开关时，启动器会检查安装的 SGLang 是否支持它。
SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT=1 bash llm/qwen3.5_moe_online_w4a8.sh 6 7
# 服务就绪后在另一终端运行，菜单选 Qwen3.6-35B-A3B。
FLUSH_CACHE=1 BENCH_ROOT="$PWD/artifacts/w4a8-route-a" bash llm/bench_serving.sh

# 完成 A 并手动停止服务后，启动 B；同样等待服务就绪再压测。
SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT=0 bash llm/qwen3.5_moe_online_w4a8.sh 6 7
FLUSH_CACHE=1 BENCH_ROOT="$PWD/artifacts/w4a8-route-b" bash llm/bench_serving.sh
```

同一服务可重复运行压测命令三次；两组使用相同模型、TP、数据集、并发 32、500 请求，原生 benchmark 默认 seed=42。不启用 profiler；每次清空 prefix cache，结果写入独立目录。回传两组末尾的吞吐、TTFT、TPOT、ITL（含尾延迟）摘要及实际服务 SHA，不必传逐请求记录。此 A/B 先判断真实服务收益；模型精度仍需单独验证。

### 2. 双卡真实服务 profile

仍用现有启动脚本，用户手动选择、更新并重启目标分支。下面采集器必须在服务所在机器运行；两次服务模型、TP、数据集、并发一致，**依次启动 BF16 和 W4A8 服务，不同时占用同一组卡**。采集会清空 prefix cache；不要与其他压测或 profiler 同时运行。

```bash
# BF16 服务启动后；--server-revision 可选，填写实际运行服务的 SHA，不是客户端 SHA。
python3 llm/diagnostics/npu_profile_serving.py \
    --label bf16 --model-path "$MODEL_PATH" --server-url http://127.0.0.1:6969

# 换为 W4A8 服务并完成启动后。
python3 llm/diagnostics/npu_profile_serving.py \
    --label w4a8 --model-path "$MODEL_PATH" --server-url http://127.0.0.1:6969
```

默认沿用 ShareGPT 路径、并发 32、500 请求，启用 `--flush-cache`，每阶段采集 5 步；`--dataset-path`、`--concurrency`、`--num-prompts`、`--steps` 可调。使用 SGLang 原生 `--profile-by-stage`；prefill/decode 的实际覆盖和 batch 形状以 trace/服务日志为准，短请求或过小负载未必覆盖目标步数。NPU 后端将 API 的 `CPU GPU` 活动映射为 CPU/NPU，不需要修改启动器。不要将该运行输出的吞吐加入正式性能表。

默认产物在当前目录 `artifacts/npu-profile/<标签-唯一运行目录>/`：`metadata.json`（服务配置白名单，不保存 API key/原始启动命令）、`benchmark.log`、`traces.json`、`traces/`，以及自动生成的 `summary.txt` / `summary.json`。检查 benchmark 确认 profiler 启动且确有 trace 文件；失败非零退出并保留证据。实际 SHA 不能由客户端 checkout 推断，因此可补 `--server-revision`。

本轮对照沿用上述两条 profile 命令；服务依次使用 `llm/qwen3.5_moe_bf16.sh`、`llm/qwen3.5_moe_online_w4a8.sh`，由用户手动选择相同 TP 和实际运行版本。元数据补充保留服务返回的 `max_prefill_tokens`、`enable_mixed_chunk`、`disable_cuda_graph_padding`，便于核对调度与 graph 设置。

**只需复制两次运行末尾 `=== NPU PROFILE SUMMARY ... ===` 到 `=== END NPU PROFILE SUMMARY ===` 之间的文字**，或各自的 `summary.txt`；每份不超过 6500 字符。原始 traces、日志、metadata 留在服务器，不需要传出。已采集过的目录无需重跑服务，直接执行：

```bash
python3 llm/diagnostics/summarize_npu_profile.py ./artifacts/npu-profile/实际运行目录
```

摘要使用 `kernel_details.csv` 按采集窗口、设备分别汇总算子类型、次数、累计耗时和前 6 项；不会再从 trace 重复累计 kernel。`name_hints_ms` 仅按算子名分类，融合算子的耗时不拆分。`CANN_ms` 来自 `step_trace_time.csv` 的 Computing、Communication、Communication(Not Overlapped)、Overlapped 等字段（原始单位 µs，摘要转为 ms），这些列不能相加；缺少时明确显示通信覆盖未知。字段依据 [torch_npu CSV 定义](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/analysis/prof_common_func/_csv_headers.py)和 [step 时间导出实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/analysis/prof_view/_trace_step_time_parser.py)。

NPU 原生目录名未必包含 TP rank 或阶段，脚本流式读取 `trace_view.json`（也支持 `.gz`），只根据显式 `step[EXTEND/DECODE ...]` 标记报告阶段及 batch/token 范围；没有标记就显示 `unknown`。设备编号不当作 TP rank，也不按目录排序猜阶段。`summary.json` 保留完整聚合结果和本地来源映射；若文本标记 `omitted>0`，可追加 `--capture C01` 等参数只打印某一窗口。没有有效 kernel CSV 或部分 kernel CSV 解析失败时退出非零，保留短诊断，不需要先传原始文件。

摘要 v2 将 `gating_delta_rule` 正确归入 GDN，并把 `HcclLaunchAicpuKernel` 单列为 `comm_launch`；其耗时不等于 collective 完成耗时。存在 HCCL 事件但 CANN Communication 为零时明确提示口径不足，不能推断通信免费。可直接读取旧导出的 CSV，查看至多三个指定算子的 input shape/dtype、调用次数、平均/最小/最大耗时，结果打印到终端并保存在 `summary-detail.txt`：

```bash
python3 llm/diagnostics/summarize_npu_profile.py ./artifacts/npu-profile/实际运行目录 \
    --capture C01 --operators IndexFill MoeInitRoutingV3
```

每个算子最多打印耗时最高的六组 shape，超出部分保留在本地 JSON；shape 只来自 CSV，不根据模型配置补猜。仅需要某个算子时在 `--operators` 后只传它的名字。此命令不访问服务、不重新运行 profiler。

原生 serving benchmark 未请求 `record_shapes`，当前服务默认关闭它；trace 不保证包含输入 shape 或明确的 graph 命中标记，不从缺失字段推断 batch 或未命中原因。摘要只是 profile 窗口的算子累计耗时，重叠 stream/rank 不可相加当作 wall time；短窗口也不保证捕获 P95/P99 对应的慢步骤。只凭微基准不决定修改生产路径。

CPU/模拟边界回归（不执行 NPU）：

```bash
python3 -m unittest discover -s llm/tests -p 'test_mxfp4_perf_probe.py' -v
python3 -m unittest discover -s llm/tests -p 'test_npu_profile_serving.py' -v
python3 -m unittest discover -s llm/tests -p 'test_summarize_npu_profile.py' -v
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
