# LLM 启动、评测与诊断

**命令默认在 `llm/` 运行**，避免主仓同名 `sglang/` 遮蔽安装包。用户手动选择并安装目标 SGLang checkout；脚本不 fetch、切分支或改 PYTHONPATH。不额外粘贴整套缓存环境变量。

## 普通使用

```bash
bash ./qwen3.5_moe_online_w4a8.sh 6 7
# 服务就绪后，在另一终端的 llm/ 中执行：
bash ./curl.sh
VLLM_PORT=6969 bash ./gsm8k.sh --batch-size 16
VLLM_PORT=6969 bash ./bench_serving.sh
```

`qwen*_bf16.sh`、`qwen*_online_*.sh`、`qwen*_offline_*.sh` 分别用于原始、在线和离线权重；默认端口 6969。`gpqa.sh`、`gsm8k.sh`、`mmmu.sh` 共用 `ais_bench/`。模型路径以脚本和实际 config 为准，文件名不证明模型版本。

### 设备与清理

- 显式卡号数量决定 TP：`0 1` 为 TP2，`all` 为当前可访问卡数；不传卡号保留脚本默认 TP、全部卡可见，且不清理。`TP_SIZE` 不覆盖这一规则。
- 所选卡存在 PID 时，显示进程、启动时间与当前工作目录：**Enter 清理，P 保留继续，R 换空闲卡，Esc 取消**。R 不杀进程、不减少 TP；空闲卡不足则留在菜单。
- `-f/--force` 显式跳过确认，不提升权限；非交互且有占用时默认取消。不要为方便给正常命令加 `-f`。
- 无法查询 NPU、无效/重复卡号会中止。空闲仅指查询时无 PID，不保证拓扑或稍后无人占用。
- `npu-cleaner.sh` 可单独清理（不提供 R 换卡）；LoRA 入口只选卡、不清理。

```bash
bash ./qwen3.5_dense_bf16.sh 0 1
bash ./npu-cleaner.sh 0 1
bash ./a5_fia_mixed_split_serve.sh on 0 1
```

### 权重与配置

| 前缀 | BF16 与在线量化共用默认路径 |
| --- | --- |
| `qwen3.5_dense` | `/mnt/share/weights/Qwen3.5-27B` |
| `qwen3.5_moe` | `/mnt/share/weight/Qwen3.6-35B-A3B` |
| `qwen3_dense` | `/mnt/share/weight/Qwen3-32B-ori/Qwen3-32B` |
| `qwen3_moe` | `/mnt/share/weights/Qwen3-30B-A3B` |

在线不能指向 ModelSlim 预量化目录。Dense W4A8 在线/离线脚本支持 `MODEL_PATH` 覆盖；离线默认 `/mnt/share/weight/Qwen3.6-27B_mxfp-w4a8-0821`，由描述自动选择 scheme。因此这两个 Dense 默认路径**不是同模型版本的量化对照**。

Dense W4A8 与 Qwen3.5 MoE BF16/在线 W4A8 入口支持 `EXTRA_ARGS`，按空白分词、不执行 shell 表达式；含空格的单个参数不适合放入此变量。

```bash
MODEL_PATH=/实际的BF16权重目录 bash ./qwen3.5_dense_online_w4a8.sh 0
VLLM_PORT=6970 CONTEXT_LENGTH=8192 EXTRA_ARGS="--disable-cuda-graph" \
    bash ./qwen3.5_dense_offline_w4a8.sh 0 1
```

graph、池容量和 state dtype 不是通用优化开关。恢复默认后需重启服务；不要把历史缩池值作为生产建议。当前部署的 BF16 SSM、FIA 与视觉 K=4304 回退边界见 [陷阱](../docs/known-pitfalls.md)。

## W4A4 加载峰值记录

独立入口 **`record_w4a4_load_memory.sh`**，原有启动脚本不变。旧/新 SGLang 都用此入口；`before` / `after` 只是输出标签，**不会切换、更新或修改 SGLang**。修改前恢复点为远端 tag `archive/pr32602-pre-rebase-20261030`，修改后选择 #32602 当前分支；版本见 [分支表](../docs/branches.md)。两轮必须固定模型、NPU、TP、dtype、软件栈和启动环境，并使用独立的新服务进程。

在 `llm/` 执行，每轮先手动选择并安装相应 checkout：

```bash
# 修改前版本；6 为示例空闲卡号。
bash ./record_w4a4_load_memory.sh before 6
# 等各加载 worker 的报告 complete=true 后正常停止本轮服务，切换/安装修改后版本，再运行：
bash ./record_w4a4_load_memory.sh after 6

# 离线汇总，两轮目录也可以单独传入；不会启动模型。
python3 diagnostics/npu_load_memory.py summarize artifacts/load-memory
```

输出位于 `artifacts/load-memory/<标签>-<时间>-<唯一后缀>/`：`server.log` 是服务日志，`load-memory-<pid>-<时间>.json` 是每个 worker/target/draft 的记录。可只为记录加载峰值而运行，不需要请求或压测。

- 在实际 worker 的 `ModelRunner.load_model` 入口同步、重置 allocator 峰值，返回后同步并读取。窗口包括模型构造、权重加载、post-load 及该方法的收尾，**不含后续 KV pool、graph capture、warmup**；不是整次启动峰值。
- `summary_gib.peak_allocated` 是主要比较指标；同时保留 `peak_reserved`、`end_allocated`、`end_reserved`、`peak_allocated_above_begin`，原始数据单位为 bytes。`reserved` 包含缓存，不能把它保持高位直接解释为源权重未释放。它们是 **PyTorch NPU allocator** 统计，不含全部驱动/HCCL/其他进程占用，不是整卡 HBM 总量。
- 报告记录实际加载源码路径、Git revision/dirty 状态、设备、可见卡映射、TP/PP rank、模型参数和 torch/torch_npu 版本。对照时核对 `source_revision`，不要把标签当版本凭据；`source_dirty=true` 需说明额外修改。
- 只把 `status=ok` 且 `complete=true` 的记录视为完整窗口。OOM/异常仍保留部分峰值和错误；进程被强杀可能仅留下 `loading` 记录；尚未进入加载函数则没有记录。若启用了延后权重加载，记录为 `deferred_weights`，不冒充完整加载。还需核对所有预期 rank 都有报告。
- TP 各卡分别比较，不能把不同时间的各卡峰值简单相加。前后峰值差反映整组代码变更，若要严格归因到 loader，需另做单因素对照；加载结束占用下降也不单独证明每个源张量都释放。

## ShareGPT benchmark

```bash
bash ./bench_serving.sh
MAX_CONCURRENCY=16 NUM_PROMPTS=100 VLLM_PORT=7979 bash ./bench_serving.sh
bash ./bench_serving.sh --help
```

交互菜单 ↑/↓ 选择、Enter 开始、Esc/Q 退出；选项只读取同目录 BF16 脚本模型路径，不会启动模型或清理卡。选择与服务匹配的 tokenizer。

默认 ShareGPT `/mnt/share/dataset/ShareGPT_V3_unfiltered_cleaned_split.json`、并发 32、500 请求；`SERVER_HOST`、`VLLM_PORT`、`DATASET_PATH` 可覆盖。结果写入当前目录 `artifacts/bench-serving/<唯一目录>/`，`BENCH_ROOT` 可覆盖；失败保留日志并返回非零。

- `FLUSH_CACHE=1` 每轮清 prefix cache，默认 0；不是清编译缓存。A/B 使用相同值和相同完整预热策略。
- `OUTPUT_DETAILS=1` 保存逐请求明细并自动生成短摘要。原始 JSONL 可能含生成文本/错误，不必回传；一般只分享摘要。
- 分位数不能跨轮直接合并；客户端 ITL 可能按 token 平摊，长间隔样本数不等于独立停顿次数。
- `TRACE_STALLS=1` 依赖历史实验版客户端，不是当前正式 SGLang 的保证接口；普通测试不要开启。

已有结果无需重测，可离线汇总：

```bash
python3 diagnostics/summarize_serving_tails.py artifacts/bench-serving --last 1
```

## 按需诊断

| 工具 | 保留用途 |
| --- | --- |
| `diagnostics/mtp_env_doctor.py` | 实际 checkout、wheel 注册、SSM dtype/layout 与旧补丁残留 |
| `diagnostics/npu_fault_kernel.sh` | 从已有 plog 查 fault kernel，不重跑服务 |
| `diagnostics/recurrent_gated_delta_rule_check.py` | recurrent GDN 状态/数值与配套 wheel 检查 |
| `diagnostics/causal_conv1d_verify_check.py` | MTP verify 输出与 state 更新 |
| `diagnostics/gdn_prefill_layout_check.py` | K≠V 检测状态轴契约 |
| `diagnostics/gdn_prefill_batch_stress.py` | 多序列/变长 prefill 边界 |
| `diagnostics/modelslim_w4a8_check.py` | 按需对照单层 packed FP4/UE8M0 软件参考，不是部署前置 |
| `diagnostics/npu_profile_serving.py` | 采集正在运行服务的 trace/配置，不启动或修改服务 |
| `diagnostics/summarize_npu_profile.py` | 离线汇总已有 profiler CSV/trace |

先看工具 `--help`；独立算子 probe 只能在空闲卡运行，不与 serving 抢卡。`npu_profile_serving.py` 的 `--dry-run` 不发请求，正式采集需同机共享文件系统、实际服务 SHA；client SHA 不能替代 server SHA。profile 运行的吞吐不是性能验收，跨 stream/rank 的累计时间也不是 wall time。

```bash
python3 diagnostics/mtp_env_doctor.py
bash diagnostics/npu_fault_kernel.sh <scheduler-PID>
python3 diagnostics/npu_profile_serving.py --help
python3 diagnostics/summarize_npu_profile.py artifacts/npu-profile/实际目录
```

## 测试与专题

本仓保留的全部 CPU/mock 回归（不启动模型、不调用真实 kill）：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

- FIA mixed split：`benchmarks/` 与 `a5_fia_mixed_split_serve.sh`；只在对应分支按需运行。
- MoE LoRA：[独立说明](ascend_moe_lora/README.md)。
- 图片素材：`test_images/`，由 `curl.sh` 使用；量化描述样例在 `llm_quant_descriptions/`。
- AIS Bench 配置写入 `.ais_bench_configs/`，FIA 结果写入 `fia_bench/`；均为本地产物。
- 已删除旧 text-patch 工具、G2 验收框架和暂停的 G6 微基准，历史从 Git 恢复。**不重新安排性能测试**；结论与恢复版本见 [G6](../docs/mxfp-refactoring-plan.md#g6-performance-archive)。
