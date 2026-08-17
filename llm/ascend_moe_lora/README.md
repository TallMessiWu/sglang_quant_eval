# Ascend NPU MoE-LoRA MVP 验证

这组脚本验证 `junlin-ascend-moe-lora` 分支的第一阶段范围：Qwen3-30B-A3B、BF16 非量化 expert、TP-only、无 shared expert，以及 eager/NPU Graph 下的 base-only、单 adapter、多 adapter 混合 batch。

## 0. 准备

服务器分别拉取两个仓库分支，并让脚本显式使用目标 SGLang checkout：

```bash
git -C /path/to/sglang fetch origin junlin-ascend-moe-lora
git -C /path/to/sglang switch junlin-ascend-moe-lora

git -C /path/to/sglang_quant_eval pull
export SGLANG_DIR=/path/to/sglang
```

准备两个真正的 per-expert adapter。MVP 不接受 `experts.<module>` 共享 outer factor，也不接受 shared expert；adapter key 应包含 `experts.<expert_id>.<module>`。先做格式预检：

```bash
python3 llm/ascend_moe_lora/validate.py preflight-adapter /weights/lora-a
python3 llm/ascend_moe_lora/validate.py preflight-adapter /weights/lora-b
```

## 1. 算子与小层测试

```bash
SGLANG_DIR=/path/to/sglang \
  ./llm/ascend_moe_lora/run_unit_tests.sh all
```

其中 routing 测试覆盖 dispatch 后的 token/adapter/expert 顺序恢复和首版 fail-fast；NPU 测试覆盖 BGMV rank 8/16/64、多 adapter、`-1` sentinel、显式 `(W + BA)` 小层 reference，以及全 base-only 时 tensor bitwise identical。数值 reference 使用 `atol=2e-2, rtol=2e-2`。

## 2. Eager 端到端

每个服务单独启动并保存日志。以下以 TP=2 为例；TP=4 时把卡号和 `TP_SIZE` 改为 4。Qwen3-30B-A3B TP=1 只有在 HBM 足够时运行，否则以单元小层覆盖 TP=1。

先启动纯 MoE baseline：

```bash
export MODEL_PATH=/weights/Qwen3-30B-A3B
export TP_SIZE=2
./llm/ascend_moe_lora/serve.sh baseline eager 0 1 \
  2>&1 | tee baseline-eager-tp2.log
```

另一个终端采集固定的 3-request batch，然后停止 baseline 服务：

```bash
python3 llm/ascend_moe_lora/validate.py capture \
  --case base --output results/baseline-eager-tp2.json
```

启动 LoRA 服务：

```bash
export ADAPTER_A_PATH=/weights/lora-a
export ADAPTER_B_PATH=/weights/lora-b
./llm/ascend_moe_lora/serve.sh lora eager 0 1 \
  2>&1 | tee lora-eager-tp2.log
```

采集并比较：

```bash
python3 llm/ascend_moe_lora/validate.py capture \
  --case mixed --output results/lora-eager-tp2.json
python3 llm/ascend_moe_lora/validate.py compare \
  --baseline results/baseline-eager-tp2.json \
  --candidate results/lora-eager-tp2.json
```

`PASS` 表示：所有 base-only 行的输出 token IDs 和序列化 chosen-token logprob 与纯 MoE baseline 完全一致；两个 adapter 都产生了可观察 delta；相同 mixed mapping 的多次 replay 一致。内部 tensor 的 bitwise 条件由第 1 步小层测试负责，HTTP 测试不能直接观察模型内部 tensor。

## 3. NPU Graph 端到端

用同样流程把 `eager` 改为 `graph`，并为 baseline 与 LoRA 分别采集新 artifact：

```bash
./llm/ascend_moe_lora/serve.sh baseline graph 0 1 \
  2>&1 | tee baseline-graph-tp2.log
# capture --case base -> results/baseline-graph-tp2.json

./llm/ascend_moe_lora/serve.sh lora graph 0 1 \
  2>&1 | tee lora-graph-tp2.log
# capture --case mixed -> results/lora-graph-tp2.json

python3 llm/ascend_moe_lora/validate.py compare \
  --baseline results/baseline-graph-tp2.json \
  --candidate results/lora-graph-tp2.json
```

脚本固定 decode graph bucket 为 batch size 3，并连续请求 base-only、single-A、mixed、base-only、reverse-mixed 和重复 mixed。检查日志中每个 TP rank 只在启动捕获阶段出现一次对应 `Capturing batches (bs=3`，请求阶段不应再次捕获、graph break 或回退 eager：

```bash
grep -nE 'Capturing batches \(bs=3|Capture cuda graph failed|graph break' \
  lora-graph-tp2.log
```

## 4. Adapter reference（可选但建议）

在显式 BF16 `(W + BA)` reference 服务或已知正确的实现上，用相同 adapter alias 和同一脚本生成 mixed artifact；然后增加：

```bash
python3 llm/ascend_moe_lora/validate.py compare \
  --baseline results/baseline-graph-tp2.json \
  --candidate results/lora-graph-tp2.json \
  --adapter-reference results/reference-mixed-tp2.json \
  --logprob-atol 0.1
```

该检查要求 adapter token IDs 一致，chosen-token logprob 最大绝对差不超过 `0.1`。

## 5. 负向验收

`run_unit_tests.sh routing` 已直接覆盖 EP、AlltoAll、量化 expert、shared expert、virtual expert 和 shared-outer 的 fail-fast。真实模型启动还应确认错误发生在首个 forward 前，且没有自动关闭 LoRA、切 dispatcher 或回退非图模式：

- EP/AlltoAll：在测试命令中设置 `--ep-size 2 --moe-a2a-backend deepep`，期望错误包含 `TP-only` 或 `--moe-a2a-backend none`。
- 量化 expert：换用 W8A8/MXFP checkpoint，期望错误包含 `unquantized expert weights only`。
- shared expert：换用带 shared expert 的模型，期望错误包含 `shared experts`。

不要把 Qwen3.5-MoE、DeepSeek V3.2、Kimi K3 或 GLM5 计入本 MVP 的正向验收；它们需要后续 shared-expert 支持。
