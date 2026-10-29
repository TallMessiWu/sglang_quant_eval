# Ascend MoE LoRA 验证

保留场景：`junlin-ascend-moe-lora` 的 Qwen3-30B-A3B、BF16 非量化专家、TP-only、无 shared expert。测试覆盖目标不等于全部真机已通过；分支版本见 [分支表](../../docs/branches.md)。

## 环境

用户手动选择并安装目标 SGLang checkout。**以下命令在 `llm/` 执行**；`serve.sh` 使用当前 Python 环境，不切换分支、不改 PYTHONPATH、不清理进程。`SGLANG_DIR` 只为单元测试定位源码文件，不决定服务的导入版本。

两个 adapter 必须为真正 per-expert 格式：key 包含 `experts.<expert_id>.<module>`；不接受 `experts.<module>` 共享 outer factor 或 shared expert。

```bash
python3 ascend_moe_lora/validate.py preflight-adapter /weights/lora-a
python3 ascend_moe_lora/validate.py preflight-adapter /weights/lora-b
SGLANG_DIR=/实际的目标checkout bash ascend_moe_lora/run_unit_tests.sh all
```

`routing` 是路由/拒绝条件回归；`npu` 需要空闲 NPU，检查 BGMV、rank 8/16/64、多 adapter、`-1` sentinel、显式 `(W + BA)` reference（`atol=rtol=2e-2`）与 base-only bitwise。不得与 serving 在同卡运行。

## 服务对照

分别启动并停止 baseline/LoRA，不同时占同卡。显式卡号数量决定 TP，`all` 为全部可访问卡，不传默认 TP1；显存不足时不自动降低模型或 TP。

```bash
MODEL_PATH=/weights/Qwen3-30B-A3B \
    bash ascend_moe_lora/serve.sh baseline eager 0 1
# 就绪后在另一终端：
python3 ascend_moe_lora/validate.py capture \
    --case base --output results/baseline-eager-tp2.json
```

停止 baseline，再启动 LoRA：

```bash
MODEL_PATH=/weights/Qwen3-30B-A3B \
ADAPTER_A_PATH=/weights/lora-a ADAPTER_B_PATH=/weights/lora-b \
    bash ascend_moe_lora/serve.sh lora eager 0 1
python3 ascend_moe_lora/validate.py capture \
    --case mixed --output results/lora-eager-tp2.json
python3 ascend_moe_lora/validate.py compare \
    --baseline results/baseline-eager-tp2.json \
    --candidate results/lora-eager-tp2.json
```

Graph 对照将两次启动的 `eager` 改为 `graph`，写入新的结果文件，不覆盖 eager 证据。默认 decode bucket=3；日志应只在启动阶段 capture，不应在请求时重复捕获或回退 eager。采集覆盖 base-only、single-A、mixed、reverse-mixed 和重复请求。

`compare` 检查 base-only token IDs/chosen-token logprob 与 baseline 一致、adapter 有可观察 delta、相同 mapping 重放一致；不能直接观察模型内部 tensor。可选 `--adapter-reference <reference.json> --logprob-atol 0.1` 比较显式 BF16 `(W+BA)` 或已知正确实现。

## 拒绝条件

routing 测试覆盖 EP、AlltoAll、量化 expert、shared/virtual expert、shared-outer 的 fail-fast。实际启动应在首个 forward 前明确拒绝，不自动关闭 LoRA 或换 dispatcher。Qwen3.5 MoE、DeepSeek V3.2、Kimi K3、GLM5 的 shared-expert 支持不计入这个 MVP。
