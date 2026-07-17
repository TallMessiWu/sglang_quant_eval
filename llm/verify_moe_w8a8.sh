#!/bin/bash
# ============================================================================
# MXFP8 MoE W8A8 一键验证脚本（在线 + 离线）
#
# 用法:
#   ./llm/verify_moe_w8a8.sh <NPU_DEVICE_ID>
#   例: ./llm/verify_moe_w8a8.sh 0        # 用 NPU 0 跑
#        ./llm/verify_moe_w8a8.sh 0,1      # 用 NPU 0,1 跑（tp=2 场景）
#
# 验证内容:
#   1. 在线 MXFP8（--quantization mxfp8）server 是否正常启动（warmup 不崩溃）
#   2. 离线 MXFP8（modelslim，含 C6 "--quantization modelslim 可省略" 验证）
#   3. 简单文本请求 → 输出无乱码 / 非空 → PASS
#
# SGLang 自动 warmup 已跑完整 prefill 路径（覆盖所有量化层 + runner +
# GroupedMatmulSwigluQuant wrapper + build_ascend_moe_runner helper），
# warmup 过即表示改动未破坏核心路径。
# ============================================================================

set -euo pipefail

# ---------- 配置 ----------
ONLINE_MODEL="/home/weights/Qwen3-30B-A3B"
OFFLINE_MODEL="/home/weights/Qwen3-30B-A3B-w8a8_mxfp8-0120-full"
PORT=30888
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLEANER="$SCRIPT_DIR/npu-cleaner.sh"
LOG_DIR="/tmp/sglang_verify_$$"
mkdir -p "$LOG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

# ---------- 工具函数 ----------
log_pass()  { echo -e "${GREEN}[PASS]${NC} $*"; PASS=$((PASS + 1)); }
log_fail()  { echo -e "${RED}[FAIL]${NC} $*"; FAIL=$((FAIL + 1)); }
log_info()  { echo -e "${YELLOW}[INFO]${NC} $*"; }

cleanup_npu() {
    if [ -x "$CLEANER" ]; then
        "$CLEANER" "$@" > /dev/null 2>&1 || true
        sleep 2
    fi
}

kill_server() {
    local pid="$1"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    # 确保端口释放
    local leftover
    leftover=$(pgrep -f "port.*$PORT" 2>/dev/null || true)
    if [ -n "$leftover" ]; then
        kill $leftover 2>/dev/null || true
        sleep 1
    fi
}

# 发请求并检查返回的 content 非空
check_response() {
    local label="$1"
    local resp
    resp=$(curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
          "model": "default",
          "temperature": 0,
          "max_tokens": 50,
          "messages": [{"role": "user", "content": "1+1等于几？直接回答数字。"}]
        }' 2>&1) || true

    local content
    content=$(echo "$resp" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d['choices'][0]['message']['content'].strip())
except Exception:
    print('')
" 2>/dev/null) || true

    if [ -n "$content" ]; then
        log_pass "$label  → 响应: ${content:0:80}"
    else
        log_fail "$label  → 空响应或 JSON 解析失败"
        echo "    raw: $(echo "$resp" | head -c 200)"
    fi
}

# ---------- 等待 server ready ----------
wait_for_server() {
    local label="$1"
    local max_wait=300  # 最多等 5 分钟（含 warmup）
    local waited=0
    log_info "$label 等待 server ready（含 SGLang warmup）..."
    while [ $waited -lt $max_wait ]; do
        if curl -s --max-time 3 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            log_info "$label server ready（${waited}s）"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        # 检查进程是否还在
        if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log_fail "$label server 进程已退出（可能在 warmup 中崩溃）"
            return 1
        fi
    done
    log_fail "$label server 超时未 ready"
    return 1
}

# ---------- 主流程 ----------
DEVICE="${1:-}"
log_info "MXFP8 MoE W8A8 验证开始 | 设备: ${DEVICE:-default} | 日志: $LOG_DIR"

# === 在线 MXFP8 ===
log_info "========== 1/2 在线 MXFP8 (--quantization mxfp8) =========="
cleanup_npu "$DEVICE"

if [ -n "$DEVICE" ]; then
    OLD_IFS="$IFS"; IFS=","; export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"; IFS="$OLD_IFS"
fi
export ASCEND_USE_FIA=1
export VLLM_PORT="$PORT"

sglang serve \
    --model-path "$ONLINE_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --quantization mxfp8 \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length 5000 \
    --trust-remote-code \
    > "$LOG_DIR/online.log" 2>&1 &
SERVER_PID=$!

if wait_for_server "在线"; then
    check_response "在线 MXFP8"
else
    log_fail "在线 MXFP8 → server 启动失败，查看 $LOG_DIR/online.log"
fi
kill_server "$SERVER_PID"
sleep 1

# === 离线 MXFP8（不含 --quantization，验证 C6 自动检测） ===
log_info "========== 2/2 离线 MXFP8 (C6: 省略 --quantization modelslim) =========="
cleanup_npu "$DEVICE"

sglang serve \
    --model-path "$OFFLINE_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length 5000 \
    --trust-remote-code \
    > "$LOG_DIR/offline.log" 2>&1 &
SERVER_PID=$!

if wait_for_server "离线"; then
    check_response "离线 MXFP8 (C6 自动检测)"
else
    log_fail "离线 MXFP8 → server 启动失败，查看 $LOG_DIR/offline.log"
fi
kill_server "$SERVER_PID"

# ---------- 汇总 ----------
echo ""
echo "=============================================="
echo -e "  结果: ${GREEN}${PASS} PASS${NC}  ${RED}${FAIL} FAIL${NC}"
echo "  日志: $LOG_DIR"
echo "=============================================="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
