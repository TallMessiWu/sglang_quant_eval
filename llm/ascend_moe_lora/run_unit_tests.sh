#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(readlink -f "$0")")
repo_root=$(dirname "$(dirname "$script_dir")")
sglang_dir=${SGLANG_DIR:-${repo_root}/sglang/ascend_moe_lora}
test_scope=${1:-all}

usage() {
    cat <<'EOF'
Usage:
  run_unit_tests.sh [routing|npu|all]

Environment:
  SGLANG_DIR  SGLang checkout containing the Ascend MoE-LoRA branch.
EOF
}

case "$test_scope" in
    routing|npu|all) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown test scope: $test_scope" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ ! -d "$sglang_dir/python/sglang" ]]; then
    echo "SGLang checkout not found: $sglang_dir" >&2
    echo "Set SGLANG_DIR to the junlin-ascend-moe-lora checkout." >&2
    exit 2
fi

export PYTHONPATH="${sglang_dir}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "$sglang_dir"

python3 - <<'PY'
import sys

import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_npu_available={hasattr(torch, 'npu') and torch.npu.is_available()}")
PY

if [[ "$test_scope" == routing || "$test_scope" == all ]]; then
    python3 -m pytest -q \
        test/registered/unit/lora/test_ascend_moe_lora_routing.py
fi

if [[ "$test_scope" == npu || "$test_scope" == all ]]; then
    python3 - <<'PY'
import torch

if not hasattr(torch, "npu") or not torch.npu.is_available():
    raise SystemExit("Ascend NPU is required for the BGMV tests")
PY
    python3 -m pytest -q -s \
        test/registered/unit/npu/test_npu_moe_lora_bgmv.py
fi
