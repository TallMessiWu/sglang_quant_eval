#!/usr/bin/env python3
"""给 MTP verify 的 state 索引加一个零成本越界检查（可回滚）。

背景：verify 时 ssm_state_indices = arange(batch_size * draft_token_num)，索引的是
intermediate_ssm，而它只有 (mamba_spec_state_size + 1) * draft_token_num 行，
mamba_spec_state_size 取自 max_running_requests——但那个值可能被显存预算下调
（model_runner 的 capped_max_running_requests）。一旦实际并发超过它，
recurrent_gated_delta_rule 就会越界写。它是 AIV-only 算子，所以表现是

    [Error]: The vector core execution is abnormal.
    reason=vector core exception

——一个和越界毫无关系的报错，而且要等到后面某次同步才爆出来。

这个检查纯在 host 侧做：arange 的上界和张量行数都是已知的 Python 数值，不需要
任何 device 同步，所以零开销。越界时直接抛 AssertionError，指出真实原因。

用法（NPU 机器，SGLang 源码安装）：
    python3 llm/patch_gdn_verify_bounds.py --apply
    python3 llm/patch_gdn_verify_bounds.py --restore
    python3 llm/patch_gdn_verify_bounds.py --show
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- verify state bounds check (patch_gdn_verify_bounds.py) ---"

OLD = """        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, num_value_heads, head_k_dim, head_v_dim
            )
"""

NEW = """        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, num_value_heads, head_k_dim, head_v_dim
            )
            # --- verify state bounds check (patch_gdn_verify_bounds.py) ---
            # ssm_state_indices is arange(batch_size * seq_len); the pool has
            # (mamba_spec_state_size + 1) * draft_token_num rows and that size
            # can be capped below max_running_requests by the memory budget.
            # Overflowing it makes the AIV operator write out of bounds, which
            # surfaces later as an unrelated "vector core exception".
            _needed = batch_size * seq_len
            _have = intermediate_state.shape[0]
            assert _needed <= _have, (
                f"GDN verify would write {_needed} intermediate states into a pool "
                f"of {_have} rows (batch_size={batch_size}, seq_len={seq_len}). "
                "Lower --max-running-requests or raise the speculative state pool."
            )
            # --- end check ---
"""


def target_path() -> Path:
    from sglang.srt.hardware_backend.npu.attention import (  # noqa: PLC0415
        ascend_gdn_backend as mod,
    )

    return Path(mod.__file__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--restore", action="store_true")
    g.add_argument("--show", action="store_true")
    p.add_argument("--file")
    args = p.parse_args()

    path = Path(args.file) if args.file else None
    if path is None:
        try:
            path = target_path()
        except ImportError as exc:
            print(f"无法 import sglang：{exc}", file=sys.stderr)
            return 2
    backup = path.with_suffix(path.suffix + ".bounds-orig")

    if args.show:
        print(path)
        print(f"  检查已加入: {MARKER in path.read_text()}")
        print(f"  备份存在: {backup.exists()}")
        return 0
    if args.restore:
        if not backup.exists():
            print(f"没有备份 {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"已还原 {path}")
        return 0

    text = path.read_text()
    if MARKER in text:
        print("已经加过了")
        return 0
    if text.count(OLD) != 1:
        print(f"锚点不唯一（{text.count(OLD)} 处）", file=sys.stderr)
        return 1
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"已加入越界检查 {path}（重启生效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
