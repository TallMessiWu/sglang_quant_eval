#!/usr/bin/env python3
"""临时补丁：把 NPU 投机路径的 SSM state 转置加回去，好让**旧 wheel**继续能用。

背景：state layout 是跨仓契约，两边必须成对过。
  - kernel #747（`67be199`，已合入）把 FLA triton kernel 从 `b_h[BK, BV]` 翻成
    `b_h[BV, BK]`，统一到 AscendC 算子按物理内存读的那套 `(HV, V, K)`。
  - SGLang #39589（`8ac39c66d8`，已合入）随之删掉 memory_pool.py 里的
    `temporal_state.transpose(-1, -2)` —— 那行原本是用来桥接两种约定的。
wheel 停在 #747 之前、而 SGLang 已过 #39589 时，prefill（triton）和 verify（算子）
对同一块内存差一个转置，MTP 一开口就乱码截断。

正规修法是重建 wheel（合到含 #747 的 upstream/main）。但重编一次要一小时，所以这个
补丁提供另一条路：把转置加回 SGLang 侧，恢复旧 wheel 期待的配对 —— 逻辑视图回到
`(HV, K, V)` 给旧 triton，物理排布仍是 `(HV, V, K)` 给算子，两边各自都对。

**装上含 #747 的新 wheel 之后必须 --restore**，否则又会反向错配。
只影响投机解码；非投机路径本来就不走这段。

用法（NPU 机器，SGLang 源码安装）：
    python3 llm/patch_npu_spec_state_transpose.py --apply
    python3 llm/patch_npu_spec_state_transpose.py --restore
    python3 llm/patch_npu_spec_state_transpose.py --show
改完要重启服务。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- old-wheel SSM layout shim (patch_npu_spec_state_transpose.py) ---"

OLD = """            if speculative_num_draft_tokens is not None:
                # Cache intermediate SSM states per draft token during target verify
"""

NEW = """            if speculative_num_draft_tokens is not None:
                # --- old-wheel SSM layout shim (patch_npu_spec_state_transpose.py) ---
                # Restores the pre-#39589 pairing for a wheel built before
                # sgl-kernel-npu #747: the logical view goes back to (HV, K, V)
                # for the old FLA kernels while the physical buffer stays
                # (HV, V, K) for recurrent_gated_delta_rule. Remove once the
                # wheel carries #747.
                if _is_npu:
                    temporal_state = temporal_state.transpose(-1, -2)
                    temporal_state_shape = (
                        *temporal_state_shape[:-2],
                        temporal_state_shape[-1],
                        temporal_state_shape[-2],
                    )
                # --- end shim ---
                # Cache intermediate SSM states per draft token during target verify
"""


def target_path() -> Path:
    from sglang.srt.mem_cache import memory_pool as mod  # noqa: PLC0415

    return Path(mod.__file__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="加回转置（配旧 wheel）")
    g.add_argument("--restore", action="store_true", help="还原（配含 #747 的新 wheel）")
    g.add_argument("--show", action="store_true", help="打印目标文件与当前状态")
    p.add_argument("--file", help="直接指定 memory_pool.py")
    args = p.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        try:
            path = target_path()
        except ImportError as exc:
            print(f"无法 import sglang：{exc}；可用 --file 指定路径", file=sys.stderr)
            return 2

    backup = path.with_suffix(path.suffix + ".layout-orig")

    if args.show:
        text = path.read_text()
        print(path)
        print(f"  转置 shim 已打上: {MARKER in text}")
        print(f"  备份存在: {backup.exists()}")
        return 0

    if args.restore:
        if not backup.exists():
            print(f"没有备份文件 {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"已还原 {path}（现在需要含 #747 的 wheel）")
        return 0

    text = path.read_text()
    if MARKER in text:
        print(f"{path} 已经打过了")
        return 0
    if text.count(OLD) != 1:
        print(
            f"{path} 里没有找到唯一的锚点（找到 {text.count(OLD)} 处）。"
            "这个补丁只适用于已经合入 SGLang #39589 的版本；"
            "如果你的 SGLang 还带着 temporal_state.transpose，就不需要它。",
            file=sys.stderr,
        )
        return 1
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"已备份到 {backup}")
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"已打补丁 {path}（重启服务生效）")
    print("装上含 #747 的新 wheel 后记得 --restore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
