#!/usr/bin/env python3
"""补上 GDN prefill 的 gating 维度：sgl-kernel-npu #747 把它从 [1, T, H] 改成了 [T, H]。

sgl-kernel-npu #747（67be199，已合入）把 fused_gdn_gating_npu 的输出从
    g = torch.empty(1, batch, num_heads, ...)
改成
    g = torch.empty(batch, num_heads, ...)
配套是让它对上同一个 PR 新增的 AscendC 算子 chunk_gated_delta_rule（beta 是 (T, Nv) 二维）。

但 SGLang 侧至今（upstream/main efa7be2091）仍然 import 的是旧的 triton 路径
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
而它断言 len(beta.shape) == 3。于是新 wheel + 当前 SGLang 在 GDN prefill 上直接炸：
    AssertionError: beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise.

这个补丁在调用 kernel_dispatcher.extend 之前把维度补回去，只在拿到二维时才动手，
所以新旧 wheel 都能用。verify 路径不受影响：它走的是
fused_gdn_gating_kernel_without_sigmoid（torch.empty_like(a)，#747 没改），而且本来就有 unsqueeze(0)。

用法（NPU 机器，SGLang 源码安装）：
    python3 llm/patch_gdn_gating_shape.py --apply
    python3 llm/patch_gdn_gating_shape.py --restore
    python3 llm/patch_gdn_gating_shape.py --show
改完要重启服务。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- gating shape shim (patch_gdn_gating_shape.py) ---"

OLD = """            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
"""

NEW = """            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            # --- gating shape shim (patch_gdn_gating_shape.py) ---
            # sgl-kernel-npu #747 dropped the leading dim to match its new
            # AscendC chunk operator; the triton chunk kernel this path still
            # calls asserts [B, T, H]. Guarded so old wheels are untouched.
            if g.dim() == 2:
                g = g.unsqueeze(0)
                beta = beta.unsqueeze(0)
            # --- end shim ---
"""


def target_path() -> Path:
    from sglang.srt.hardware_backend.npu.attention import (  # noqa: PLC0415
        ascend_gdn_backend as mod,
    )

    return Path(mod.__file__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="补上维度")
    g.add_argument("--restore", action="store_true", help="从备份还原")
    g.add_argument("--show", action="store_true", help="打印目标文件与当前状态")
    p.add_argument("--file", help="直接指定 ascend_gdn_backend.py")
    args = p.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        try:
            path = target_path()
        except ImportError as exc:
            print(f"无法 import sglang：{exc}；可用 --file 指定路径", file=sys.stderr)
            return 2

    # 和其他几个调试补丁分开备份，避免互相覆盖
    backup = path.with_suffix(path.suffix + ".gating-orig")

    if args.show:
        text = path.read_text()
        print(path)
        print(f"  shim 已打上: {MARKER in text}")
        print(f"  备份存在: {backup.exists()}")
        return 0

    if args.restore:
        if not backup.exists():
            print(f"没有备份文件 {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"已还原 {path}")
        return 0

    text = path.read_text()
    if MARKER in text:
        print(f"{path} 已经打过了")
        return 0
    if text.count(OLD) != 1:
        print(
            f"{path} 里没有找到唯一的 fused_gdn_gating 调用点（找到 {text.count(OLD)} 处）",
            file=sys.stderr,
        )
        return 1
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"已备份到 {backup}")
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"已打补丁 {path}（重启服务生效，--restore 可还原）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
