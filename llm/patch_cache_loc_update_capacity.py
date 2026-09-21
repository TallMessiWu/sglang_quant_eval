#!/usr/bin/env python3
"""给 torch.ops.npu.cache_loc_update 的输出张量补足它实际会读写的容量（可回滚）。

根因（MTP 并发下的 "vector core exception / 507035"，plog 里
`fault kernel_name=cache_loc_assign`）：

  sgl-kernel-npu 的 csrc/cache_location_assign 不看 out_cache_loc 的真实大小，
  host 直接写死 `cacheLocSize = batchSize * MAX_STEP`（MAX_STEP = 16，#411 从 5 改来）。
  device kernel 开头 DataCopy 读进这么多 int32，update 模式结尾再 DataCopyPad 原样写回。
  而 SGLang 只分配了 `batch_size * draft_token_num` 个（NEXTN 是 4），于是每次 verify
  准备都对 out_cache_loc 尾部之后的 `batch * 12` 个 int32 做一次越界读 + 越界写回。

  平时不出事：写回去的就是刚读出来的内容；而且分配器按 512 字节取整，`batch * 64` 字节
  在 batch <= 8 时还落在自己那一块里——所以单请求、低并发永远复现不了。batch >= 9 起
  越过块边界；当这块恰好贴着已映射显存的末尾，MTE 就访问到非法 GM 地址，整个进程崩。
  和 GDN、graph padding、Ascend 950 的算子适配都无关。

这个补丁只改 SGLang 一侧：按 kernel 假定的容量（batch * 16）分配，返回前
batch * draft_token_num 个元素的视图。kernel 把结果按前缀和紧凑地写在最前面，所以取值
不变。正路是修 kernel（按 out_cache_loc.numel() 定大小），那需要重建 wheel；这个补丁
用来先确认根因、先把评测跑起来，kernel 修好后删掉。

用法（NPU 机器，SGLang 源码安装）：
    python3 llm/patch_cache_loc_update_capacity.py --apply
    python3 llm/patch_cache_loc_update_capacity.py --restore
    python3 llm/patch_cache_loc_update_capacity.py --show
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- cache_loc_update capacity (patch_cache_loc_update_capacity.py) ---"

OLD = """    elif _is_npu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

        return out_cache_loc
"""

NEW = """    elif _is_npu:
        # --- cache_loc_update capacity (patch_cache_loc_update_capacity.py) ---
        # The op ignores the size of out_cache_loc: its host code fixes the view
        # at batch * MAX_STEP int32 (MAX_STEP = 16, csrc/cache_location_assign),
        # loads that many and stores them back. A batch * draft_token_num tensor
        # is overrun 4x at draft_token_num = 4; once the tail crosses into
        # unmapped device memory the MTE faults ("vector core exception").
        # Results are packed at the front, so the prefix view is the answer.
        _cache_loc_max_step = max(16, draft_token_num)
        _cache_loc_buf = torch.empty(
            (batch_size * _cache_loc_max_step,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            _cache_loc_buf,
        )

        return _cache_loc_buf[: batch_size * draft_token_num]
        # --- end capacity ---
"""


def target_path() -> Path:
    from sglang.kernels.ops.speculative import cache_locs as mod  # noqa: PLC0415

    return Path(mod.__file__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--restore", action="store_true")
    g.add_argument("--show", action="store_true")
    p.add_argument("--file", help="直接指定 cache_locs.py（默认从已安装的 sglang 定位）")
    args = p.parse_args()

    path = Path(args.file) if args.file else None
    if path is None:
        try:
            path = target_path()
        except ImportError as exc:
            print(f"无法 import sglang：{exc}", file=sys.stderr)
            return 2
    backup = path.with_suffix(path.suffix + ".cacheloc-orig")

    if args.show:
        print(path)
        print(f"  补丁已加入: {MARKER in path.read_text()}")
        print(f"  备份存在: {backup.exists()}")
        return 0
    if args.restore:
        if not backup.exists():
            print(f"没有备份 {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        backup.unlink()
        print(f"已还原 {path}")
        return 0

    text = path.read_text()
    if MARKER in text:
        print("已经加过了")
        return 0
    if text.count(OLD) != 1:
        print(
            f"锚点不唯一或不存在（{text.count(OLD)} 处）：这个版本的 cache_locs.py 结构不符，未修改",
            file=sys.stderr,
        )
        return 1
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"已修改 {path}（重启服务生效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
