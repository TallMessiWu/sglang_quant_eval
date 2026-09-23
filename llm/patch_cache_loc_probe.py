#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 torch.ops.npu.cache_loc_update 的每次调用打桩，打印它的缓冲区几何关系（可回滚）。

要确认的猜测：sgl-kernel-npu 的 cache_loc kernel 不看 out_cache_loc 有多大，host 把大小写死成
`batchSize * MAX_STEP`（16，见 csrc/cache_location_assign/op_host/tiling/cache_loc_assign.h），
device 端按这个数读进来、再整块写回去。而 SGLang 只给 `batch_size * draft_token_num` 个
（NEXTN 下是 4），于是每次 verify 准备都越界读写 `batch * 12` 个 int32。

这个越界**在数值上是隐形的**：尾部读出来什么就写回什么，所以比对内容永远发现不了，
唯一的外在症状就是踩到未映射显存时的 vector core exception。既然如此，打桩只能打在几何
关系上——每次调用打印张量实际多大、kernel 会碰多大、超出多少、超出部分会不会跨出所在的
2MB 段（torch 缓存分配器的小块 segment 就是 2MB，跨出去才有机会碰到未映射的页）。

第一次 verify 就会打印，**一条 curl 就够，不用跑测试集**。

模式由环境变量 SGLANG_CACHE_LOC_PROBE 选：

    observe（默认）  不改行为，照原样分配 batch*draft，只打印几何关系。
                     用它确认"每次调用都在越界"，以及有多大比例的调用会跨段。
    strict           同 observe，但一旦遇到"超出部分跨出 2MB 段"的调用，在调用算子**之前**
                     抛 Python 异常。得到的是干净的 Python 栈和完整几何信息，而不是一个
                     指不到地方的 device fault。
    guard            按 batch*16 分配、返回前缀视图（就是修复本身），并打印。
                     用来做 A/B：同样的请求，observe 会看到越界，guard 不会。
    force            照原样分配 batch*draft，但把它放在一块**独占 segment 的末尾**，
                     让越界部分直接越过整个 segment。目的是把"跑半个测试集才偶发"变成
                     "第一次 verify 就必现"。默认只对前 1 次调用生效
                     （SGLANG_CACHE_LOC_PROBE_FORCE_CALLS 可改）。
                     注意：不 fault 不代表没越界，只说明后面那页恰好是映射过的。

辅助环境变量：
    SGLANG_CACHE_LOC_PROBE_EVERY=200     每多少次调用打一行汇总（0 = 不打）
    SGLANG_CACHE_LOC_PROBE_SEGMENT=...   段大小，默认 2MB；分配器实现不同可改，测试也用它

用法（NPU 机器，SGLang 源码安装）：
    python3 llm/patch_cache_loc_probe.py --apply
    SGLANG_CACHE_LOC_PROBE=observe MTP=1 ./llm/qwen3.5_dense_bf16.sh 0
    # 另开一个终端发一条请求即可，日志里 grep cache-loc-probe
    python3 llm/patch_cache_loc_probe.py --restore

与 llm/patch_cache_loc_update_capacity.py 互斥（两者改同一处）：先 --restore 那个再用这个。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- cache_loc probe (patch_cache_loc_probe.py) ---"
CAPACITY_MARKER = "# --- cache_loc_update capacity (patch_cache_loc_update_capacity.py) ---"

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
        # --- cache_loc probe (patch_cache_loc_probe.py) ---
        return _cache_loc_probe_update(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            start_offset=start_offset,
            end_offset=end_offset,
            batch_size=batch_size,
            draft_token_num=draft_token_num,
            device=device,
        )
        # --- end probe ---
"""

HELPER = '''

# --- cache_loc probe (patch_cache_loc_probe.py) ---
# Diagnostic instrumentation, not a fix. Remove with patch_cache_loc_probe.py --restore.
#
# The kernel sizes its view of out_cache_loc as batch * MAX_STEP int32 regardless of
# what the tensor holds, loads that many and stores them back. The overrun is invisible
# in the values (it writes back what it read), so this records the geometry instead.
_CACHE_LOC_MAX_STEP = 16  # csrc/cache_location_assign/op_host/tiling/cache_loc_assign.h
_cache_loc_probe_state = {
    "calls": 0,
    "crossing": 0,
    "forced": 0,
    "last_key": None,
    "logger": None,
}


def _cache_loc_probe_cfg():
    import os

    return (
        os.environ.get("SGLANG_CACHE_LOC_PROBE", "observe").strip().lower(),
        int(os.environ.get("SGLANG_CACHE_LOC_PROBE_SEGMENT", 2 << 20)),
        int(os.environ.get("SGLANG_CACHE_LOC_PROBE_EVERY", 200)),
        int(os.environ.get("SGLANG_CACHE_LOC_PROBE_FORCE_CALLS", 1)),
    )


def _cache_loc_probe_logger():
    import logging

    if _cache_loc_probe_state["logger"] is None:
        _cache_loc_probe_state["logger"] = logging.getLogger("sglang.cache_loc_probe")
    return _cache_loc_probe_state["logger"]


def _cache_loc_probe_alloc(mode, have_n, touch_n, device, force_calls):
    # Returns (tensor handed to the op, note for the log line).
    if mode == "guard":
        backing = torch.empty(
            (max(have_n, touch_n),), dtype=torch.int32, device=device
        )
        return backing[:have_n], "guard: allocated batch*MAX_STEP, returning the prefix view"
    if mode == "force" and _cache_loc_probe_state["forced"] < force_calls:
        _cache_loc_probe_state["forced"] += 1
        # >= 10MB and a multiple of 2MB, so the caching allocator gives this
        # allocation a segment of its own; the view sits at its very end.
        seg_n = (12 << 20) // 4
        if seg_n < have_n:
            seg_n = have_n
        backing = torch.empty((seg_n,), dtype=torch.int32, device=device)
        return backing[seg_n - have_n:], "force: view placed at the tail of a dedicated segment"
    return torch.empty((have_n,), dtype=torch.int32, device=device), ""


def _cache_loc_probe_update(
    *,
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    batch_size,
    draft_token_num,
    device,
):
    mode, segment, every, force_calls = _cache_loc_probe_cfg()
    have_n = batch_size * draft_token_num
    touch_n = batch_size * _CACHE_LOC_MAX_STEP

    out_cache_loc, note = _cache_loc_probe_alloc(
        mode, have_n, touch_n, device, force_calls
    )

    st = _cache_loc_probe_state
    st["calls"] += 1
    log = _cache_loc_probe_logger()

    have_b = have_n * 4
    touch_b = touch_n * 4
    over_b = touch_b - have_b if touch_b > have_b else 0
    ptr = out_cache_loc.data_ptr()
    off = ptr % segment if segment > 0 else 0
    # The tensor itself never straddles a segment; the overrun leaving it is the
    # only way this reaches memory the allocator never handed out.
    crosses = over_b > 0 and (off + touch_b) > segment
    if crosses:
        st["crossing"] += 1

    key = (batch_size, draft_token_num, mode, crosses)
    if st["calls"] == 1 or key != st["last_key"]:
        st["last_key"] = key
        log.warning(
            "[cache-loc-probe] call#%d mode=%s bs=%d draft=%d | tensor %d int32 (%d B) at 0x%x "
            "| kernel touches %d int32 (%d B) -> %d B past the end "
            "| offset in %d B segment: %d, room to its end: %d B | overrun leaves the segment: %s%s",
            st["calls"], mode, batch_size, draft_token_num, have_n, have_b, ptr,
            touch_n, touch_b, over_b, segment, off, segment - off,
            "YES" if crosses else "no",
            (" | " + note) if note else "",
        )
    if every > 0 and st["calls"] % every == 0:
        log.warning(
            "[cache-loc-probe] summary: %d calls, %d of them (%.1f%%) had their overrun leave the segment",
            st["calls"], st["crossing"], 100.0 * st["crossing"] / st["calls"],
        )

    if mode == "strict" and crosses:
        raise RuntimeError(
            "[cache-loc-probe] refusing to launch cache_loc_update: it would touch "
            f"{touch_b} B starting at 0x{ptr:x}, but out_cache_loc holds {have_b} B and only "
            f"{segment - off} B remain in its {segment} B segment -- the last {over_b} B land "
            "outside any block the allocator handed out. This is the overrun that surfaces as "
            "'vector core exception' when those bytes are not mapped."
        )

    torch.ops.npu.cache_loc_update(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
    )

    return out_cache_loc
# --- end probe ---
'''


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
    backup = path.with_suffix(path.suffix + ".probe-orig")

    if args.show:
        text = path.read_text()
        print(path)
        print(f"  打桩已加入: {MARKER in text}")
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
    if CAPACITY_MARKER in text:
        print(
            "这个文件上已经打了 patch_cache_loc_update_capacity.py（两者改同一处）。\n"
            "先 python3 llm/patch_cache_loc_update_capacity.py --restore 再来。",
            file=sys.stderr,
        )
        return 1
    if text.count(OLD) != 1:
        print(
            f"锚点不唯一或不存在（{text.count(OLD)} 处）：这个版本的 cache_locs.py 结构不符，未修改",
            file=sys.stderr,
        )
        return 1
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(OLD, NEW, 1).rstrip("\n") + "\n" + HELPER)
    print(f"已加入打桩 {path}（重启服务生效）")
    print("模式用 SGLANG_CACHE_LOC_PROBE 选：observe / strict / guard / force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
