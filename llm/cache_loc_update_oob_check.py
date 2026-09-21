#!/usr/bin/env python3
"""离线复现：torch.ops.npu.cache_loc_update 对 out_cache_loc 的越界读写（不加载模型）。

背景见 llm/patch_cache_loc_update_capacity.py：kernel 把 out_cache_loc 当成
`batch * MAX_STEP(16)` 个 int32 来读、再整块写回，而 SGLang 只分配
`batch * draft_token_num` 个。越界部分"读什么写回什么"，所以从数值上看不出来，只有当
张量贴着已映射显存的末尾时才会 fault——e2e 里就是偶发的 vector core exception。

这里把"贴着末尾"这件事变成必然：连续分配成千上万个同样大小的小张量并全部持有，
它们会按地址顺序铺满若干个 2MB 的小块 segment，其中必有一块是某个 segment 的最后一块。
然后轮流拿每一块当 out_cache_loc 调算子，每次调用后立刻 synchronize：

  roomy  按 batch * 16 分配（补丁之后的做法）——访问全部落在张量内，预期永不 fault；
  tight  按 batch * draft 分配（SGLang 现在的做法）——预期扫到某个 segment 末尾时 fault。

先跑 roomy 再跑 tight，因为 fault 之后这个进程的 device 上下文就不可用了。两种模式下都会
先拿结果和 `req_to_token[req, start:end]` 比对，确认多分配不改变取值。

用法（NPU 机器，装好 sgl-kernel-npu wheel）：
    python3 llm/cache_loc_update_oob_check.py
    python3 llm/cache_loc_update_oob_check.py --bs 64 --segments 16   # 越界更远、扫得更多
    python3 llm/cache_loc_update_oob_check.py --mode roomy            # 只验证补丁侧不 fault
退出码：0 = 没有 fault 且取值正确；1 = fault 或取值不对；2 = 环境不可用。
注意：tight 没 fault 不代表没有越界（越界写在源码里是确定的），只说明这次扫到的块后面
恰好都是已映射的显存；可以加大 --segments / --bs 再试。
"""

from __future__ import annotations

import argparse
import sys

import torch

MAX_STEP = 16  # sgl-kernel-npu csrc/cache_location_assign/op_host/tiling/cache_loc_assign.h
SMALL_SEGMENT = 2 << 20  # caching allocator 的小块 segment 大小
MIN_BLOCK = 512  # caching allocator 的最小块


def make_inputs(bs, draft, rows, row_len, req_dtype, gen):
    token_pool = torch.randint(1, 1 << 20, (rows, row_len), generator=gen, dtype=torch.int32)
    req = torch.randperm(rows, generator=gen)[:bs].to(req_dtype)
    start = torch.randint(0, row_len - 2 * MAX_STEP, (bs,), generator=gen, dtype=torch.int64)
    end = start + draft
    golden = torch.stack([token_pool[int(r), int(s) : int(s) + draft] for r, s in zip(req, start)]).flatten()
    return token_pool, req, start, end, golden


def capacity(bs, draft, mode):
    return bs * (max(MAX_STEP, draft) if mode == "roomy" else draft)


def check_values(bs, draft, mode, args, gen):
    token_pool, req, start, end, golden = make_inputs(bs, draft, args.rows, args.row_len, args.req_dtype, gen)
    out = torch.zeros(capacity(bs, draft, mode), dtype=torch.int32).npu()
    torch.ops.npu.cache_loc_update(req.npu(), token_pool.npu(), start.npu(), end.npu(), out)
    torch.npu.synchronize()
    got = out[: bs * draft].cpu()
    ok = torch.equal(got, golden)
    print(f"  [{mode}] bs={bs:3d} draft={draft}: 取值{'一致' if ok else '不一致'}")
    if not ok:
        bad = (got != golden).nonzero().flatten()[:8].tolist()
        print(f"         前几个不一致的位置 {bad}")
    return ok


def hunt(bs, draft, mode, args, gen):
    n_elem = capacity(bs, draft, mode)
    block = max(MIN_BLOCK, -(-n_elem * 4 // MIN_BLOCK) * MIN_BLOCK)
    n_blocks = args.segments * SMALL_SEGMENT // block
    touched = bs * MAX_STEP * 4
    print(
        f"  [{mode}] 每块 {n_elem} 个 int32（分配器块 {block}B），算子实际访问 {touched}B，"
        f"越过块尾 {max(0, touched - block)}B；扫描 {n_blocks} 块（约 {args.segments} 个 2MB segment）"
    )
    token_pool, req, start, end, _ = make_inputs(bs, draft, args.rows, args.row_len, args.req_dtype, gen)
    token_pool, req, start, end = token_pool.npu(), req.npu(), start.npu(), end.npu()
    # 全部持有：这样它们才会按地址顺序铺满 segment，而不是反复复用同一块。
    outs = [torch.empty(n_elem, dtype=torch.int32, device=req.device) for _ in range(n_blocks)]
    torch.npu.synchronize()
    for i, out in enumerate(outs):
        try:
            torch.ops.npu.cache_loc_update(req, token_pool, start, end, out)
            torch.npu.synchronize()
        except Exception as exc:
            ptr = out.data_ptr()
            to_end = SMALL_SEGMENT - ptr % SMALL_SEGMENT
            print(
                f"  [{mode}] FAULT：第 {i} 块，地址 0x{ptr:x}，距所在 2MB 边界 {to_end}B"
                f"（算子从这里起访问 {touched}B）\n         {str(exc).splitlines()[0]}",
                file=sys.stderr,
            )
            return False
        if (i + 1) % 8192 == 0:
            print(f"         已扫 {i + 1} 块，无 fault")
    print(f"  [{mode}] 扫完 {n_blocks} 块，无 fault")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs", type=int, default=16, help="批大小；<= 8 时访问落在 512B 块内，永远不会 fault")
    p.add_argument("--draft", type=int, default=4, help="speculative_num_draft_tokens")
    p.add_argument("--segments", type=int, default=8, help="要铺满并扫描的 2MB segment 个数")
    p.add_argument("--mode", choices=["both", "roomy", "tight"], default="both")
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--row-len", type=int, default=4096)
    p.add_argument("--req-dtype", choices=["int64", "int32"], default="int64")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.req_dtype = getattr(torch, args.req_dtype)
    if args.bs > args.rows:
        args.rows = args.bs

    try:
        import torch_npu  # noqa: F401
        import sgl_kernel_npu  # noqa: F401  注册 torch.ops.npu 下的自定义算子
    except ImportError as exc:
        print(f"环境不可用：{exc}", file=sys.stderr)
        return 2
    if not hasattr(torch.ops.npu, "cache_loc_update"):
        print("torch.ops.npu.cache_loc_update 未注册", file=sys.stderr)
        return 2

    gen = torch.Generator().manual_seed(args.seed)
    modes = ["roomy", "tight"] if args.mode == "both" else [args.mode]
    ok = True
    for mode in modes:
        print(f"== {mode}：out_cache_loc 按 batch * {'max(16, draft)' if mode == 'roomy' else 'draft'} 分配 ==")
        for bs in sorted({1, 8, 9, args.bs}):
            ok &= check_values(bs, args.draft, mode, args, gen)
        if not hunt(args.bs, args.draft, mode, args, gen):
            print("\n结论：复现到 device fault（device 上下文已不可用，终止）。")
            if mode == "tight":
                print("      roomy 不 fault、tight fault —— 就是 out_cache_loc 的越界访问。")
            return 1
    print("\n结论：" + ("未触发 fault，取值全部正确。" if ok else "取值有不一致，见上。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
