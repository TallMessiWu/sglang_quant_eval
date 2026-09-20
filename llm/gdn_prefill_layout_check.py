#!/usr/bin/env python3
"""量 GDN prefill 的 triton kernel 到底把 SSM state 写成 (K,V) 还是 (V,K)。

背景：sgl-kernel-npu #747 把 decode 的 triton kernel 翻成了 V-major
（`b_h = tl.zeros([BV, BK])`，注释 "V row K col"），AscendC 的
recurrent_gated_delta_rule 本来就按物理内存读 (nv, dv, dk)；SGLang #39589 随之
去掉了 pool 的转置。但 #747 **没有**动 prefill 这条路径
（sgl_kernel_npu.fla.chunk -> chunk_delta_h），它的 docstring 至今写着
`initial_state ... [N, H, K, V]`。

如果 prefill 仍是 K-major，而 decode/verify 是 V-major，那 prefill 写进 pool 的
state 相对于 verify 读到的就差一个转置 —— 表现为一开口就乱码/截断。

模型的 dk == dv == 128，两种布局形状相同，分不出来。这里用 **dk != dv** 让形状本身
就能区分，再拿 CPU float32 参考实现比对，给出确定结论。

用法（NPU 机器）：
    python3 llm/gdn_prefill_layout_check.py
    python3 llm/gdn_prefill_layout_check.py --dk 64 --dv 32 --tokens 8
退出码：0 = 两条 prefill 路径布局一致；1 = 差一个转置或结果不一致；2 = 环境不可用。
"""

from __future__ import annotations

import argparse
import sys

import torch


def reference_state(q, k, v, g, beta, scale, nk, nv, dk, dv):
    """CPU float32 参考：返回 S[v][k] 语义的最终 state，形状 (nv, dv, dk)。

    与 llm/recurrent_gated_delta_rule_check.py 里那份（已在 950 上和算子对过）同源。
    """
    t = q.shape[0]
    qn = torch.nn.functional.normalize(q.float(), p=2, dim=-1) * scale
    kn = torch.nn.functional.normalize(k.float(), p=2, dim=-1)
    vf, gf, bf = v.float(), g.float().exp(), beta.float()
    state = torch.zeros(nv, dv, dk, dtype=torch.float32)
    for i in range(t):
        for h in range(nv):
            qk_h = h // (nv // nk)
            k_i, v_i = kn[i][qk_h], vf[i][h]
            cur = state[h] * gf[i][h]
            pred = (cur * k_i.unsqueeze(-2)).sum(dim=-1)
            delta = (v_i - pred) * bf[i][h]
            state[h] = cur + delta[:, None] * k_i[None, :]
    return state


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nk", type=int, default=2)
    p.add_argument("--nv", type=int, default=4)
    p.add_argument("--dk", type=int, default=64, help="故意和 dv 不同，用来区分布局")
    p.add_argument("--dv", type=int, default=32)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.dk == args.dv:
        print("dk 必须不等于 dv，否则两种布局形状相同，分辨不出来", file=sys.stderr)
        return 2

    try:
        import torch_npu  # noqa: F401
        import sgl_kernel_npu  # noqa: F401
        from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    except ImportError as exc:
        print(f"需要 NPU 环境（torch_npu + sgl_kernel_npu）：{exc}", file=sys.stderr)
        return 2

    nk, nv, dk, dv, t = args.nk, args.nv, args.dk, args.dv, args.tokens
    gen = torch.Generator().manual_seed(args.seed)
    scale = dk**-0.5

    q = torch.randn(t, nk, dk, generator=gen).to(torch.bfloat16)
    k = torch.randn(t, nk, dk, generator=gen).to(torch.bfloat16)
    v = torch.randn(t, nv, dv, generator=gen).to(torch.bfloat16)
    g = -torch.rand(t, nv, generator=gen)
    beta = torch.rand(t, nv, generator=gen).to(torch.bfloat16)

    want_vk = reference_state(q, k, v, g, beta, scale, nk, nv, dk, dv)  # (nv, dv, dk)
    print(f"形状：nk={nk} nv={nv} dk={dk} dv={dv} tokens={t}")
    print(f"参考 state（V-major 语义）: {tuple(want_vk.shape)}")

    print()
    print("=== 1. triton prefill（sgl_kernel_npu.fla.chunk，SGLang 现在调的就是它）===")
    qn = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(torch.bfloat16)
    kn = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(torch.bfloat16)
    cu = torch.tensor([0, t], dtype=torch.int64).npu()

    o_tri = s_tri = None
    try:
        o_tri, s_tri, _ = chunk_gated_delta_rule_npu(
            q=qn.unsqueeze(0).npu(), k=kn.unsqueeze(0).npu(), v=v.unsqueeze(0).npu(),
            g=g.unsqueeze(0).npu(), beta=beta.unsqueeze(0).npu(), scale=scale,
            initial_state=torch.zeros(1, nv, dk, dv, dtype=torch.bfloat16).npu(),
            output_final_state=True, cu_seqlens=cu,
            head_first=False, use_qk_l2norm_in_kernel=False,
        )
        print(f"  输出 o {tuple(o_tri.shape)}   final_state {tuple(s_tri.shape)}")
        lay = "K-major (nv, dk, dv)" if tuple(s_tri.shape[-2:]) == (dk, dv) else "V-major (nv, dv, dk)"
        print(f"  -> state 布局: {lay}")
    except Exception as exc:  # noqa: BLE001
        print(f"  调用失败 -> {type(exc).__name__}: {exc}")

    print()
    print("=== 2. AscendC chunk 算子（#747 新增，SGLang 尚未改调它）===")
    o_asc = s_asc = None
    try:
        from sgl_kernel_npu.fla.chunk_gated_delta_rule_npu import (
            chunk_gated_delta_rule_npu as chunk_ascendc,
        )

        o_asc, s_asc = chunk_ascendc(
            qn.npu(), kn.npu(), v.npu(),
            beta=beta.npu(), initial_state=torch.zeros(1, nv, dv, dk, dtype=torch.bfloat16).npu(),
            actual_seq_lengths=torch.tensor([t], dtype=torch.int32).npu(),
            scale=scale, g=g.npu(),
        )
        print(f"  输出 o {tuple(o_asc.shape)}   final_state {tuple(s_asc.shape)}")
        lay = "V-major (nv, dv, dk)" if tuple(s_asc.shape[-2:]) == (dv, dk) else "K-major (nv, dk, dv)"
        print(f"  -> state 布局: {lay}")
    except Exception as exc:  # noqa: BLE001
        print(f"  调用失败 -> {type(exc).__name__}: {exc}")

    if o_tri is None or o_asc is None:
        print("\n两边没能都跑起来，看上面的报错")
        return 2

    print()
    print("=== 3. 两者对比（attention 输出与布局无关，先看它）===")
    a = o_tri.float().cpu().reshape(t, nv, dv)
    b = o_asc.float().cpu().reshape(t, nv, dv)
    out_err = (a - b).abs().max().item()
    print(f"  attention 输出最大绝对差 = {out_err:.4e}")

    sa = s_tri.float().cpu().reshape(nv, *s_tri.shape[-2:])
    sb = s_asc.float().cpu().reshape(nv, *s_asc.shape[-2:])
    straight = (sa - sb).abs().max().item() if sa.shape == sb.shape else float("inf")
    flipped = (sa - sb.transpose(-1, -2)).abs().max().item() if sa.shape == sb.transpose(-1, -2).shape else float("inf")
    print(f"  state 直接比   = {straight:.4e}")
    print(f"  state 转置后比 = {flipped:.4e}")

    print()
    tol = 5e-2
    if out_err > tol:
        print(f"结论：两个 prefill 实现的输出本身就不一致（{out_err:.4e}），不只是 state 打包方式的问题")
        return 1
    if flipped < straight:
        print("结论：两者算的是同一个东西，但 state 打包方式差一个转置。")
        print("      SGLang 调的是 triton 那条（K-major），而 pool / verify 算子是 V-major -> 错配。")
        return 1
    print("结论：两者的 state 布局一致，prefill 不是问题所在。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
