#!/usr/bin/env python3
"""单算子自检：torch.ops.npu.recurrent_gated_delta_rule（Ascend）。

参考实现与 sgl-kernel-npu 的 tests/python/sgl_kernel_npu/test_recurrent_gated_delta_rule.py
一致，但在 CPU 上用 float32 重算，不依赖 torchair，形状可从命令行给定，有用例不通过就以
非 0 退出，便于在 950 上快速确认算子本身是否正确。

用法：
    python3 llm/recurrent_gated_delta_rule_check.py                    # 默认几组形状
    python3 llm/recurrent_gated_delta_rule_check.py --b 4 --mtp 4 --nk 16 --nv 32
    python3 llm/recurrent_gated_delta_rule_check.py --dry-run          # 不碰 NPU，只跑参考实现
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch

# 与测试脚本一致的容差（bf16 输出的半个 ULP）
TOLERANCE = 2**-8


@dataclass
class Case:
    b: int
    mtp: int
    nk: int
    nv: int
    dk: int
    dv: int

    def __str__(self) -> str:
        return (
            f"B={self.b} MTP={self.mtp} nk={self.nk} nv={self.nv} "
            f"dk={self.dk} dv={self.dv}"
        )


def make_inputs(case: Case, seed: int):
    """在 CPU 上按测试脚本的分布造输入；mix_qkv 按 [q, k, v] 沿最后一维拼接。"""
    gen = torch.Generator().manual_seed(seed)
    b, s, nk, nv, dk, dv = case.b, case.mtp, case.nk, case.nv, case.dk, case.dv
    max_slots = b + 1

    def rand(*shape, dtype=torch.bfloat16):
        return torch.rand(*shape, generator=gen, dtype=torch.float32).to(dtype)

    q = rand(b, s, nk * dk)
    k = rand(b, s, nk * dk)
    v = rand(b, s, nv * dv)
    cache_indices = torch.randperm(max_slots, generator=gen)[:b].to(torch.int32)

    return {
        "mix_qkv": torch.cat([q, k, v], dim=-1).contiguous(),
        "recurrent_state": rand(max_slots, nv, dv, dk),
        "intermediate_state": rand(max_slots, s, nv, dv, dk) if s > 1 else None,
        "cache_indices": cache_indices if s > 1 else None,
        "ssm_state_indices": (
            cache_indices.to(torch.int64)[:, None] * s + torch.arange(s)
        )
        .to(torch.int32)
        .contiguous(),
        "actual_seq_lengths": torch.full((b,), s, dtype=torch.int32),
        "num_accepted_tokens": torch.randint(
            1, s + 1, (b,), generator=gen, dtype=torch.int32
        ),
        "beta": rand(b, s, nv),
        "g": -torch.rand(b, s, nv, generator=gen, dtype=torch.float32),
        "scale": dk**-0.5,
        "max_slots": max_slots,
    }


def reference(case: Case, inp) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU float32 参考实现，返回 (attention 输出, 写回后的 state)。"""
    b, s, nk, nv, dk, dv = case.b, case.mtp, case.nk, case.nv, case.dk, case.dv
    t_all = b * s

    mix = inp["mix_qkv"].to(torch.float32).view(t_all, -1)
    q, k, v = torch.split(mix, [nk * dk, nk * dk, nv * dv], dim=-1)
    q = q.view(t_all, nk, dk)
    k = k.view(t_all, nk, dk)
    v = v.view(t_all, nv, dv)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    q = q * inp["scale"]

    recurrent_state = inp["recurrent_state"].to(torch.float32).clone()
    if inp["intermediate_state"] is not None:
        state = inp["intermediate_state"].to(torch.float32).clone()
        cache = inp["cache_indices"].to(torch.int64)
        # 首个 slot 用主 state 初始化，和 kernel 的 needRecurrentInit 一致
        state[cache, 0] = recurrent_state[cache]
        state = state.view(-1, nv, dv, dk)
    else:
        state = recurrent_state

    ssm = inp["ssm_state_indices"].to(torch.int64).view(-1)
    nacc = inp["num_accepted_tokens"].to(torch.int64)
    g = inp["g"].to(torch.float32).view(t_all, nv).exp()
    beta = inp["beta"].to(torch.float32).view(t_all, nv).sigmoid()
    out = torch.empty(t_all, nv, dv, dtype=torch.float32)

    seq_start = 0
    for i in range(b):
        init_state = state[ssm[seq_start + nacc[i] - 1]]
        for head in range(nv):
            cur = init_state[head].clone()  # [dv, dk]
            for slot in range(seq_start, seq_start + s):
                qk_head = head // (nv // nk)
                q_i, k_i = q[slot][qk_head], k[slot][qk_head]
                cur = cur * g[slot][head]
                x = (cur * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v[slot][head] - x) * beta[slot][head]
                cur = cur + y[:, None] * k_i[None, :]
                state[ssm[slot]][head] = cur
                out[slot][head] = (cur * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += s

    return out.view(b, s, nv, dv), state


def run_npu(case: Case, inp) -> tuple[torch.Tensor, torch.Tensor]:
    """在 NPU 上调用算子，返回 (attention 输出, 写回后的 state)。"""
    b, s, nv, dk, dv = case.b, case.mtp, case.nv, case.dk, case.dv

    recurrent_state = inp["recurrent_state"].npu().clone()
    intermediate = None
    if inp["intermediate_state"] is not None:
        intermediate = inp["intermediate_state"].npu().clone().view(-1, nv, dv, dk)

    out = torch.ops.npu.recurrent_gated_delta_rule(
        inp["mix_qkv"].npu(),
        recurrent_state,
        beta=inp["beta"].npu(),
        scale=inp["scale"],
        actual_seq_lengths=inp["actual_seq_lengths"].npu(),
        ssm_state_indices=inp["ssm_state_indices"].npu(),
        nk=case.nk,
        nv=nv,
        intermediate_state=intermediate,
        cache_indices=(
            inp["cache_indices"].npu() if inp["cache_indices"] is not None else None
        ),
        num_accepted_tokens=inp["num_accepted_tokens"].npu(),
        g=inp["g"].npu(),
    )
    state = intermediate if intermediate is not None else recurrent_state
    return out.to(torch.float32).cpu(), state.to(torch.float32).cpu()


def report(name: str, got: torch.Tensor, want: torch.Tensor) -> bool:
    got = got.reshape(-1).to(torch.float32)
    want = want.reshape(-1).to(torch.float32)
    diff = (got - want).abs()
    bad = diff > (TOLERANCE + TOLERANCE * want.abs())
    ok = not bool(bad.any())
    print(
        f"    {name}: {'通过' if ok else '未通过'}"
        f"  最大绝对误差={diff.max().item():.3e}"
        f"  超容差元素={int(bad.sum())}/{bad.numel()}"
    )
    if not ok:
        idx = int(bad.nonzero()[0])
        print(f"      首个不一致: index={idx} npu={got[idx].item():.9f} ref={want[idx].item():.9f}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b", type=int, help="batch size")
    p.add_argument("--mtp", type=int, help="每个请求一次 verify 的 token 数（kernel 上限 8）")
    p.add_argument("--nk", type=int, default=8, help="key/query head 数")
    p.add_argument("--nv", type=int, default=16, help="value head 数")
    p.add_argument("--dk", type=int, default=128, help="key/query head_dim")
    p.add_argument("--dv", type=int, default=128, help="value head_dim")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="只跑 CPU 参考实现，不调用 NPU")
    p.add_argument(
        "--model-config",
        help="模型目录或 config.json，自动取 GDN 的 linear_num_*_heads / linear_*_head_dim",
    )
    args = p.parse_args()

    if args.model_config:
        import json
        from pathlib import Path

        cfg_path = Path(args.model_config)
        if cfg_path.is_dir():
            cfg_path = cfg_path / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg = cfg.get("text_config", cfg)
        try:
            args.nk = int(cfg["linear_num_key_heads"])
            args.nv = int(cfg["linear_num_value_heads"])
            args.dk = int(cfg["linear_key_head_dim"])
            args.dv = int(cfg["linear_value_head_dim"])
        except KeyError as exc:
            print(f"{cfg_path} 里没有 GDN 头数字段 {exc}", file=sys.stderr)
            return 2
        print(f"取自 {cfg_path}: nk={args.nk} nv={args.nv} dk={args.dk} dv={args.dv}")

    if args.b is not None or args.mtp is not None:
        cases = [
            Case(args.b or 4, args.mtp or 4, args.nk, args.nv, args.dk, args.dv)
        ]
    else:
        cases = [
            Case(2, 1, args.nk, args.nv, args.dk, args.dv),
            Case(4, 2, args.nk, args.nv, args.dk, args.dv),
            Case(4, 4, args.nk, args.nv, args.dk, args.dv),
            Case(8, 8, args.nk, args.nv, args.dk, args.dv),
        ]

    if not args.dry_run:
        try:
            import torch_npu  # noqa: F401  提供 torch.ops.npu 命名空间
        except ImportError as exc:
            print(f"import torch_npu 失败：{exc}", file=sys.stderr)
            return 2
        try:
            # 自定义算子由 wheel 里的 libsgl_kernel_npu.so 注册，只 import torch_npu 探测不到
            import sgl_kernel_npu  # noqa: F401
        except ImportError as exc:
            print(f"import sgl_kernel_npu 失败（wheel 未安装？）：{exc}", file=sys.stderr)
            return 2

        if not hasattr(torch.ops.npu, "recurrent_gated_delta_rule"):
            print(
                "torch.ops.npu.recurrent_gated_delta_rule 未注册："
                "sgl_kernel_npu 已加载，但这个 wheel 没有把该算子编进去",
                file=sys.stderr,
            )
            return 2
        print("算子已注册：torch.ops.npu.recurrent_gated_delta_rule")

    failed = 0
    for case in cases:
        if case.mtp > 8:
            print(f"跳过 {case}：MTP 超过 kernel 的 MAX_MTP=8")
            continue
        print(f"用例 {case}")
        inp = make_inputs(case, args.seed)
        ref_out, ref_state = reference(case, inp)
        if args.dry_run:
            print(f"    参考实现输出 {tuple(ref_out.shape)}，state {tuple(ref_state.shape)}")
            continue
        npu_out, npu_state = run_npu(case, inp)
        ok = report("attention 输出", npu_out, ref_out)
        ok &= report("state", npu_state, ref_state.view_as(npu_state))
        failed += 0 if ok else 1

    if args.dry_run:
        print("dry-run 完成（未调用 NPU）")
        return 0
    print(f"用例总数 {len(cases)}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
