#!/usr/bin/env python3
"""单算子自检：torch.ops.npu.recurrent_gated_delta_rule（Ascend）。

参考实现与 sgl-kernel-npu 的 tests/python/sgl_kernel_npu/test_recurrent_gated_delta_rule.py
一致，但在 CPU 上用 float32 重算，不依赖 torchair，形状可从命令行给定，有用例不通过就以
非 0 退出，便于在 950 上快速确认算子本身是否正确。

用法：
    python3 llm/recurrent_gated_delta_rule_check.py                    # 默认几组形状
    python3 llm/recurrent_gated_delta_rule_check.py --b 4 --mtp 4 --nk 16 --nv 32
    python3 llm/recurrent_gated_delta_rule_check.py --dry-run          # 不碰 NPU，只跑参考实现

两个 wheel 的 A/B（例如 PR #808 的 arch35 实现 vs #823 的只放开编译）：
    # 各自装好 wheel 后，用同一个 seed 各跑一次，把输出存下来
    python3 llm/recurrent_gated_delta_rule_check.py --bench 100 --dump /tmp/pr808.pt
    python3 llm/recurrent_gated_delta_rule_check.py --bench 100 --dump /tmp/pr823.pt
    # 再直接比两边的输出（不经过容差，逐元素比）
    python3 llm/recurrent_gated_delta_rule_check.py --compare /tmp/pr808.pt /tmp/pr823.pt
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
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


def make_inputs(case: Case, seed: int, state_dtype: torch.dtype = torch.bfloat16):
    """在 CPU 上按测试脚本的分布造输入；mix_qkv 按 [q, k, v] 沿最后一维拼接。

    state_dtype 单独给，因为生产里 q/k/v/beta 是 bf16 而 SSM pool 是 float32。
    """
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
        "recurrent_state": rand(max_slots, nv, dv, dk, dtype=state_dtype),
        "intermediate_state": (
            rand(max_slots, s, nv, dv, dk, dtype=state_dtype) if s > 1 else None
        ),
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


def upload(case: Case, inp, strided_state: bool = False):
    """把一组输入搬到 NPU，返回 (args, kwargs, 写回目标, 可复位的物理张量)。

    只搬一次，供 run_npu 和 bench_npu 共用：计时时不能把 H2D 拷贝算进去。

    strided_state=True 时按生产形态传入主 pool：SGLang 在 NPU + 投机解码下会把
    temporal_state 换成 transpose(-1, -2) 的非连续视图（memory_pool.py 里的
    `if _is_npu: temporal_state = temporal_state.transpose(-1, -2)`）。算子 host 没有对
    recurrent_state 调 .contiguous()，kernel 直接按物理内存读，所以生产里**物理排布不变**，
    变的只是 torch 看到的逻辑形状。这里照同样的方式构造。
    """
    b, s, nv, dk, dv = case.b, case.mtp, case.nv, case.dk, case.dv

    # phys 始终是 (…, nv, dv, dk) 的物理排布，也就是算子和参考实现共同的约定
    phys = inp["recurrent_state"].npu().clone()
    if strided_state:
        recurrent_state = phys.transpose(-1, -2)
        assert not recurrent_state.is_contiguous()
    else:
        recurrent_state = phys
    intermediate = None
    if inp["intermediate_state"] is not None:
        intermediate = inp["intermediate_state"].npu().clone().view(-1, nv, dv, dk)

    args = (inp["mix_qkv"].npu(), recurrent_state)
    kwargs = dict(
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
    # 算子原地写 state：intermediate 存在时写它，否则写主 pool 的物理张量 phys
    # （strided 时传进去的是 phys 的转置视图，写回的物理内存还是 phys）
    written = intermediate if intermediate is not None else phys
    return args, kwargs, written


def run_npu(case: Case, inp, strided_state: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """在 NPU 上调用一次算子，返回 (attention 输出, 写回后的 state)。"""
    args, kwargs, written = upload(case, inp, strided_state)
    out = torch.ops.npu.recurrent_gated_delta_rule(*args, **kwargs)
    return out.to(torch.float32).cpu(), written.to(torch.float32).cpu()


def bench_npu(case: Case, inp, strided_state: bool, iters: int, rounds: int, warmup: int):
    """计时：每轮连续发射 iters 次，轮末 synchronize，取每次调用的耗时（毫秒）。

    算子原地改 state，连续调用会让 state 漂移；每轮开始前从快照复位，复位不计时。
    形状固定、kernel 无数据相关分支，所以漂移不影响耗时，只是别拿这里的数值当精度用。
    """
    args, kwargs, written = upload(case, inp, strided_state)
    snapshot = written.clone()

    for _ in range(warmup):
        torch.ops.npu.recurrent_gated_delta_rule(*args, **kwargs)
    torch.npu.synchronize()

    per_round = []
    for _ in range(rounds):
        written.copy_(snapshot)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.ops.npu.recurrent_gated_delta_rule(*args, **kwargs)
        torch.npu.synchronize()
        per_round.append((time.perf_counter() - t0) / iters * 1e3)
    return min(per_round), statistics.median(per_round)


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


def compare_dumps(path_a: str, path_b: str) -> int:
    """逐元素比两个 --dump 的结果。

    比的是「同一份输入、同一个 seed，两个 wheel 各自算出什么」，不走 TOLERANCE：
    两边各自跟 golden 比都在容差内，也可能彼此差很多，那更值得知道。
    """
    dumps = {}
    for path in (path_a, path_b):
        d = torch.load(path, map_location="cpu", weights_only=False)
        meta = d.get("meta", {})
        print(f"{path}: seed={meta.get('seed')} wheel={meta.get('wheel')}")
        dumps[path] = d

    a, b = dumps[path_a]["results"], dumps[path_b]["results"]
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    for key in sorted(only_a):
        print(f"  只在 {path_a} 里: {key}")
    for key in sorted(only_b):
        print(f"  只在 {path_b} 里: {key}")

    shared = sorted(set(a) & set(b))
    worst = 0.0
    differing = 0
    for key in shared:
        print(f"用例 {key}")
        for field in ("out", "state"):
            ta, tb = a[key][field], b[key][field]
            if ta.shape != tb.shape:
                print(f"    {field}: 形状不同 {tuple(ta.shape)} vs {tuple(tb.shape)}")
                differing += 1
                continue
            diff = (ta - tb).abs().max().item()
            worst = max(worst, diff)
            same = torch.equal(ta, tb)
            if not same:
                differing += 1
            print(f"    {field}: {'逐位相同' if same else '有差异'}  最大绝对差={diff:.3e}")
        ba, bb = a[key].get("bench"), b[key].get("bench")
        if ba and bb:
            speedup = bb[0] / ba[0] if ba[0] else float("nan")
            print(
                f"    耗时: {os.path.basename(path_a)} {ba[0]:.4f}ms"
                f"  {os.path.basename(path_b)} {bb[0]:.4f}ms"
                f"  （前者快 {speedup:.2f}x，取每组最快一轮）"
            )

    print(f"共 {len(shared)} 个用例，{differing} 项有差异，全局最大绝对差={worst:.3e}")
    return 1 if (differing or only_a or only_b) else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b", type=int, help="batch size")
    p.add_argument("--mtp", type=int, help="每个请求一次 verify 的 token 数（kernel 上限 8）")
    p.add_argument("--nk", type=int, default=8, help="key/query head 数")
    p.add_argument("--nv", type=int, default=16, help="value head 数")
    p.add_argument("--dk", type=int, default=128, help="key/query head_dim")
    p.add_argument("--dv", type=int, default=128, help="value head_dim")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--state-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="SSM state 的 dtype；生产（MambaPool 的 ssm_dtype）是 float32",
    )
    p.add_argument(
        "--strided-state",
        action="store_true",
        help="按生产形态把主 pool 作为 transpose(-1,-2) 的非连续视图传进去（物理排布不变）",
    )
    p.add_argument(
        "--matrix",
        action="store_true",
        help="对每个形状跑 dtype x 连续性 四种组合，用来定位生产形态下才出现的问题",
    )
    p.add_argument("--dry-run", action="store_true", help="只跑 CPU 参考实现，不调用 NPU")
    p.add_argument(
        "--bench",
        type=int,
        default=0,
        metavar="N",
        help="每轮连续发射 N 次算子并计时；0（默认）不测性能",
    )
    p.add_argument("--bench-rounds", type=int, default=5, help="计时轮数，报告最快与中位那轮")
    p.add_argument("--bench-warmup", type=int, default=10, help="计时前的预热次数")
    p.add_argument(
        "--dump",
        metavar="PATH",
        help="把每个用例的输出（和 --bench 的耗时）存到文件，供 --compare 跨 wheel 对比",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="比较两个 --dump 文件，逐元素对比输出并列出耗时；不碰 NPU",
    )
    p.add_argument(
        "--model-config",
        help="模型目录或 config.json，自动取 GDN 的 linear_num_*_heads / linear_*_head_dim",
    )
    args = p.parse_args()

    if args.compare:
        return compare_dumps(*args.compare)

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

    if args.matrix:
        variants = [
            (torch.bfloat16, False),
            (torch.bfloat16, True),
            (torch.float32, False),
            (torch.float32, True),
        ]
    else:
        variants = [
            (getattr(torch, args.state_dtype), args.strided_state),
        ]

    failed = 0
    total = 0
    results = {}
    for case in cases:
        if case.mtp > 8:
            print(f"跳过 {case}：MTP 超过 kernel 的 MAX_MTP=8")
            continue
        print(f"用例 {case}")
        for state_dtype, strided in variants:
            inp = make_inputs(case, args.seed, state_dtype)
            ref_out, ref_state = reference(case, inp)
            if args.dry_run:
                print(
                    f"  state dtype={str(state_dtype).split('.')[-1]} "
                    f"非连续={strided}: 参考实现输出 {tuple(ref_out.shape)}，"
                    f"state {tuple(ref_state.shape)}"
                )
                continue
            label = (
                f"state dtype={str(state_dtype).split('.')[-1]} 非连续={strided}"
                + ("  ← 生产形态" if (state_dtype is torch.float32 and strided) else "")
            )
            print(f"  {label}")
            total += 1
            try:
                npu_out, npu_state = run_npu(case, inp, strided)
            except Exception as exc:  # noqa: BLE001  算子直接报错也是结论
                print(f"    算子调用失败: {type(exc).__name__}: {exc}")
                failed += 1
                continue
            ok = report("attention 输出", npu_out, ref_out)
            ok &= report("state", npu_state, ref_state.view_as(npu_state))
            failed += 0 if ok else 1

            record = {"out": npu_out, "state": npu_state}
            if args.bench:
                best, med = bench_npu(
                    case, inp, strided, args.bench, args.bench_rounds, args.bench_warmup
                )
                print(
                    f"    耗时: 最快 {best:.4f}ms  中位 {med:.4f}ms"
                    f"  （{args.bench} 次/轮 × {args.bench_rounds} 轮）"
                )
                record["bench"] = (best, med)
            if args.dump:
                results[f"{case} | state={str(state_dtype).split('.')[-1]} | 非连续={strided}"] = record

    if args.dry_run:
        print("dry-run 完成（未调用 NPU）")
        return 0

    if args.dump:
        import sgl_kernel_npu  # noqa: PLC0415  只为记录这次用的是哪个 wheel

        torch.save(
            {
                "meta": {
                    "seed": args.seed,
                    "wheel": getattr(sgl_kernel_npu, "__file__", None),
                    "argv": sys.argv[1:],
                },
                "results": results,
            },
            args.dump,
        )
        print(f"已写入 {args.dump}（{len(results)} 个用例）")

    print(f"检查总数 {total}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
