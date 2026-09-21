#!/usr/bin/env python3
"""多序列压力自检：torch.ops.npu.chunk_gated_delta_rule（GDN prefill 的 AscendC 算子）。

为什么需要：单请求 e2e（llm/curl.sh）里 prefill 永远是 b=1，
llm/gdn_prefill_layout_check.py 也只测过 b=1、小头数、不带 chunk_state。而并发下的
prefill 是 **多条变长序列 + chunk_state + 真实头数（nk=16, nv=48, dk=dv=128）**，
这个组合此前只在崩溃的 e2e 里出现过。这里不加载模型、用随机输入把它单独跑出来，
每次调用后立刻 synchronize，所以 device fault 会直接落在出错的那一次调用上。

每个 trial 检查三件事：
  1. 不 fault、输出全有限；
  2. chunk_state 的网格：每条序列第 0 块的"进入态"必须逐位等于它的 initial_state
     （算子原样拷贝，bf16 -> bf16），任何一条对不上就说明多序列时 chunk 偏移错了；
  3. 整批结果与"逐条 b=1 调用"的结果一致（同一个 kernel，只差分组，应当几乎逐位相同）。

用法（NPU 机器，装好 sgl-kernel-npu wheel）：
    python3 llm/gdn_prefill_batch_stress.py --model-config /mnt/share/weights/Qwen3.5-27B
    python3 llm/gdn_prefill_batch_stress.py --trials 50 --max-batch 32 --max-len 2048
    python3 llm/gdn_prefill_batch_stress.py --chunk-state off     # 只测不带 chunk_state
退出码：0 = 全部通过；1 = 有 trial 不一致或 fault；2 = 环境不可用。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

CHUNK = 64  # 算子的 chunk 大小（csrc/chunk_gated_delta_rule: CHUNK_SIZE）


def make_inputs(lens, nk, nv, dk, dv, gen):
    t = sum(lens)
    b = len(lens)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    # 生产里 q/k 在调用前已经 l2norm；g 是 <= 0 的 log 衰减，beta 在 (0, 1)。
    q = torch.nn.functional.normalize(rand(t, nk, dk), p=2, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(rand(t, nk, dk), p=2, dim=-1).to(torch.bfloat16)
    v = (rand(t, nv, dv) * 0.5).to(torch.bfloat16)
    beta = torch.sigmoid(rand(t, nv)).to(torch.bfloat16)
    g = -torch.rand(t, nv, generator=gen, dtype=torch.float32) * 0.1
    init = (rand(b, nv, dv, dk) * 0.05).to(torch.bfloat16)
    return q, k, v, beta, g, init


def call_op(q, k, v, beta, g, init, lens, dk, with_chunk_state):
    nv, dv = v.shape[1], v.shape[2]
    lens_t = torch.tensor(lens, dtype=torch.int32).npu()
    h = None
    if with_chunk_state:
        chunks = sum((n + CHUNK - 1) // CHUNK for n in lens)
        # 故意填 NaN：算子没写到的行会在后面的 isfinite 检查里暴露出来。
        h = torch.full((chunks, nv, dv, dk), float("nan"), dtype=torch.bfloat16).npu()
    out, state = torch.ops.npu.chunk_gated_delta_rule(
        q.npu(),
        k.npu(),
        v.npu(),
        beta=beta.npu(),
        initial_state=init.npu(),
        actual_seq_lengths=lens_t,
        scale=dk**-0.5,
        g=g.npu(),
        chunk_state=h,
    )
    torch.npu.synchronize()  # fault 必须落在这一次调用上，而不是下一次
    return out.float().cpu(), state.float().cpu(), None if h is None else h.float().cpu()


def run_trial(idx, lens, nk, nv, dk, dv, gen, with_chunk_state, tol):
    q, k, v, beta, g, init = make_inputs(lens, nk, nv, dk, dv, gen)
    tag = f"trial {idx:3d} b={len(lens):2d} T={sum(lens):5d} chunk_state={'on' if with_chunk_state else 'off'}"
    try:
        out, state, h = call_op(q, k, v, beta, g, init, lens, dk, with_chunk_state)
    except Exception as exc:  # device fault：后续调用都不可信，直接终止
        print(f"{tag}: FAULT lens={lens}\n  {exc}", file=sys.stderr)
        raise

    problems = []
    if not torch.isfinite(out).all() or not torch.isfinite(state).all():
        problems.append("out/state 含 NaN/Inf")

    if h is not None:
        if not torch.isfinite(h).all():
            rows = (~torch.isfinite(h).flatten(1).all(dim=1)).nonzero().flatten().tolist()
            problems.append(f"chunk_state 有未写入/非有限的行 {rows[:8]}")
        offset = 0
        for bid, n in enumerate(lens):
            if not torch.equal(h[offset], init[bid].float()):
                diff = (h[offset] - init[bid].float()).abs().max().item()
                problems.append(f"序列 {bid} 的第 0 块进入态 != initial_state（max diff {diff:.3e}）")
            offset += (n + CHUNK - 1) // CHUNK

    # 逐条 b=1 重算，与整批结果比对。
    start = 0
    worst = 0.0
    for bid, n in enumerate(lens):
        sl = slice(start, start + n)
        o1, s1, _ = call_op(
            q[sl], k[sl], v[sl], beta[sl], g[sl], init[bid : bid + 1], [n], dk, False
        )
        worst = max(
            worst,
            (o1 - out[sl]).abs().max().item(),
            (s1[0] - state[bid]).abs().max().item(),
        )
        start += n
    if worst > tol:
        problems.append(f"整批与逐条结果不一致（max diff {worst:.3e} > {tol}）")

    status = "OK  " if not problems else "FAIL"
    print(f"{status} {tag} batch-vs-single={worst:.2e}" + ("" if not problems else f"  lens={lens}"))
    for msg in problems:
        print(f"       - {msg}")
    return not problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", help="模型目录或 config.json，取 GDN 头数；不给则用 Qwen3.5-27B 的值")
    p.add_argument("--nk", type=int, default=16)
    p.add_argument("--nv", type=int, default=48)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--dv", type=int, default=128)
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=600, help="单条序列最长 token 数（gsm8k 0-shot 提示词在几百以内）")
    p.add_argument("--chunk-state", choices=["on", "off", "both"], default="both")
    p.add_argument("--tol", type=float, default=2e-2, help="整批 vs 逐条的容差（bf16 累积量级）")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.model_config:
        cfg_path = Path(args.model_config)
        if cfg_path.is_dir():
            cfg_path = cfg_path / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg = cfg.get("text_config", cfg)
        args.nk = int(cfg["linear_num_key_heads"])
        args.nv = int(cfg["linear_num_value_heads"])
        args.dk = int(cfg["linear_key_head_dim"])
        args.dv = int(cfg["linear_value_head_dim"])
    print(f"nk={args.nk} nv={args.nv} dk={args.dk} dv={args.dv}")

    try:
        import torch_npu  # noqa: F401
        import sgl_kernel_npu  # noqa: F401  注册 torch.ops.npu 下的自定义算子
    except ImportError as exc:
        print(f"环境不可用：{exc}", file=sys.stderr)
        return 2
    if not hasattr(torch.ops.npu, "chunk_gated_delta_rule"):
        print("torch.ops.npu.chunk_gated_delta_rule 未注册（wheel 不含 #747？）", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    # 固定几组边界形状，再补随机形状：单 token、恰好整块、跨块、批里混着长短序列。
    fixed = [
        [1],
        [CHUNK],
        [CHUNK + 1],
        [1, 1],
        [CHUNK, CHUNK - 1, CHUNK + 1],
        [130, 7, 64, 200, 1, 65],
        [150] * 16,
    ]
    shapes = list(fixed)
    while len(shapes) < args.trials:
        b = rng.randint(2, args.max_batch)
        shapes.append([rng.randint(1, args.max_len) for _ in range(b)])

    modes = {"on": [True], "off": [False], "both": [False, True]}[args.chunk_state]
    failed = 0
    total = 0
    for with_chunk_state in modes:
        for idx, lens in enumerate(shapes):
            total += 1
            try:
                ok = run_trial(idx, lens, args.nk, args.nv, args.dk, args.dv, gen, with_chunk_state, args.tol)
            except Exception:
                print(f"\nfailed: {failed + 1}/{total}（device fault，已终止）")
                return 1
            failed += 0 if ok else 1
    print(f"\nfailed: {failed}/{total}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
