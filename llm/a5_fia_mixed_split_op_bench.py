#!/usr/bin/env python3
"""Compare one mixed FIA call with prefill-first split FIA calls on Ascend NPU."""

import argparse
import math
import statistics

import torch
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--mask-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--min-gain", type=float, default=5.0)
    return parser.parse_args()


def validate_args(args):
    if args.num_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num-heads must be divisible by num-kv-heads")
    if args.prefill_tokens <= 0 or args.decode_tokens <= 0:
        raise ValueError("mixed input requires positive prefill and decode tokens")
    if args.kv_len < args.prefill_tokens:
        raise ValueError("kv-len must be at least prefill-tokens")
    if args.mask_size < args.prefill_tokens:
        raise ValueError("mask-size must be at least prefill-tokens")
    if args.block_size <= 0 or args.head_dim <= 0:
        raise ValueError("block-size and head-dim must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")


def run_fia(
    query,
    key,
    value,
    block_table,
    mask,
    q_cumulative,
    kv_lengths,
    args,
):
    output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        input_layout="TND",
        block_size=args.block_size,
        block_table=block_table,
        atten_mask=mask,
        sparse_mode=3,
        actual_seq_lengths=q_cumulative,
        actual_seq_lengths_kv=kv_lengths,
        scale=1.0 / math.sqrt(args.head_dim),
    )
    return output


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def measure_npu_ms(fn):
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), output


def main():
    args = parse_args()
    validate_args(args)

    torch_npu.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    dtype = torch.bfloat16

    num_requests = 1 + args.decode_tokens
    num_tokens = args.prefill_tokens + args.decode_tokens
    blocks_per_request = math.ceil(args.kv_len / args.block_size)
    num_blocks = num_requests * blocks_per_request

    query = torch.randn(
        (num_tokens, args.num_heads, args.head_dim),
        dtype=dtype,
        device=device,
    )
    key = torch.randn(
        (
            num_blocks,
            args.block_size,
            args.num_kv_heads * args.head_dim,
        ),
        dtype=dtype,
        device=device,
    )
    value = torch.randn_like(key)
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).view(num_requests, blocks_per_request)
    mask = torch.triu(
        torch.ones(
            (args.mask_size, args.mask_size), dtype=torch.int8, device=device
        ),
        diagonal=1,
    )

    mixed_q_cumulative = [args.prefill_tokens] + [
        args.prefill_tokens + index
        for index in range(1, args.decode_tokens + 1)
    ]
    decode_q_cumulative = list(range(1, args.decode_tokens + 1))
    kv_lengths = torch.full(
        (num_requests,), args.kv_len, dtype=torch.int32
    )

    def run_off():
        return run_fia(
            query,
            key,
            value,
            block_table,
            mask,
            mixed_q_cumulative,
            kv_lengths,
            args,
        )

    def run_on():
        prefill = run_fia(
            query[: args.prefill_tokens],
            key,
            value,
            block_table[:1],
            mask,
            [args.prefill_tokens],
            kv_lengths[:1],
            args,
        )
        decode = run_fia(
            query[args.prefill_tokens :],
            key,
            value,
            block_table[1:],
            mask,
            decode_q_cumulative,
            kv_lengths[1:],
            args,
        )
        output = torch.empty_like(query)
        output[: args.prefill_tokens].copy_(prefill)
        output[args.prefill_tokens :].copy_(decode)
        return output

    off_output = run_off()
    on_output = run_on()
    torch_npu.npu.synchronize()
    max_abs_diff = (off_output.float() - on_output.float()).abs().max().item()
    if not torch.allclose(
        off_output,
        on_output,
        rtol=args.rtol,
        atol=args.atol,
    ):
        mismatch_count = torch.count_nonzero(
            ~torch.isclose(
                off_output,
                on_output,
                rtol=args.rtol,
                atol=args.atol,
            )
        ).item()
        raise RuntimeError(
            "OFF and ON outputs are not close: "
            f"mismatches={mismatch_count}, max_abs_diff={max_abs_diff}, "
            f"rtol={args.rtol}, atol={args.atol}"
        )

    for _ in range(args.warmup):
        run_off()
        run_on()
    torch_npu.npu.synchronize()

    timings = {"off": [], "on": []}
    functions = {"off": run_off, "on": run_on}
    for iteration in range(args.iters):
        order = ("off", "on") if iteration % 2 == 0 else ("on", "off")
        for name in order:
            elapsed_ms, _ = measure_npu_ms(functions[name])
            timings[name].append(elapsed_ms)

    off_p50 = statistics.median(timings["off"])
    on_p50 = statistics.median(timings["on"])
    off_p90 = percentile(timings["off"], 0.9)
    on_p90 = percentile(timings["on"], 0.9)
    gain = (off_p50 - on_p50) / off_p50 * 100.0
    target = "PASS" if gain >= args.min_gain else "MISS"

    print(
        "Shape: "
        f"prefill={args.prefill_tokens}, decode={args.decode_tokens}, "
        f"kv_len={args.kv_len}, heads={args.num_heads}/{args.num_kv_heads}, "
        f"head_dim={args.head_dim}"
    )
    print(
        "Correctness: torch.allclose=True "
        f"(rtol={args.rtol:g}, atol={args.atol:g}, "
        f"max_abs_diff={max_abs_diff:.8g})"
    )
    print(f"OFF single FIA: p50={off_p50:.3f} ms, p90={off_p90:.3f} ms")
    print(f"ON split FIA:  p50={on_p50:.3f} ms, p90={on_p90:.3f} ms")
    print(
        f"Median gain: {gain:+.2f}% "
        f"(target >= {args.min_gain:.2f}%: {target})"
    )


if __name__ == "__main__":
    main()
