#!/usr/bin/env python3
"""Compare mixed FIA calls across real Qwen model attention shapes on Ascend NPU."""

import argparse
import math
import statistics
from dataclasses import dataclass

import torch
import torch_npu


@dataclass(frozen=True)
class ModelShape:
    model_id: str
    num_heads: int
    num_kv_heads: int
    head_dim: int


MODEL_PRESETS = {
    "qwen3.5-0.8b": ModelShape("Qwen/Qwen3.5-0.8B", 8, 2, 256),
    "qwen3.5-2b": ModelShape("Qwen/Qwen3.5-2B", 8, 2, 256),
    "qwen3.5-4b": ModelShape("Qwen/Qwen3.5-4B", 16, 4, 256),
    "qwen3.5-9b": ModelShape("Qwen/Qwen3.5-9B", 16, 4, 256),
    "qwen3.5-27b": ModelShape("Qwen/Qwen3.5-27B", 24, 4, 256),
    "qwen3.5-35b-a3b": ModelShape("Qwen/Qwen3.5-35B-A3B", 16, 2, 256),
    "qwen3.5-122b-a10b": ModelShape("Qwen/Qwen3.5-122B-A10B", 32, 2, 256),
    "qwen3.5-397b-a17b": ModelShape("Qwen/Qwen3.5-397B-A17B", 32, 2, 256),
    "qwen3-0.6b": ModelShape("Qwen/Qwen3-0.6B", 16, 8, 128),
    "qwen3-1.7b": ModelShape("Qwen/Qwen3-1.7B", 16, 8, 128),
    "qwen3-4b": ModelShape("Qwen/Qwen3-4B", 32, 8, 128),
    "qwen3-8b": ModelShape("Qwen/Qwen3-8B", 32, 8, 128),
    "qwen3-14b": ModelShape("Qwen/Qwen3-14B", 40, 8, 128),
    "qwen3-32b": ModelShape("Qwen/Qwen3-32B", 64, 8, 128),
    "qwen3-30b-a3b": ModelShape("Qwen/Qwen3-30B-A3B", 32, 4, 128),
    "qwen3-235b-a22b": ModelShape("Qwen/Qwen3-235B-A22B", 64, 4, 128),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*MODEL_PRESETS, "custom"],
        help="model presets; defaults to all, deduplicated by local FIA shape",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        help="simulated TP sizes; defaults to 1 2 4 8",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        help="simulate one TP size (backward-compatible alias)",
    )
    parser.add_argument("--num-heads", type=int, help="custom global query heads")
    parser.add_argument("--num-kv-heads", type=int, help="custom global KV heads")
    parser.add_argument("--head-dim", type=int, help="custom attention head dimension")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--mask-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--min-gain", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    custom_values = (args.num_heads, args.num_kv_heads, args.head_dim)
    if args.models is None:
        args.models = (
            ["custom"]
            if any(value is not None for value in custom_values)
            else list(MODEL_PRESETS)
        )
    if args.tp_sizes is not None and args.tp_size is not None:
        parser.error("--tp-sizes and --tp-size cannot be used together")
    args.tp_sizes = list(
        dict.fromkeys(
            args.tp_sizes
            if args.tp_sizes is not None
            else ([args.tp_size] if args.tp_size is not None else [1, 2, 4, 8])
        )
    )
    return args


def validate_args(args):
    if args.prefill_tokens <= 0 or args.decode_tokens <= 0:
        raise ValueError("mixed input requires positive prefill and decode tokens")
    if args.kv_len < args.prefill_tokens:
        raise ValueError("kv-len must be at least prefill-tokens")
    if args.mask_size < args.prefill_tokens:
        raise ValueError("mask-size must be at least prefill-tokens")
    if args.block_size <= 0 or any(tp_size <= 0 for tp_size in args.tp_sizes):
        raise ValueError("block-size and all tp-sizes must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    custom_values = (args.num_heads, args.num_kv_heads, args.head_dim)
    if "custom" in args.models:
        if args.models != ["custom"]:
            raise ValueError("custom cannot be combined with model presets")
        if any(value is None or value <= 0 for value in custom_values):
            raise ValueError(
                "custom requires positive --num-heads, --num-kv-heads, "
                "and --head-dim"
            )
    elif any(value is not None for value in custom_values):
        raise ValueError("custom head arguments require --models custom")


def resolve_model_shape(model_name, args, tp_size):
    if model_name == "custom":
        shape = ModelShape(
            "custom", args.num_heads, args.num_kv_heads, args.head_dim
        )
    else:
        shape = MODEL_PRESETS[model_name]

    if shape.num_heads % tp_size != 0:
        raise ValueError(
            f"{shape.model_id}: query heads {shape.num_heads} are not divisible "
            f"by TP={tp_size}"
        )
    if shape.num_kv_heads >= tp_size:
        if shape.num_kv_heads % tp_size != 0:
            raise ValueError(
                f"{shape.model_id}: KV heads {shape.num_kv_heads} are not "
                f"divisible by TP={tp_size}"
            )
    elif tp_size % shape.num_kv_heads != 0:
        raise ValueError(
            f"{shape.model_id}: TP={tp_size} cannot evenly replicate "
            f"{shape.num_kv_heads} KV heads"
        )

    return ModelShape(
        shape.model_id,
        shape.num_heads // tp_size,
        max(1, shape.num_kv_heads // tp_size),
        shape.head_dim,
    )


def run_fia(
    query,
    key,
    value,
    block_table,
    mask,
    q_cumulative,
    kv_lengths,
    shape,
    block_size,
):
    output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=shape.num_heads,
        num_key_value_heads=shape.num_kv_heads,
        input_layout="TND",
        block_size=block_size,
        block_table=block_table,
        atten_mask=mask,
        sparse_mode=3,
        actual_seq_lengths=q_cumulative,
        actual_seq_lengths_kv=kv_lengths,
        scale=1.0 / math.sqrt(shape.head_dim),
    )
    return output


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def one_sided_sign_test(gains, threshold):
    above = sum(gain > threshold for gain in gains)
    below = sum(gain < threshold for gain in gains)
    sample_count = above + below
    if sample_count == 0:
        return 1.0, above, sample_count
    tail_count = sum(
        math.comb(sample_count, successes)
        for successes in range(above, sample_count + 1)
    )
    return tail_count / (2**sample_count), above, sample_count


def measure_npu_ms(fn):
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), output


def benchmark_model(model_name, represented_presets, tp_size, args, device):
    shape = resolve_model_shape(model_name, args, tp_size)
    num_requests = 1 + args.decode_tokens
    num_tokens = args.prefill_tokens + args.decode_tokens
    blocks_per_request = math.ceil(args.kv_len / args.block_size)
    num_blocks = num_requests * blocks_per_request

    query = torch.randn(
        (num_tokens, shape.num_heads, shape.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        (
            num_blocks,
            args.block_size,
            shape.num_kv_heads * shape.head_dim,
        ),
        dtype=torch.bfloat16,
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
            shape,
            args.block_size,
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
            shape,
            args.block_size,
        )
        decode = run_fia(
            query[args.prefill_tokens :],
            key,
            value,
            block_table[1:],
            mask,
            decode_q_cumulative,
            kv_lengths[1:],
            shape,
            args.block_size,
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
    paired_gains = [
        (off_ms - on_ms) / off_ms * 100.0
        for off_ms, on_ms in zip(timings["off"], timings["on"])
    ]
    median_gain = statistics.median(paired_gains)
    p_value, successes, test_samples = one_sided_sign_test(
        paired_gains, args.min_gain
    )
    reject_null = p_value < args.alpha

    represented_models = [
        (
            MODEL_PRESETS[preset].model_id
            if preset != "custom"
            else "custom"
        )
        for preset in represented_presets
    ]
    print(f"Models: {', '.join(represented_models)}")
    print(f"Presets: {', '.join(represented_presets)} (TP={tp_size})")
    print(
        "Shape: "
        f"prefill={args.prefill_tokens}, decode={args.decode_tokens}, "
        f"kv_len={args.kv_len}, local_heads={shape.num_heads}/"
        f"{shape.num_kv_heads}, head_dim={shape.head_dim}"
    )
    print(
        "Correctness: torch.allclose=True "
        f"(rtol={args.rtol:g}, atol={args.atol:g}, "
        f"max_abs_diff={max_abs_diff:.8g})"
    )
    print(f"OFF single FIA: p50={off_p50:.3f} ms, p90={off_p90:.3f} ms")
    print(f"ON split FIA:  p50={on_p50:.3f} ms, p90={on_p90:.3f} ms")
    print(f"Paired median gain: {median_gain:+.2f}% ({args.iters} pairs)")
    print("Test: exact one-sided paired sign test")
    print(f"H0: median paired gain <= {args.min_gain:.2f}%")
    print(f"Ha: median paired gain > {args.min_gain:.2f}%")
    print(
        f"alpha={args.alpha:g}, p-value={p_value:.6g}, "
        f"pairs above H0 boundary={successes}/{test_samples}"
    )
    print(
        "Decision: "
        + ("REJECT H0" if reject_null else "FAIL TO REJECT H0")
    )


def main():
    args = parse_args()
    validate_args(args)

    torch_npu.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    for tp_index, tp_size in enumerate(args.tp_sizes):
        if tp_index:
            print()
        print(f"=== Simulated TP={tp_size} on one NPU rank ===")

        shape_groups = {}
        for model_name in args.models:
            shape = resolve_model_shape(model_name, args, tp_size)
            shape_key = (shape.num_heads, shape.num_kv_heads, shape.head_dim)
            shape_groups.setdefault(shape_key, []).append(model_name)

        for shape_index, represented_presets in enumerate(
            shape_groups.values()
        ):
            print()
            torch.manual_seed(args.seed + tp_index * 1000 + shape_index)
            benchmark_model(
                represented_presets[0],
                represented_presets,
                tp_size,
                args,
                device,
            )


if __name__ == "__main__":
    main()
