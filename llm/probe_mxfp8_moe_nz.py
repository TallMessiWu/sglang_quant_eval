"""A5 probe: does FRACTAL_NZ weight layout work (and help) for the MXFP8 MoE?

PR #30768 review (OrangeRedeng): "Did you try to use npu_format_cast (NZ
conversion) for mxfp8 case? Previously for int cases it gives ~10% E2E
performance improvement".

The int8 MoE reference is vllm-ascend
``AscendW8A8DynamicFusedMoEMethod.process_weights_after_loading``::

    layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
    layer.w13_weight.data = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)

Our MXFP8 path instead keeps the weights as *strided transpose views* (no
.contiguous()), because for ND that preserves the block-scale mapping and the
K-major reduction stride. NZ is a physical re-tiling, so it is mutually
exclusive with the strided view: to try NZ we must go contiguous first.

Three unknowns this probe answers, in order:
  1. Does npu_format_cast accept a float8_e4m3fn tensor at all (directly, or
     only through the uint8-view + customize_dtype form vllm-ascend's fp8.py
     uses)?
  2. Do npu_grouped_matmul_swiglu_quant_v2 (gmm1) and npu_grouped_matmul (gmm2)
     accept an NZ weight while the e8m0 block scale stays ND -- and stay
     numerically correct (cos vs the bf16 per-expert reference)?
  3. Is it actually faster than today's strided-ND view?

Run on the A5 box:  python llm/probe_mxfp8_moe_nz.py
"""

import time

import torch
import torch.nn.functional as F
import torch_npu

E4M3 = torch.float8_e4m3fn
# e8m0 MUST come from torch_npu for the grouped matmuls (int 293); see AGENTS.md.
E8M0 = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))
ACL_FORMAT_FRACTAL_NZ = 29

DEVICE = f"npu:{torch.npu.current_device()}"
DTYPE = torch.bfloat16

# Qwen3-30B-A3B-ish shapes, shrunk but NZ-alignable (fp8 needs k%16, n%32).
H = 2048  # hidden_size
I = 768  # moe_intermediate_size
E = 4  # experts
NUM_TOKENS = 512  # large enough that the matmuls, not the launch, dominate
TOP_K = 2

WARMUP, ITERS = 5, 30

torch.manual_seed(0)


def _stat(name, t):
    if not isinstance(t, torch.Tensor):
        print(f"    {name}: {t}")
        return
    fmt = torch_npu.get_npu_format(t)
    print(
        f"    {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"contig={t.is_contiguous()} npu_format={fmt}"
    )


def _cos(ref, out):
    return F.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0
    ).item()


def _normalize_scale(scale):
    if scale is None or scale.ndim != 2:
        return scale
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


def _swiglu_ref(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


# ---------------------------------------------------------------- NZ variants


def _to_nz_direct(w):
    """Cast the e4m3 tensor straight to NZ."""
    return torch.ops.npu.npu_format_cast(w.contiguous(), ACL_FORMAT_FRACTAL_NZ)


def _to_nz_via_uint8(w):
    """vllm-ascend fp8.py form: feed the uint8 view, declare the real dtype."""
    return torch.ops.npu.npu_format_cast(
        w.contiguous().view(torch.uint8),
        ACL_FORMAT_FRACTAL_NZ,
        customize_dtype=E4M3,
    )


def probe_format_cast_support(qw13):
    """Unknown #1: which npu_format_cast spelling accepts fp8, if any."""
    print("=" * 100)
    print("1) npu_format_cast on float8_e4m3fn  [E, K, N] transposed weight")
    w = qw13.transpose(1, 2)
    working = {}
    for label, fn in (
        ("direct (e4m3 tensor)     ", _to_nz_direct),
        ("uint8 view + customize   ", _to_nz_via_uint8),
    ):
        try:
            out = fn(w)
            print(f"    {label}: OK")
            _stat(f"      -> {label}", out)
            if torch_npu.get_npu_format(out) != ACL_FORMAT_FRACTAL_NZ:
                print("      !! format is NOT 29 -- the cast silently no-op'd")
            else:
                working[label.strip()] = fn
        except Exception as exc:  # noqa: BLE001
            print(f"    {label}: RAISED {type(exc).__name__}: {exc}")
    return working


# ------------------------------------------------------------------- pipeline


def _init_routing(hidden, topk_ids, quant_mode):
    return torch.ops.npu.npu_moe_init_routing_v2(
        hidden,
        topk_ids,
        active_num=NUM_TOKENS * TOP_K,
        expert_num=E,
        expert_tokens_num_type=1,  # COUNT
        expert_tokens_num_flag=True,
        active_expert_range=[0, E],
        quant_mode=quant_mode,
    )


def _run_moe(qx, x_scale, expert_tokens, w13, w13_scale, w2, w2_scale):
    g1_out, g1_scale = torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
        x=qx,
        weight=[w13],
        group_list=expert_tokens.cumsum(0),
        weight_scale=[w13_scale],
        x_scale=x_scale,
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=E4M3,
        x_dtype=None,
        weight_dtype=None,
        weight_scale_dtype=E8M0,
        x_scale_dtype=E8M0,
    )
    return torch.ops.npu.npu_grouped_matmul(
        x=[g1_out],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[_normalize_scale(g1_scale)],
        scale_dtype=E8M0,
        per_token_scale_dtype=E8M0,
        x_dtype=None,
        weight_dtype=None,
        split_item=2,
        group_list_type=1,  # COUNT
        group_type=0,
        group_list=expert_tokens,
        output_dtype=DTYPE,
    )[0]


def _bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms


def main():
    print("=" * 100)
    print("ENV")
    print(f"    torch     = {torch.__version__}")
    print(f"    torch_npu = {getattr(torch_npu, '__version__', '?')}")
    print(f"    soc       = {torch_npu.npu.get_device_name()}")
    print(f"    shapes: H={H} I={I} E={E} tokens={NUM_TOKENS} top_k={TOP_K}")

    w13_bf = torch.randn(E, 2 * I, H, device=DEVICE, dtype=DTYPE) * 0.05
    w2_bf = torch.randn(E, H, I, device=DEVICE, dtype=DTYPE) * 0.05
    qw13, s13 = torch.ops.npu.npu_dynamic_mx_quant(w13_bf, dst_type=E4M3)
    qw2, s2 = torch.ops.npu.npu_dynamic_mx_quant(w2_bf, dst_type=E4M3)

    working = probe_format_cast_support(qw13)
    if not working:
        print("\nNo npu_format_cast spelling works for e4m3 -> NZ is a dead end here.")
        return

    hidden = torch.randn(NUM_TOKENS, H, device=DEVICE, dtype=DTYPE)
    topk_ids = torch.randint(
        0, E, (NUM_TOKENS, TOP_K), device=DEVICE, dtype=torch.int32
    )

    # bf16 per-expert reference over the same permutation.
    ref_states, _, ref_counts, _ = _init_routing(hidden, topk_ids, quant_mode=-1)
    ref_counts = ref_counts.to(torch.int64)
    parts, off = [], 0
    for e in range(E):
        n = int(ref_counts[e])
        xe = ref_states[off : off + n]
        parts.append(
            _swiglu_ref(xe @ w13_bf[e].transpose(0, 1)) @ w2_bf[e].transpose(0, 1)
        )
        off += n
    y_ref = torch.cat(parts, 0)

    qx, _, expert_tokens, x_scale_raw = _init_routing(hidden, topk_ids, quant_mode=3)
    expert_tokens = expert_tokens.to(torch.int64)
    x_scale = _normalize_scale(x_scale_raw)

    # Scales stay ND transposed views in every variant -- only the weight layout
    # changes. If a variant needs a contiguous scale too, that shows up as a raise.
    s13_t, s2_t = s13.transpose(1, 2), s2.transpose(1, 2)

    variants = {
        "A  ND strided view (production today)": (
            qw13.transpose(1, 2),
            qw2.transpose(1, 2),
        ),
        "B  ND contiguous                     ": (
            qw13.transpose(1, 2).contiguous(),
            qw2.transpose(1, 2).contiguous(),
        ),
    }
    for label, fn in working.items():
        variants[f"C  NZ via {label:<24}"] = (
            fn(qw13.transpose(1, 2)),
            fn(qw2.transpose(1, 2)),
        )

    print("=" * 100)
    print("2+3) correctness and latency per weight layout")
    baseline = None
    for label, (w13, w2) in variants.items():
        try:
            y = _run_moe(qx, x_scale, expert_tokens, w13, s13_t, w2, s2_t)
            cos = _cos(y_ref, y)
            ok = cos > 0.97
            ms = _bench(
                lambda w13=w13, w2=w2: _run_moe(
                    qx, x_scale, expert_tokens, w13, s13_t, w2, s2_t
                )
            )
            if baseline is None:
                baseline = ms
            delta = (baseline - ms) / baseline * 100
            verdict = "PASS" if ok else "FAIL (numerically wrong -- unusable)"
            print(
                f"    {label}: cos={cos:.5f} {verdict:<38} "
                f"{ms:7.3f} ms  ({delta:+5.1f}% vs A)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    {label}: RAISED {type(exc).__name__}: {exc}")

    print("=" * 100)
    print("Read it as: a NZ variant is worth landing only if it both PASSes and")
    print("beats A by more than run-to-run noise. If NZ passes for gmm2 but not")
    print("gmm1 (or vice versa), rerun with per-gmm variants before concluding.")


if __name__ == "__main__":
    main()
