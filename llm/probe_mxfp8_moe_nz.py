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

Run 1 on A5 (torch_npu 2.10.0.post2, Ascend950PR) settled two questions and
raised the real one:

  * npu_format_cast DOES accept a float8_e4m3fn tensor directly and really
    produces FRACTAL_NZ. (The uint8-view + customize_dtype form vllm-ascend's
    fp8.py uses fails here: no TransData bin for FRACTAL_NZ_C0_32.)
  * But every non-strided weight variant died before reaching the matmul::

        CheckMXTranspose failed. The transposition of weightScale/weight
        should be equal, but actual transpositions are true/false.

    gmm1 asserts that weight and weight_scale carry the SAME transpose flag.
    Run 1 varied only the weight (contiguous / NZ => false) while the scale
    stayed a transpose view (true), so it never tested NZ at all. This is also
    the real mechanism behind the "don't .contiguous() the MoE weight+scale"
    pitfall -- an explicit kernel assertion, not a bandwidth effect.

So the question is how to get NZ and the transpose flag to coexist. The
variants below pair each weight layout with a scale layout that matches it,
and try both orderings: transpose-then-cast (vllm-ascend w8a8_dynamic.py, the
int8 path OrangeRedeng cited, which has no MX scale to keep in sync) and
cast-then-transpose (vllm-ascend fp8.py:109-121, which does).

What is still unknown:
  1. Which (weight, scale) layout pairs pass CheckMXTranspose.
  2. Of those, which stay numerically correct -- an NZ weight against an ND
     e8m0 block scale may pass the assertion and still read scales wrong.
  3. Whether any of them actually beats today's strided-ND view.

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

# (label, hidden, moe_intermediate, experts, tokens, top_k). Run 2 measured only
# the first row and saw contiguous beat the strided view by 52%; with 4 experts
# and top_k 2 each group is huge and long-strided, which is exactly the regime
# that flatters contiguity, so the real Qwen3-30B-A3B expert count has to
# confirm it before the number means anything. K dims stay divisible by 64 so
# the block count K/32 is even (pair-split needs it).
SHAPES = [
    ("decode  E=128 tokens=32  ", 2048, 768, 128, 32, 8),
    ("prefill E=128 tokens=2048", 2048, 768, 128, 2048, 8),
]

WARMUP, ITERS, ROUNDS = 20, 100, 10

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


def _to_nz(w):
    """Cast an e4m3 tensor to NZ. Confirmed working on A5 (run 1)."""
    return torch.ops.npu.npu_format_cast(w.contiguous(), ACL_FORMAT_FRACTAL_NZ)


# ------------------------------------------------------------------- pipeline


def _init_routing(hidden, topk_ids, quant_mode, *, experts, tokens, top_k):
    return torch.ops.npu.npu_moe_init_routing_v2(
        hidden,
        topk_ids,
        active_num=tokens * top_k,
        expert_num=experts,
        expert_tokens_num_type=1,  # COUNT
        expert_tokens_num_flag=True,
        active_expert_range=[0, experts],
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
    """Return (best, worst) round mean in ms.

    Run 3 left the NZ candidate at +1.9%/+3.5%, the same order as the -5.9% a
    losing variant scored on another shape -- so a single number cannot say
    whether that is signal. Reporting the spread across rounds makes the noise
    floor visible: a gap smaller than a variant's own best-to-worst spread is
    not a result.
    """
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    rounds = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.npu.synchronize()
        rounds.append((time.perf_counter() - t0) / ITERS * 1e3)  # ms
    return min(rounds), max(rounds)


def run_shape(label, H, I, E, NUM_TOKENS, TOP_K):
    print("=" * 100)
    print(f"SHAPE {label}  H={H} I={I} E={E} tokens={NUM_TOKENS} top_k={TOP_K}")

    w13_bf = torch.randn(E, 2 * I, H, device=DEVICE, dtype=DTYPE) * 0.05
    w2_bf = torch.randn(E, H, I, device=DEVICE, dtype=DTYPE) * 0.05
    qw13, s13 = torch.ops.npu.npu_dynamic_mx_quant(w13_bf, dst_type=E4M3)
    qw2, s2 = torch.ops.npu.npu_dynamic_mx_quant(w2_bf, dst_type=E4M3)

    hidden = torch.randn(NUM_TOKENS, H, device=DEVICE, dtype=DTYPE)
    topk_ids = torch.randint(
        0, E, (NUM_TOKENS, TOP_K), device=DEVICE, dtype=torch.int32
    )
    routing = dict(experts=E, tokens=NUM_TOKENS, top_k=TOP_K)

    # bf16 per-expert reference over the same permutation.
    ref_states, _, ref_counts, _ = _init_routing(
        hidden, topk_ids, quant_mode=-1, **routing
    )
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

    qx, _, expert_tokens, x_scale_raw = _init_routing(
        hidden, topk_ids, quant_mode=3, **routing
    )
    expert_tokens = expert_tokens.to(torch.int64)
    x_scale = _normalize_scale(x_scale_raw)

    # Each variant pairs a weight layout with the scale layout that matches its
    # transpose flag -- run 1 proved gmm1 rejects a mismatched pair outright.
    # Naming: T = transpose view, C = contiguous, NZ = FRACTAL_NZ.
    variants = {
        # Production today. Both sides transposed => flags agree.
        "A  w=T        s=T   (production)   ": (
            (qw13.transpose(1, 2), qw2.transpose(1, 2)),
            (s13.transpose(1, 2), s2.transpose(1, 2)),
        ),
        # Run 1's B, with the scale made contiguous too so the flags agree.
        # Isolates what contiguity alone costs, before NZ enters the picture.
        "B  w=T.C      s=T.C                ": (
            (qw13.transpose(1, 2).contiguous(), qw2.transpose(1, 2).contiguous()),
            (s13.transpose(1, 2).contiguous(), s2.transpose(1, 2).contiguous()),
        ),
        # vllm-ascend w8a8_dynamic.py order (the int8 path OrangeRedeng cited):
        # transpose, contiguous, then cast. NZ weight vs contiguous ND scale.
        "C  w=NZ(T.C)  s=T.C  (int8 order)  ": (
            (_to_nz(qw13.transpose(1, 2)), _to_nz(qw2.transpose(1, 2))),
            (s13.transpose(1, 2).contiguous(), s2.transpose(1, 2).contiguous()),
        ),
        # vllm-ascend fp8.py order: cast on [E, N, K] FIRST, transpose after, so
        # the weight keeps a transpose flag of true and matches a T scale.
        "D  w=NZ().T   s=T    (fp8.py order)": (
            (_to_nz(qw13).transpose(1, 2), _to_nz(qw2).transpose(1, 2)),
            (s13.transpose(1, 2), s2.transpose(1, 2)),
        ),
    }

    baseline = None
    for vlabel, ((w13, w2), (sc13, sc2)) in variants.items():
        try:
            y = _run_moe(qx, x_scale, expert_tokens, w13, sc13, w2, sc2)
            cos = _cos(y_ref, y)
            ok = cos > 0.97
            ms, worst = _bench(
                lambda w13=w13, w2=w2, sc13=sc13, sc2=sc2: _run_moe(
                    qx, x_scale, expert_tokens, w13, sc13, w2, sc2
                )
            )
            if baseline is None:
                baseline = ms
            delta = (baseline - ms) / baseline * 100
            spread = (worst - ms) / ms * 100
            verdict = "PASS" if ok else "FAIL (numerically wrong -- unusable)"
            print(
                f"    {vlabel}: fmt={torch_npu.get_npu_format(w13)!s:<12} "
                f"cos={cos:.5f} {verdict:<10} {ms:7.3f} ms "
                f"({delta:+5.1f}% vs A, own spread {spread:4.1f}%)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    {vlabel}: RAISED {type(exc).__name__}: {exc}")


def main():
    print("=" * 100)
    print("ENV")
    print(f"    torch     = {torch.__version__}")
    print(f"    torch_npu = {getattr(torch_npu, '__version__', '?')}")
    print(f"    soc       = {torch_npu.npu.get_device_name()}")
    print(f"    bench: warmup={WARMUP} iters={ITERS} rounds={ROUNDS} (min of means)")

    for shape in SHAPES:
        try:
            run_shape(*shape)
        except Exception as exc:  # noqa: BLE001
            print(f"    SHAPE {shape[0]} RAISED {type(exc).__name__}: {exc}")

    print("=" * 100)
    print("Run 3 killed the micro shape: at E=4 contiguity looked like a +58% win,")
    print("at the real 128 experts the same variant turns NEGATIVE (-5.9% decode).")
    print("So the strided views production uses are fine, and NZ shows nothing like")
    print("the ~10% the int path saw -- the best case, D, was +1.9%/+3.5%.")
    print("The only question left is whether D clears the noise floor: compare each")
    print("delta against that variant's own best-to-worst spread. If it does not,")
    print("the answer to the review is simply 'no measurable gain on A5'.")


if __name__ == "__main__":
    main()
