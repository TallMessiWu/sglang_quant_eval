"""A5 MXFP8 dense-linear op-level micro-benchmark.

Goal: isolate the kernel-level BF16 -> MXFP8 speedup from end-to-end dilution.

It times, at Qwen3-8B-ish shapes, for both a prefill-like M and decode-like M:
  - BF16 path:   x @ w.T                         (torch.matmul, bf16)
  - MXFP8 path:  npu_dynamic_mx_quant(x)  +  npu_quant_matmul(...)
and reports each sub-step so you can see how much is activation-quant overhead
vs the actual quantized GEMM.

Run on the A5 box:
    python scripts/bench_mxfp8_dense_op.py

If the MXFP8 GEMM here is ~20% faster than BF16 but your end-to-end
bench_serving shows ~0% gain, the loss is NOT in the kernel -> look at graph
capture / per-step dispatch overhead (eager fallback), not at this op.
"""

import torch
import torch_npu

MXFP8_BLOCK_SIZE = 32
_FLOAT8_E8M0FNU_DTYPE = getattr(
    torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None)
)

DEVICE = f"npu:{torch.npu.current_device()}"
DTYPE = torch.bfloat16

# (name, M tokens, K in-features, N out-features) — Qwen3-8B dense shapes.
# qkv_proj: K=4096 N=4096+1024+1024=6144 ; o_proj K=4096 N=4096
# gate_up:  K=4096 N=2*12288=24576       ; down  K=12288 N=4096
SHAPES = [
    # decode-like (small M): step time is what bench_serving decode cares about
    ("decode  gate_up", 1, 4096, 24576),
    ("decode  down   ", 1, 12288, 4096),
    ("decode  qkv    ", 1, 4096, 6144),
    # batched decode
    ("bs64    gate_up", 64, 4096, 24576),
    ("bs64    down   ", 64, 12288, 4096),
    # prefill-like (large M)
    ("prefill gate_up", 4096, 4096, 24576),
    ("prefill down   ", 4096, 12288, 4096),
]

ITERS = 100
WARMUP = 20


def _sync():
    torch.npu.synchronize()


def _time(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / iters  # ms per call


def make_mxfp8_weight(w_bf16):
    """Mimic process_weights_after_loading: quantise + strided transpose view."""
    qw, w_scale = torch_npu.npu_dynamic_mx_quant(
        w_bf16, dst_type=torch_npu.float8_e4m3fn
    )
    # [out, in] -> [in, out] strided view (NO .contiguous(), matches prod code)
    return qw.transpose(0, 1), w_scale.transpose(0, 1)


def bench_one(name, M, K, N):
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)  # [out, in]

    # ---- BF16 reference ----
    wt = w.t().contiguous()  # [in, out]
    t_bf16 = _time(lambda: torch.matmul(x, wt))

    # ---- MXFP8 ----
    qw_t, w_scale_t = make_mxfp8_weight(w)

    def act_quant():
        return torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)

    t_actq = _time(act_quant)

    qx, input_scale = act_quant()

    def gemm():
        return torch_npu.npu_quant_matmul(
            qx,
            qw_t,
            w_scale_t,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            bias=None,
            output_dtype=DTYPE,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

    t_gemm = _time(gemm)
    t_mxfp8_total = t_actq + t_gemm

    speedup_gemm = t_bf16 / t_gemm
    speedup_total = t_bf16 / t_mxfp8_total
    print(
        f"{name} | M={M:>5} K={K:>5} N={N:>5} | "
        f"bf16 {t_bf16:7.3f}ms | mxfp8 gemm {t_gemm:7.3f}ms (x{speedup_gemm:4.2f}) "
        f"+ actquant {t_actq:7.3f}ms = {t_mxfp8_total:7.3f}ms (x{speedup_total:4.2f})"
    )


if __name__ == "__main__":
    print(f"device={DEVICE} dtype={DTYPE} e8m0={_FLOAT8_E8M0FNU_DTYPE}")
    print("-" * 110)
    for name, M, K, N in SHAPES:
        try:
            bench_one(name, M, K, N)
        except Exception as e:  # noqa: BLE001 — surface per-shape failures, keep going
            print(f"{name} | M={M} K={K} N={N} | FAILED: {type(e).__name__}: {e}")
    print("-" * 110)
    print(
        "Read: 'mxfp8 gemm xN' = pure kernel speedup vs bf16. "
        "If that is ~1.2 but your serving gain is ~0, the loss is graph/dispatch "
        "overhead (eager), not the kernel."
    )
