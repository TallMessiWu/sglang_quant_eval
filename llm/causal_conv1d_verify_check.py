#!/usr/bin/env python3
"""单算子自检：torch.ops.npu.causal_conv1d 的 MTP verify 模式（run_mode=1 + num_accepted_tokens）。

这是 GDN + NEXTN verify 链上唯一没有单测覆盖的 AscendC 算子：
  - tests/python/sgl_kernel_npu/test_conv1d_prefill.py 只测 run_mode=0；
  - test_conv1d_update.py / test_mamba_conv.py 测的是另外两个实现
    （torch.ops.npu.causal_conv1d_update 和 triton 的 causal_conv1d_update_v2）。
SGLang 的 ascend_gdn_backend.py 在 target_verify 里调的是 causal_conv1d(run_mode=1,
num_accepted_tokens=...)，本脚本按同样的参数形式调用并和 CPU float32 参考实现对比。

语义（与 test_mamba_conv.py::native_causal_conv1d_update_mtp 一致）：
    full   = cat([conv_states[slot], x_req])          # [state_len + seq, dim]
    y[t]   = silu(sum_j weight[j] * full[-(w-1+seq)+t+j])
    新 state = full[-state_len:]                      # 纯搬运，必须逐位一致
其中 state_len = (width - 1) + (draft_token_num - 1)，末尾 width-1 行就是“全部 draft
token 都被接受”时的窗口，spec_utils.conv_state_rollback 再按 accept 长度右移回退。

用法：
    python3 llm/causal_conv1d_verify_check.py                       # 默认几组形状
    python3 llm/causal_conv1d_verify_check.py --model-config /mnt/share/weights/Qwen3.5-27B
    python3 llm/causal_conv1d_verify_check.py --dry-run             # 只跑 CPU 参考实现
退出码：0 全部通过；1 有用例不通过；2 环境/算子不可用。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch

# bf16 卷积输出的容差；state 是纯数据搬运，按逐位相等检查
OUT_RTOL = 2e-2
OUT_ATOL = 2e-2


@dataclass
class Case:
    name: str
    bs: int
    seq: int  # 每个请求一次调用处理的 token 数（verify 时 = draft_token_num）
    dim: int
    width: int
    spec: bool  # 是否传 num_accepted_tokens（走 kernel 的 spec decoding 分支）

    @property
    def state_len(self) -> int:
        # verify：window 里要留出 draft-1 格给回退；decode：只有 width-1
        return self.width - 1 + (self.seq - 1 if self.spec else 0)

    def __str__(self) -> str:
        return (
            f"{self.name} bs={self.bs} seq={self.seq} dim={self.dim} "
            f"width={self.width} state_len={self.state_len} spec={self.spec}"
        )


def make_inputs(case: Case, seed: int):
    gen = torch.Generator().manual_seed(seed)
    pool = case.bs + 3

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype)

    return {
        "x": randn(case.bs * case.seq, case.dim),
        # SGLang 侧传的是 layer.conv_weights.transpose(0, 1) → [width, dim]
        "weight_t": randn(case.width, case.dim),
        "conv_states": randn(pool, case.state_len, case.dim),
        "bias": randn(case.dim),
        "cache_indices": torch.randperm(pool, generator=gen)[: case.bs].to(torch.int32),
        "query_start_loc": torch.arange(
            0, case.bs * case.seq + 1, step=case.seq, dtype=torch.int32
        ),
        "num_accepted_tokens": torch.full((case.bs,), case.seq, dtype=torch.int32),
        "pool": pool,
    }


def reference(case: Case, inp, use_bias: bool):
    """CPU float32 参考实现，返回 (卷积输出, 更新后的 conv_states)。"""
    w = inp["weight_t"].to(torch.float32)  # [width, dim]
    bias = inp["bias"].to(torch.float32) if use_bias else None
    states = inp["conv_states"].to(torch.float32).clone()
    y = torch.zeros(case.bs * case.seq, case.dim, dtype=torch.float32)

    for b in range(case.bs):
        slot = int(inp["cache_indices"][b])
        if slot < 0:  # pad_slot_id
            continue
        hist = inp["conv_states"][slot].to(torch.float32)
        x = inp["x"][b * case.seq : (b + 1) * case.seq].to(torch.float32)
        full = torch.cat([hist, x], dim=0)
        comp = full[-(case.width - 1 + case.seq) :]
        for t in range(case.seq):
            acc = (comp[t : t + case.width] * w).sum(dim=0)
            if bias is not None:
                acc = acc + bias
            y[b * case.seq + t] = acc * torch.sigmoid(acc)  # silu
        states[slot] = full[-case.state_len :]

    return y, states


def run_npu(case: Case, inp, use_bias: bool):
    conv_states = inp["conv_states"].npu().clone()
    kwargs = dict(
        conv_states=conv_states,
        bias=inp["bias"].npu() if use_bias else None,
        query_start_loc=inp["query_start_loc"].npu(),
        cache_indices=inp["cache_indices"].npu(),
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )
    if case.spec:
        kwargs["num_accepted_tokens"] = inp["num_accepted_tokens"].npu()

    y = torch.ops.npu.causal_conv1d(
        inp["x"].npu().clone(), inp["weight_t"].npu(), **kwargs
    )
    return y.to(torch.float32).cpu(), conv_states.to(torch.float32).cpu()


def report_out(got: torch.Tensor, want: torch.Tensor) -> bool:
    diff = (got - want).abs()
    bad = diff > (OUT_ATOL + OUT_RTOL * want.abs())
    ok = not bool(bad.any())
    print(
        f"    卷积输出: {'通过' if ok else '未通过'}"
        f"  最大绝对误差={diff.max().item():.3e}"
        f"  超容差元素={int(bad.sum())}/{bad.numel()}"
    )
    if not ok:
        flat = int(bad.reshape(-1).nonzero()[0])
        tok, ch = divmod(flat, got.shape[-1])
        print(
            f"      首个不一致: token={tok} channel={ch} "
            f"npu={got.reshape(-1)[flat].item():.6f} ref={want.reshape(-1)[flat].item():.6f}"
        )
    return ok


def report_state(case: Case, inp, got: torch.Tensor, want: torch.Tensor) -> bool:
    """state 只是 [旧 window, 新 token] 的切片搬运，必须逐位一致。"""
    ok = torch.equal(got, want)
    bad = int((got != want).sum())
    print(f"    conv_states: {'逐位一致' if ok else f'不一致，{bad} 个元素不同'}")
    if not ok:
        for b in range(case.bs):
            slot = int(inp["cache_indices"][b])
            g, wv = got[slot], want[slot]
            if torch.equal(g, wv):
                continue
            rows = [r for r in range(case.state_len) if not torch.equal(g[r], wv[r])]
            print(f"      slot={slot}(req {b}) 不一致的 window 行: {rows}")
            # 报告 kernel 实际写入的行来自参考序列的哪一行，便于看出错位
            full = torch.cat(
                [
                    inp["conv_states"][slot].to(torch.float32),
                    inp["x"][b * case.seq : (b + 1) * case.seq].to(torch.float32),
                ],
                dim=0,
            )
            for r in rows[:4]:
                match = [i for i in range(full.shape[0]) if torch.equal(full[i], g[r])]
                print(
                    f"        row {r}: 期望 full[{full.shape[0] - case.state_len + r}]，"
                    f"实际内容匹配 full{match if match else ' 里没有对应行（不是纯搬运）'}"
                )
            break
    return ok


def default_cases(dim: int, width: int) -> list[Case]:
    return [
        # 对照组：decode 形状（不传 num_accepted_tokens），生产里已知可用
        Case("decode-control", 4, 1, dim, width, spec=False),
        # verify：SGLang 的 MTP 调用形式
        Case("verify", 1, 2, dim, width, spec=True),
        Case("verify", 1, 4, dim, width, spec=True),
        Case("verify", 4, 4, dim, width, spec=True),
        Case("verify", 8, 4, dim, width, spec=True),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs", type=int, help="batch size")
    p.add_argument("--seq", type=int, help="draft_token_num")
    p.add_argument("--dim", type=int, default=8192, help="conv 通道数 = q_dim+k_dim+v_dim")
    p.add_argument("--width", type=int, default=4, help="linear_conv_kernel_dim")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bias", action="store_true", help="不传 bias")
    p.add_argument("--dry-run", action="store_true", help="只跑 CPU 参考实现")
    p.add_argument(
        "--model-config",
        help="模型目录或 config.json，自动算 conv 通道数与 kernel 宽度",
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
            nk = int(cfg["linear_num_key_heads"])
            nv = int(cfg["linear_num_value_heads"])
            dk = int(cfg["linear_key_head_dim"])
            dv = int(cfg["linear_value_head_dim"])
            args.width = int(cfg["linear_conv_kernel_dim"])
        except KeyError as exc:
            print(f"{cfg_path} 里没有 GDN 字段 {exc}", file=sys.stderr)
            return 2
        args.dim = nk * dk * 2 + nv * dv
        print(
            f"取自 {cfg_path}: nk={nk} nv={nv} dk={dk} dv={dv} "
            f"→ conv dim={args.dim} width={args.width}"
        )

    if args.bs is not None or args.seq is not None:
        cases = [
            Case("verify", args.bs or 1, args.seq or 4, args.dim, args.width, spec=True)
        ]
    else:
        cases = default_cases(args.dim, args.width)

    if not args.dry_run:
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            print(f"import torch_npu 失败：{exc}", file=sys.stderr)
            return 2
        try:
            import sgl_kernel_npu  # noqa: F401
        except ImportError as exc:
            print(f"import sgl_kernel_npu 失败（wheel 未安装？）：{exc}", file=sys.stderr)
            return 2
        if not hasattr(torch.ops.npu, "causal_conv1d"):
            print(
                "torch.ops.npu.causal_conv1d 未注册："
                "sgl_kernel_npu 已加载，但这个 wheel 没有把该算子编进去",
                file=sys.stderr,
            )
            return 2
        print("算子已注册：torch.ops.npu.causal_conv1d")

    use_bias = not args.no_bias
    failed = 0
    for case in cases:
        print(f"用例 {case}")
        inp = make_inputs(case, args.seed)
        ref_y, ref_states = reference(case, inp, use_bias)
        if args.dry_run:
            print(
                f"    参考实现输出 {tuple(ref_y.shape)}，"
                f"state {tuple(ref_states.shape)}"
            )
            continue
        npu_y, npu_states = run_npu(case, inp, use_bias)
        ok = report_out(npu_y.reshape(ref_y.shape), ref_y)
        ok &= report_state(case, inp, npu_states, ref_states)
        failed += 0 if ok else 1

    if args.dry_run:
        print("dry-run 完成（未调用 NPU）")
        return 0
    print(f"用例总数 {len(cases)}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
