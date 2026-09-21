#!/usr/bin/env python3
"""给 MTP（spec v2）每个阶段后面加一次 device 同步，定位异步 device fault 出在哪一段（可回滚）。

背景：Ascend 上的 "vector core exception / error code 507035" 是异步错误，Python 栈
只会指到**下一个同步点**。overlap 调度下一轮迭代里唯一的同步点是 verify replay 前的
`forward_batch.seq_lens.cpu()`，所以栈永远落在那里，而真正出错的 kernel 可能属于上一轮的
任何一段：draft、target verify、mamba state 提交、draft extend，或者中间插进来的 prefill。

这个补丁在 eagle_worker_v2.py 末尾追加一段 monkeypatch（不改任何已有行，所以不怕上游
挪代码），只有设了 SGLANG_MTP_PHASE_SYNC=1 才生效。生效后每段结束都 synchronize 一次：

    draft                 draft 模型多步解码
    target_verify+sample  target verify（graph replay）+ 采样
    mamba_commit          verify 之后的 SSM/conv state 提交（move_intermediate_cache 等）
    draft_extend_decode   decode 迭代尾部的 draft extend
    target_extend         prefill 的 target forward（GDN prefill 走 chunk 算子）
    draft_extend_prefill  prefill 之后的 draft extend

fault 冒出来时日志里会多一行：

    [phase-sync] device fault surfaced right after phase=<段名> ...raw_bs=15 graph_bs=16...

另外 verify 的 (raw_bs, graph_bs) 每次变化都会打一行，能直接看出 fault 是不是紧跟着
第一次 padded replay（投机解码的捕获 bs 是 1..8,10,12,14,16...，15 会 pad 到 16）。

注意：同步会吃掉 overlap 的收益，吞吐明显下降，只用于定位；如果加了同步就不再复现，
这本身也是结论（说明是时序/流间竞态而不是确定性的越界）。

用法（NPU 机器，SGLang 源码安装）：
    python3 llm/patch_mtp_phase_sync.py --apply
    SGLANG_MTP_PHASE_SYNC=1 MTP=1 ./llm/qwen3.5_dense_bf16.sh 0   # 然后照常跑 gsm8k
    grep phase-sync <服务日志>
    python3 llm/patch_mtp_phase_sync.py --restore
    python3 llm/patch_mtp_phase_sync.py --show
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- MTP phase sync (patch_mtp_phase_sync.py) ---"

BLOCK = '''

# --- MTP phase sync (patch_mtp_phase_sync.py) ---
# Diagnostic only: a device fault is asynchronous and surfaces at the next sync
# point, so sync after every spec-v2 phase to attribute it. Enabled by
# SGLANG_MTP_PHASE_SYNC=1; remove with patch_mtp_phase_sync.py --restore.
def _install_mtp_phase_sync():
    import functools
    import os

    if os.environ.get("SGLANG_MTP_PHASE_SYNC") != "1":
        return

    import sglang.srt.speculative.eagle_worker_common as _ewc

    _state = {"iter": 0, "last_shape": None}

    def _device_sync():
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()

    def _batch_info(batch):
        try:
            return (
                f"mode={batch.forward_mode.name} "
                f"bs={batch.seq_lens.shape[0] if batch.seq_lens is not None else 0}"
            )
        except Exception as exc:  # diagnostics must never break the forward
            return f"batch=? ({exc!r})"

    def _graph_info(target_worker):
        runner = getattr(target_worker.model_runner, "decode_cuda_graph_runner", None)
        if runner is None:
            return "graph=off"
        return (
            f"raw_bs={getattr(runner, 'raw_bs', None)} "
            f"graph_bs={getattr(runner, 'bs', None)}"
        )

    def _sync(phase, info):
        try:
            _device_sync()
        except Exception:
            logger.error(
                "[phase-sync] device fault surfaced right after phase=%s iter=%d %s",
                phase,
                _state["iter"],
                info,
            )
            raise

    def _wrap_method(cls, name, *, before=None, after=None):
        orig = getattr(cls, name)

        @functools.wraps(orig)
        def wrapper(self, batch, *args, **kwargs):
            if before is not None:
                _sync(before, _batch_info(batch))
            out = orig(self, batch, *args, **kwargs)
            if after is not None:
                _sync(after, _batch_info(batch))
            return out

        setattr(cls, name, wrapper)

    _orig_commit = _ewc.commit_mamba_states_after_verify

    @functools.wraps(_orig_commit)
    def _commit(target_worker, batch, *args, **kwargs):
        _state["iter"] += 1
        graph = _graph_info(target_worker)
        if graph != _state["last_shape"]:
            _state["last_shape"] = graph
            logger.info("[phase-sync] verify shape changed: iter=%d %s", _state["iter"], graph)
        info = f"{_batch_info(batch)} {graph}"
        _sync("target_verify+sample", info)
        out = _orig_commit(target_worker, batch, *args, **kwargs)
        _sync("mamba_commit", info)
        return out

    # run_eagle_verify resolves this name in eagle_worker_common's globals.
    _ewc.commit_mamba_states_after_verify = _commit

    _wrap_method(EagleDraftWorker, "draft", after="draft")
    _wrap_method(
        EagleDraftWorker, "_draft_extend_for_decode", after="draft_extend_decode"
    )
    _wrap_method(
        EagleDraftWorker,
        "_draft_extend_for_prefill",
        before="target_extend",
        after="draft_extend_prefill",
    )
    logger.warning(
        "[phase-sync] enabled: syncing the device after every spec-v2 phase; "
        "throughput is reduced, use for fault attribution only"
    )


_install_mtp_phase_sync()
# --- end phase sync ---
'''


def target_path() -> Path:
    from sglang.srt.speculative import eagle_worker_v2 as mod  # noqa: PLC0415

    return Path(mod.__file__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--restore", action="store_true")
    g.add_argument("--show", action="store_true")
    p.add_argument("--file", help="直接指定 eagle_worker_v2.py（默认从已安装的 sglang 定位）")
    args = p.parse_args()

    path = Path(args.file) if args.file else None
    if path is None:
        try:
            path = target_path()
        except ImportError as exc:
            print(f"无法 import sglang：{exc}", file=sys.stderr)
            return 2
    backup = path.with_suffix(path.suffix + ".phasesync-orig")

    if args.show:
        print(path)
        print(f"  补丁已加入: {MARKER in path.read_text()}")
        print(f"  备份存在: {backup.exists()}")
        return 0
    if args.restore:
        if not backup.exists():
            print(f"没有备份 {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        backup.unlink()
        print(f"已还原 {path}")
        return 0

    text = path.read_text()
    if MARKER in text:
        print("已经加过了")
        return 0
    for needed in ("class EagleDraftWorker", "def _draft_extend_for_decode", "def draft("):
        if needed not in text:
            print(f"{path} 里找不到 {needed!r}，这个版本的结构不符，未修改", file=sys.stderr)
            return 1
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.rstrip("\n") + "\n" + BLOCK)
    print(f"已追加阶段同步补丁 {path}")
    print("启动时加 SGLANG_MTP_PHASE_SYNC=1 才生效；定位完用 --restore 还原")
    return 0


if __name__ == "__main__":
    sys.exit(main())
