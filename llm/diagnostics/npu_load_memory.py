#!/usr/bin/env python3
"""Launch SGLang while recording each worker's ModelRunner.load_model peak.

The import hook preserves SGLang's normal import order and is installed again
when multiprocessing spawn imports this launcher as __mp_main__. It changes no
source files or loading policy. This measures the NPU allocator, not total HBM,
and excludes later KV-pool allocation, graph capture and serving warmup.
"""

import functools
import importlib.abc
import importlib.machinery
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_ENV = "NPU_LOAD_MEMORY_DIR"
TARGET = "sglang.srt.model_executor.model_runner"
GIB = 1 << 30
logger = logging.getLogger("npu_load_memory")


def _npu_backend():
    from sglang.srt.utils import is_npu

    if not is_npu():
        raise RuntimeError("Load-memory recording requires an Ascend NPU environment")
    import torch_npu

    return torch_npu.npu


def _metadata(runner, original, npu, device):
    import torch

    from sglang.srt.runtime_context import get_parallel

    source = Path(original.__code__.co_filename).resolve()
    revision = None
    dirty = None
    try:
        # Do not accidentally report the outer experiment repo for untracked code.
        subprocess.run(
            [
                "git",
                "-C",
                str(source.parent),
                "ls-files",
                "--error-unmatch",
                source.name,
            ],
            check=True,
            capture_output=True,
            timeout=5,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "--no-optional-locks",
                    "-C",
                    str(source.parent),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    parallel = get_parallel()
    args = getattr(runner, "server_args", None)
    return {
        "label": os.environ.get("NPU_LOAD_MEMORY_LABEL"),
        "pid": os.getpid(),
        "logical_device": device,
        "device_name": npu.get_device_name(device),
        "is_draft_worker": bool(getattr(runner, "is_draft_worker", False)),
        "ranks": {
            key: getattr(parallel, key, None)
            for key in ("tp_rank", "tp_size", "pp_rank", "pp_size", "dp_rank")
        },
        "source_file": str(source),
        "source_revision": revision,
        "source_dirty": dirty,
        "torch": str(torch.__version__),
        "torch_npu": str(sys.modules["torch_npu"].__version__),
        "config": {
            key: getattr(args, key, None)
            for key in (
                "model_path",
                "quantization",
                "dtype",
                "tp_size",
                "context_length",
                "mem_fraction_static",
                "max_total_tokens",
                "disable_cuda_graph",
                "load_format",
            )
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_USE_FIA",
                "SGLANG_MAMBA_SSM_DTYPE",
                "PYTORCH_NPU_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
                "ASCEND_LAUNCH_BLOCKING",
            )
        },
    }


def _snapshot(npu, device):
    return {
        key: int(getattr(npu, method)(device))
        for key, method in (
            ("allocated_bytes", "memory_allocated"),
            ("reserved_bytes", "memory_reserved"),
            ("peak_allocated_bytes", "max_memory_allocated"),
            ("peak_reserved_bytes", "max_memory_reserved"),
        )
    }


def _save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, default=str) + "\n")
    temporary.replace(path)


def _wrap_load(original):
    if getattr(original, "_npu_load_memory_wrapped", False):
        return original

    @functools.wraps(original)
    def measured(runner, *args, **kwargs):
        npu = _npu_backend()
        device = npu.current_device()
        record = _metadata(runner, original, npu, device)
        directory = Path(os.environ[OUTPUT_ENV])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"load-memory-{os.getpid()}-{time.time_ns()}.json"
        npu.synchronize()
        npu.reset_peak_memory_stats(device)
        record.update(
            scope="ModelRunner.load_model (construction, weights and post-load; before KV pools/graphs/warmup)",
            units="bytes; GiB = bytes / 2**30",
            status="loading",
            complete=False,
            begin=_snapshot(npu, device),
            started_at_unix=time.time(),
        )
        # A hard-killed worker leaves an explicitly incomplete record.
        _save(path, record)
        started = time.perf_counter()
        try:
            result = original(runner, *args, **kwargs)
            npu.synchronize()
            if getattr(runner, "startup_weight_load", None) is not None:
                record["status"] = "deferred_weights"
                record["error"] = (
                    "Overlapped weights are not finalized inside load_model; this window is incomplete."
                )
            else:
                record.update(status="ok", complete=True)
            return result
        except BaseException as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            try:
                record["end"] = end = _snapshot(npu, device)
                baseline = record["begin"]["allocated_bytes"]
                record["summary_gib"] = {
                    "peak_allocated": end["peak_allocated_bytes"] / GIB,
                    "peak_reserved": end["peak_reserved_bytes"] / GIB,
                    "end_allocated": end["allocated_bytes"] / GIB,
                    "end_reserved": end["reserved_bytes"] / GIB,
                    "peak_allocated_above_begin": (
                        end["peak_allocated_bytes"] - baseline
                    )
                    / GIB,
                }
            except Exception as exc:
                record.update(
                    complete=False, measurement_error=f"{type(exc).__name__}: {exc}"
                )
            try:
                _save(path, record)
                logger.info(
                    "[NPU load memory] status=%s complete=%s summary_gib=%s report=%s",
                    record["status"],
                    record["complete"],
                    record.get("summary_gib"),
                    path,
                )
            except Exception:
                # Never replace the original loading/OOM exception with an I/O error.
                logger.exception("[NPU load memory] Cannot save report %s", path)

    measured._npu_load_memory_wrapped = True
    return measured


class _RecordingLoader:
    def __init__(self, loader):
        self.loader = loader

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        module.ModelRunner.load_model = _wrap_load(module.ModelRunner.load_model)


class _RecordingFinder(importlib.abc.MetaPathFinder):
    _npu_load_memory_finder = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot locate {TARGET} for load-memory recording")
        spec.loader = _RecordingLoader(spec.loader)
        return spec


def install_hook():
    if not os.environ.get(OUTPUT_ENV):
        raise RuntimeError(f"Set {OUTPUT_ENV}; use record_w4a4_load_memory.sh")
    if TARGET in sys.modules:
        cls = sys.modules[TARGET].ModelRunner
        cls.load_model = _wrap_load(cls.load_model)
    if not any(
        getattr(finder, "_npu_load_memory_finder", False) for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _RecordingFinder())


def summarize(paths):
    files = sorted(
        {
            file
            for path in map(Path, paths)
            for file in ([path] if path.is_file() else path.rglob("load-memory-*.json"))
        }
    )
    if not files:
        raise SystemExit("No load-memory reports found")
    print(
        "run\ttp\tdevice\trole\tstatus\tpeak_alloc_GiB\tpeak_reserved_GiB\tend_alloc_GiB\tend_reserved_GiB\trevision"
    )
    incomplete = False
    for file in files:
        record = json.loads(file.read_text())
        valid = record.get("complete") and record.get("status") == "ok"
        incomplete |= not valid
        summary = record.get("summary_gib", {})
        values = [
            f"{summary[key]:.3f}" if key in summary else "-"
            for key in (
                "peak_allocated",
                "peak_reserved",
                "end_allocated",
                "end_reserved",
            )
        ]
        print(
            "\t".join(
                map(
                    str,
                    [
                        file.parent.name,
                        record.get("ranks", {}).get("tp_rank"),
                        record.get("logical_device"),
                        "draft" if record.get("is_draft_worker") else "target",
                        "ok" if valid else f"INCOMPLETE:{record.get('status')}",
                        *values,
                        (record.get("source_revision") or "unknown")[:12]
                        + ("+dirty" if record.get("source_dirty") else ""),
                    ],
                )
            )
        )
    return int(incomplete)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "Launch: bash ./record_w4a4_load_memory.sh before|after [NPU_ID ...]\n"
            "Summarize: python3 diagnostics/npu_load_memory.py summarize REPORT_DIR [REPORT_DIR ...]\n"
            "Metrics cover ModelRunner.load_model only, not entire startup or total device HBM."
        )
        return 0
    if sys.argv[1] == "summarize":
        return summarize(sys.argv[2:])
    if sys.argv[1] != "serve":
        raise SystemExit("Only serve and summarize are supported")
    install_hook()
    from sglang.cli.main import main as sglang_main

    sglang_main()
    return 0


if __name__ == "__mp_main__":
    install_hook()

if __name__ == "__main__":
    sys.exit(main())
