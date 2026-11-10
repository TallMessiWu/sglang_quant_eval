#!/usr/bin/env python3
"""Wait for automatic NPU analysis; never launch, retry, or terminate a parser."""

import argparse
import json
import sys
import time
from pathlib import Path


def wait_for_traces(root: Path, expected_ranks: int) -> None:
    captures = sorted(path for path in root.rglob("*_ascend_pt") if path.is_dir())
    if not captures:
        raise RuntimeError(f"未找到原始 NPU 采集目录：{root}")
    if len(captures) != expected_ranks:
        raise RuntimeError(
            f"期望 {expected_ranks} 份 rank 采集结果，实际只有 {len(captures)} 份：{root}"
        )

    pending = set(captures)
    announced = set()
    while pending:
        for capture in captures:
            if capture not in pending:
                continue
            status_path = capture / "sglang_analysis.json"
            trace = capture / "ASCEND_PROFILER_OUTPUT/trace_view.json"
            try:
                status = json.loads(status_path.read_text())
            except FileNotFoundError:
                status = None  # The detached interpreter may still be starting.

            if status is not None and status.get("state") == "failed":
                raise RuntimeError(
                    f"解析失败：{capture}：{status.get('error')}；"
                    f"原始数据保留，检查 {capture / 'sglang_analysis.log'}"
                )
            if status is not None and status.get("state") not in (
                "running",
                "succeeded",
            ):
                raise RuntimeError(
                    f"未知解析状态：{status_path}：{status.get('state')!r}"
                )
            if status is None or status.get("state") == "succeeded":
                if trace.is_file() and trace.stat().st_size > 0:
                    # Native non-daemon analysis has no SGLang status file.
                    print(f"NPU trace: {trace}", flush=True)
                    pending.remove(capture)
                    continue
                if status is not None:
                    raise RuntimeError(
                        f"解析报告成功，但缺少非空 trace_view.json：{capture}"
                    )
            if capture not in announced:
                print(
                    f"等待后台解析：{status_path}（Ctrl-C 只停止等待，不终止解析）",
                    flush=True,
                )
                announced.add(capture)
        if pending:
            time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=1)
    args = parser.parse_args()
    try:
        wait_for_traces(args.directory.resolve(), args.expected_ranks)
    except KeyboardInterrupt:
        print(
            "已停止等待；未终止后台解析。请稍后检查 sglang_analysis.json。",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"生成成功，但 profiling 验证失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
