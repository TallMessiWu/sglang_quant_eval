#!/usr/bin/env python3
"""Parse saved Ascend profiling data in a normal process, without model inference."""

import argparse
import os
from pathlib import Path


def trace_files(capture: Path) -> list[Path]:
    return sorted(
        path
        for path in capture.rglob("trace_view.json")
        if path.is_file() and path.stat().st_size > 0
    )


def analyze_traces(directory: Path) -> list[Path]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Profiling directory does not exist: {directory}")
    captures = (
        [directory]
        if directory.name.endswith("_ascend_pt")
        else sorted(path for path in directory.rglob("*_ascend_pt") if path.is_dir())
    )
    if not captures:
        raise RuntimeError(f"No raw *_ascend_pt captures found in {directory}")

    # Native analysis creates subprocesses; run this script outside SGLang workers.
    # Keep standalone analysis temporary files in the current working directory.
    tmpdir = os.environ.setdefault(
        "TMPDIR", str(Path.cwd() / "outputs_i2v_profile" / "analysis_tmp")
    )
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    from torch_npu.profiler.profiler import analyse

    traces = []
    for capture in captures:
        found = trace_files(capture)
        if not found:
            print(f"Parsing raw capture: {capture}", flush=True)
            analyse(str(capture), max_process_number=1, export_type="text")
            found = trace_files(capture)
        if not found:
            raise RuntimeError(
                f"Analysis produced no nonempty trace_view.json for {capture}; "
                "raw data is retained. Check the torch_npu analysis messages above."
            )
        traces.extend(found)
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Run's traces/ or a *_ascend_pt directory"
    )
    args = parser.parse_args()
    try:
        for path in analyze_traces(args.directory):
            print(f"NPU trace: {path}")
    except (ImportError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
