#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${1:-$(cd -- "$script_dir/../../../.." && pwd)}

printf '\n== Kernel checkout ==\n'
git -C "$repo_root/sgl-kernel-npu" status --short --branch
git -C "$repo_root/sgl-kernel-npu" rev-parse HEAD
git -C "$repo_root/sgl-kernel-npu" remote -v
printf '\n== Parent committed and staged gitlink ==\n'
git -C "$repo_root" ls-tree HEAD -- sgl-kernel-npu
git -C "$repo_root" ls-files --stage -- sgl-kernel-npu
printf '\n== Registered SGLang worktrees ==\n'
git -C "$repo_root/sglang/qwen3.5_dense_w8a8" worktree list --porcelain
