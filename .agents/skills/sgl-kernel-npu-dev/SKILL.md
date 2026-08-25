---
name: sgl-kernel-npu-dev
description: Develop, review, build, validate, or diagnose this repository's sgl-kernel-npu integration. Use for sgl-kernel-npu remotes and submodule/gitlink state, open PR and branch mapping, 910/950 target wheels, build.sh target selection, Gemma RMSNorm provider staging, paired SGLang changes, artifact inspection, or NPU validation planning.
---

# sgl-kernel-npu development

Treat the local fork checkout, official upstream, target-specific wheel, and paired SGLang worktree as separate layers. Verify each layer before changing another.

## Start with local state

Run the bundled read-only audit:

```powershell
powershell -ExecutionPolicy Bypass -File .agents/skills/sgl-kernel-npu-dev/scripts/check-local-state.ps1
```

Use its output to confirm the kernel branch, HEAD, tracking ref, dirty state, remotes, parent gitlink, and all registered SGLang worktree paths. Do not infer branch identity from a directory name or the parent gitlink.

## Classify the task

1. **Status, PR, or review request**: keep the operation read-only; query GitHub live and compare the PR head SHA with local HEAD.
2. **Kernel-only change**: work inside `sgl-kernel-npu/`; preserve its branch and unrelated changes.
3. **Paired SGLang change**: use the PR head branch's dedicated `sglang/<name>/` worktree. Every modified SGLang branch must live directly under `sglang/`; never edit an external temporary worktree.
4. **Build or provider change**: read [references/target-wheel.md](references/target-wheel.md) before editing or claiming validation.
5. **Parent repository update**: update the submodule gitlink only when the user intends to record the kernel commit here. A kernel checkout change alone does not authorize a parent gitlink update.

## Preserve the remote model

Expected remotes:

```text
origin   https://github.com/TallMessiWu/sgl-kernel-npu.git
upstream https://github.com/sgl-project/sgl-kernel-npu.git
```

- Fetch official changes from `upstream` and push feature branches to `origin`.
- Keep `.gitmodules` pointed at the development fork so a fresh submodule checkout gets the intended `origin`.
- Re-check both URLs before fetch, rebase, push, or PR work.

## Keep provider selection at build time

For Gemma RMSNorm and similar SoC-dependent Python providers:

- expose one stable public import path and call signature to SGLang;
- select the 910 or 950 source only while staging the wheel;
- verify `build/lib`, wheel contents, and installed `site-packages` each contain only the selected provider;
- keep logical wheel target separate from concrete CMake/AscendC compatibility target;
- do not add runtime SoC queries, import probes, `is_npu_a5()` helpers, or dual-provider branches to SGLang.

The source tree may contain private templates. Acceptance is based on staged and installed artifacts, not source-tree coexistence.

## Build narrowly

```bash
cd sgl-kernel-npu
bash build.sh -h
bash build.sh -a kernels 910   # A2/A3 wheel
bash build.sh -a kernels 950   # A5 wheel
```

Use auto-detection only when testing detection itself. For A5 release/validation, pass `950` explicitly and inspect the logged wheel target. Do not build unrelated DeepEP, attentions, or memory-saver modules for a kernel-wheel-only change.

## Validate in layers

1. **Repository**: branch, remotes, diff, user changes, parent gitlink.
2. **Static**: focused tests, formatting, compile checks, shell syntax, `git diff --check`.
3. **Staging**: selected provider exists in `build/lib`; opposite provider is absent.
4. **Wheel**: file list and contents match staging.
5. **Install**: import and inspect from clean `site-packages`, not the source checkout.
6. **Hardware**: 950 correctness and Qwen3.5 e2e for A5; 910B/910C native-provider smoke.

Report each layer separately. Static success is not wheel proof, and wheel proof is not NPU e2e proof.

## Paired SGLang integration

- Keep the public name, arguments, return contract, dtype semantics, and error behavior explicit.
- Search the exact API in both repositories.
- Modify SGLang only in the matching `sglang/<worktree>/`.
- Make old/incompatible package errors actionable; do not silently switch provenance.
- Record paired PRs in `docs/branches.md` after live GitHub verification.

For the current Gemma RMSNorm design, read [references/target-wheel.md](references/target-wheel.md).

## Avoid stale documentation

- Put current PR, branch, HEAD, worktree, and remote snapshots in `docs/branches.md`.
- Put user-facing build instructions in `docs/sgl-kernel-npu-build.md`.
- Keep durable operating rules in `AGENTS.md` and this skill.
- Do not duplicate volatile hashes across multiple files.
