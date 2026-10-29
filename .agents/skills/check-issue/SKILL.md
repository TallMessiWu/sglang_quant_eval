---
name: check-issue
description: Check live GitHub issue, PR, review and CI status for this repository's SGLang Ascend quantization work, comparing remote heads with local worktrees.
---

# Check issues and PRs

Resolve the repository root from this directory (`../../..`). Read `docs/mxfp-refactoring-plan.md` for the active task/PR list and `docs/branches.md` for checkout mapping; do not maintain another hard-coded status list here. Load the web-access skill before network operations.

For the requested scope, use `gh pr view` / `gh issue view` with the explicit upstream repository. Check state, draft, head/base SHA, mergeability, labels, review threads and check runs. Compare headRefOid, actual fork branch tip and local HEAD; a closed PR may retain an old head.

For failed CI, inspect the failed job logs at that SHA. Distinguish missing labels/Draft/approval gates, dependency/import failures, test failures, cancellation and downstream finish failures. Skipped suites are not passes; lint success is not NPU validation. UNKNOWN mergeability is not proof of no conflicts.

Report date, links, concrete failure evidence, relation to the diff and remaining uncertainty. Update only the canonical plan/branch docs when requested. Reading status does not authorize changing labels, replying/resolving threads, triggering CI, reopening PRs or pushing code.
