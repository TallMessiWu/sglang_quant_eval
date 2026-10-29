---
name: gitmoji-commit
description: Review and stage the authorized changes, create an English Gitmoji-code commit, push normally, and verify remote SHA; ask before rewriting published history.
---

# Commit and publish

1. Verify repository, branch, remote, HEAD, status and staged/unstaged diff. Follow root `AGENTS.md` worktree and submodule rules. Do not stage unrelated user edits or gitlinks.
2. Run focused validation; state hardware or dependency limits honestly. Stage explicit paths for the authorized task. If unrelated changes are already staged, stop rather than commit them accidentally.
3. Review `git diff --cached` and `git diff --cached --check`.
4. Use an **English** title with a Gitmoji **text code**, never a Unicode emoji:
   `<code> <type>(<scope>): <imperative subject>`.
   Keep it concise; add a body only for important non-obvious context.

| Intent | Code / type |
| --- | --- |
| Feature | `:sparkles: feat` |
| Fix | `:bug: fix` |
| Refactor | `:recycle: refactor` |
| Remove obsolete files | `:fire: chore` |
| Documentation | `:memo: docs` |
| Tests | `:white_check_mark: test` |
| Configuration | `:wrench: chore` |

5. Show the proposed commit command, commit, then **push normally without asking again** when publication is authorized. Do not stop at an unpushed commit.
6. Verify local SHA equals remote branch SHA and ahead/behind is `0 0`; report both the branch and SHA.

Ask before `--force` / `--force-with-lease`, changing publication targets, or including uncertain user changes. Do not amend already-published history without authorization. Respect explicit requests not to commit/push. Normal cleanup is not permission to rewrite active feature branches.
