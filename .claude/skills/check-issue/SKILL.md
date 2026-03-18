---
name: check-issue
description: Check the latest status of SGLang GitHub issues and PRs related to our MXFP8/Ascend work
---

# Check SGLang Issue Status

Fetch the latest status of key issues and PRs related to our work:

## Core Issues to Check
1. **#14424** - [Roadmap] Ascend NPU Quantization Refactoring (main tracking issue)
2. **#17093** - [Feature] Add MXFP8 quantization support (GPU side)
3. **#18258** - [Feature] ModelOpt Loader MXFP8 support
4. **#15194** - [Roadmap] Quantization Modifications (broader roadmap)

## How to Check
Use `gh` CLI commands:
```bash
gh issue view <number> --repo sgl-project/sglang
gh issue view <number> --repo sgl-project/sglang --comments
```

For PRs:
```bash
gh pr list --repo sgl-project/sglang --search "mxfp8" --state all
gh pr list --repo sgl-project/sglang --search "ascend quantization" --state all
```

## What to Report
- Any new comments or status changes
- New PRs related to MXFP8 or Ascend quantization
- Whether YChange01 has submitted any MXFP8 PR yet
- Any API changes that affect our implementation plan

Present a concise status update with dates and links.
