#!/usr/bin/env bash
# Use the activated service environment; never switch/reset an existing checkout.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 linear|boundary|capabilities|wan-baseline|wan-g22|wan-g23 SGLANG_CHECKOUT [probe arguments]" >&2
    exit 2
fi
mode=$1
repo=$(git -C "$2" rev-parse --show-toplevel)
shift 2
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
baseline=62ff466ac0c2cdd5521acf7d4c673fa7630eb083
g22=6eb8e1dd1c91678284d1355cdb182a85ab4e9863
g23=d6a46f90d97039e14476be47468c996f11caf603
case "$mode" in
    linear|boundary|wan-baseline|wan-g22)
        head=$g22; branch=junlin/diffusion-mxfp8-shared-execution ;;
    capabilities|wan-g23)
        head=$g23; branch=junlin/ascend-mxfp-capabilities ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

evidence=$(mktemp -d "${TMPDIR:-/tmp}/g2-${mode}-XXXXXXXX")
exec > >(tee "$evidence/run.log") 2>&1
echo "Evidence: $evidence"
"$python_bin" -c 'import sys; print("Python:", sys.executable)'
# Fetch the known fork without altering the checkout's configured remotes/branches.
git -C "$repo" fetch https://github.com/TallMessiWu/sglang.git "refs/heads/$branch"
fetched=$(git -C "$repo" rev-parse FETCH_HEAD)
if [[ "$fetched" != "$head" ]]; then
    echo "Remote head changed to $fetched; review the new revision before changing the acceptance pin." >&2
    exit 1
fi

prepare() {
    "$python_bin" - "$repo" "$root" "$1" "$2" <<'PY'
from pathlib import Path
import subprocess
import sys

repo, root, sha, name = sys.argv[1:]
parent = (Path(root) / "sglang").resolve()
listing = subprocess.check_output(["git", "-C", repo, "worktree", "list", "--porcelain"], text=True)
for record in listing.strip().split("\n\n"):
    fields = dict(line.split(" ", 1) for line in record.splitlines() if " " in line)
    path = Path(fields["worktree"]).resolve()
    if fields.get("HEAD") == sha and path.parent == parent:
        print(path)
        break
else:
    path = parent / name
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}; inspect its ownership/HEAD first")
    subprocess.run(["git", "-C", repo, "worktree", "add", "--detach", str(path), sha], check=True, stdout=sys.stderr)
    print(path)
PY
}

source=$(prepare "$head" "g2_validation_${mode//-/_}")
extra=()
case "$mode" in
    linear)
        base_source=$(prepare "$baseline" g2_validation_baseline)
        action=linear; extra=(--baseline-source "$base_source") ;;
    boundary) action=linear; extra=(--boundary) ;;
    capabilities) action=capabilities ;;
    wan-baseline)
        source=$(prepare "$baseline" g2_validation_baseline)
        head=$baseline; action=wan ;;
    wan-g22|wan-g23) action=wan ;;
esac
echo "Source: $source @ $head"
git -C "$repo" worktree list --porcelain
"$python_bin" -u "$root/validation/g2_npu.py" "$action" \
    --source "$source" --expected-head "$head" --output-dir "$evidence/result" \
    "${extra[@]}" "$@"
