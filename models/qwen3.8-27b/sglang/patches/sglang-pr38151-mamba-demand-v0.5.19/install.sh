#!/usr/bin/env bash
# Install alphabetc1's Mamba eviction-demand fix on the v0.5.19 source tree.
# Provenance and retirement condition: docs/UPSTREAM.md, SGLang #38151.
# Usage: install.sh [--verify]; --verify never changes the source tree.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
PATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/pr38151-v0.5.19.patch"
TARGET="$SGLANG_DIR/python/sglang/srt/mem_cache/allocation.py"

die() { echo "[mamba-demand] ERROR: $*" >&2; exit 1; }

case "$#:${1:-}" in
  0:|1:--verify) ;;
  *) die "Usage: install.sh [--verify]" ;;
esac
[ -f "$TARGET" ] || die "allocation.py missing; set SGLANG_DIR to the SGLang source root"
[ -f "$PATCH" ] || die "patch missing at $PATCH; check the compose mount"

# git apply also works on a source tree without .git. Minimal images may only
# provide patch(1); disable fuzz and prompts there and preflight every hunk.
if command -v git >/dev/null 2>&1; then
  check_applied() { git -C "$SGLANG_DIR" apply --reverse --check "$PATCH"; }
  check_pending() { git -C "$SGLANG_DIR" apply --check "$PATCH"; }
  apply_patch() { git -C "$SGLANG_DIR" apply "$PATCH"; }
elif command -v patch >/dev/null 2>&1; then
  check_applied() { patch --batch --force --fuzz=0 --dry-run -R -p1 -d "$SGLANG_DIR" < "$PATCH"; }
  check_pending() { patch --batch --forward --fuzz=0 --dry-run -p1 -d "$SGLANG_DIR" < "$PATCH"; }
  apply_patch() { patch --batch --forward --fuzz=0 --no-backup-if-mismatch -p1 -d "$SGLANG_DIR" < "$PATCH"; }
else
  die "git or patch is required; use the pinned SGLang image or install one in your custom image"
fi

if check_applied >/dev/null 2>&1; then
  echo "[mamba-demand] already applied + verified"
  exit 0
fi
[ "${1:-}" != "--verify" ] || die "patch is not fully applied or source context has drifted"
check_pending || die "patch does not match this tree; recreate with the pinned v0.5.19 image and retry"
echo "[mamba-demand] applying v0.5.19 patch..."
apply_patch || die "patch application failed"
check_applied || die "post-apply verification failed"
echo "[mamba-demand] applied + verified OK"
