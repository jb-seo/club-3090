#!/usr/bin/env bash
# Apply the three vendored patches to SGLang inside the container.
#
# Run from the compose's command, before launch_server. Idempotent: a restart
# re-runs it and detects the already-applied state instead of failing.
#
#   bash install.sh            # apply (or confirm already applied)
#   bash install.sh --verify   # report state, change nothing
#
# Base is v0.5.18. These are context diffs, not rewrites -- against a
# different base they either refuse to apply (the good case) or apply into a
# tree whose other modules moved, which surfaces at scheduler start as an
# ImportError naming something unrelated. Keep the image pinned.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
PATCHES=(
  "0001-concurrent-chunked-prefill.patch"
  "0002-attention-backend-logging.patch"
  "0003-decode-passes-per-prefill.patch"
)

# One marker per patch, chosen to survive the later patches: all three touch
# the same three files, so `git apply --reverse --check` on an earlier patch
# fails once a later one has rewritten its context. Per-patch reverse checks
# read that as "not applied" and the script then reapplies into a patched
# tree. Markers do not have that problem.
markers_scheduling_flag() {
  grep -q "    $1: A\[" "$SGLANG_DIR/python/sglang/srt/server_args.py"
}
marker_0001() { markers_scheduling_flag long_prefill_token_threshold; }
marker_0002() {
  grep -q "Hybrid GDN model: full attention on prefill" \
    "$SGLANG_DIR/python/sglang/srt/layers/attention/attention_registry.py"
}
marker_0003() { markers_scheduling_flag decode_passes_per_prefill; }

state() {
  local n=0
  marker_0001 && n=$((n + 1))
  marker_0002 && n=$((n + 1))
  marker_0003 && n=$((n + 1))
  echo "$n"
}

[ -d "$SGLANG_DIR/python/sglang" ] || {
  echo "[club-3090] no SGLang source at $SGLANG_DIR" >&2; exit 1; }

if [ "${1:-}" = "--verify" ]; then
  for m in 1 2 3; do
    printf '%-45s ' "${PATCHES[$((m - 1))]}"
    "marker_000$m" && echo "applied" || echo "NOT applied"
  done
  [ "$(state)" = "3" ] || { echo "incomplete"; exit 1; }
  echo "all three applied"
  exit 0
fi

case "$(state)" in
  3)
    echo "[club-3090] patches already applied — nothing to do"
    exit 0
    ;;
  0) ;;
  *)
    # Some markers present, not all. Reapplying would fail mid-stack and
    # leave a worse tree than either end state, so refuse.
    echo "[club-3090] SGLang at $SGLANG_DIR is PARTIALLY patched — refusing." >&2
    bash "$0" --verify >&2 || true
    echo "[club-3090] recreate the container to get a clean image tree." >&2
    exit 1
    ;;
esac

cd "$SGLANG_DIR"
for name in "${PATCHES[@]}"; do
  if git apply --check "$HERE/$name" >/dev/null 2>&1; then
    git apply "$HERE/$name"
    echo "[club-3090] $name — applied"
  else
    echo "[club-3090] $name — DOES NOT APPLY to $SGLANG_DIR" >&2
    echo "[club-3090] expected SGLang v0.5.18; pin the image tag." >&2
    git -C "$SGLANG_DIR" log --oneline -1 2>/dev/null >&2 || true
    exit 1
  fi
done

[ "$(state)" = "3" ] || {
  echo "[club-3090] patches applied but a marker is missing" >&2; exit 1; }
echo "[club-3090] patches OK — both scheduling flags declared"
