#!/usr/bin/env bash
# Apply the ten vendored patches to SGLang inside the container.
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
  "0004-autoround-w4a8.patch"
  "0005-add-dflash2-runtime-support.patch"
  "0006-support-quantized-target-lm-head-for-dflash2.patch"
  "0007-mamba-alloc-req-slots-demand.patch"
  "0008-mamba-coverage-thinning.patch"
  "0009-mamba-protect-reused-states.patch"
  "0010-fix-mamba-demand-v0518-request-fields.patch"
)

# Markers for 0001..0003, chosen to survive later patches: these touch
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
# 0004 touches separate files. Check the entire patch, including CUDA sources.
marker_0004() {
  git -C "$SGLANG_DIR" apply --reverse --check "$HERE/${PATCHES[3]}" >/dev/null 2>&1
}

# Later patches rewrite earlier context. Reverse the installed suffix in a
# private copy and validate the full series, including added test files.
snapshot_files() {
  local dest="$1" path
  shift
  while IFS= read -r path; do
    if [ -f "$SGLANG_DIR/$path" ]; then
      mkdir -p "$dest/$(dirname "$path")"
      cp "$SGLANG_DIR/$path" "$dest/$path"
    fi
  done < <(git apply --numstat "$@" | cut -f3 | sort -u)
}

detect_suffix_state() (
  local scratch count i valid
  local suffix=("${PATCHES[@]:4}")
  scratch="$(mktemp -d)"
  trap 'rm -rf "$scratch"' EXIT
  local paths=()
  for name in "${suffix[@]}"; do paths+=("$HERE/$name"); done
  mkdir "$scratch/snapshot"
  snapshot_files "$scratch/snapshot" "${paths[@]}"
  for ((count=${#suffix[@]}; count>=0; count--)); do
    cp -a "$scratch/snapshot" "$scratch/candidate-$count"
    valid=1
    for ((i=count-1; i>=0; i--)); do
      git -C "$scratch/candidate-$count" apply --reverse "${paths[$i]}" >/dev/null 2>&1 || { valid=0; break; }
    done
    [ "$valid" -eq 1 ] || continue
    # Reapply everything from the recovered base to reject missing, modified
    # or out-of-order patches before touching the installed source tree.
    for name in "${paths[@]}"; do
      git -C "$scratch/candidate-$count" apply "$name" >/dev/null 2>&1 || { valid=0; break; }
    done
    if [ "$valid" -eq 1 ]; then
      echo "$count"
      exit 0
    fi
  done
  exit 1
)
marker_0005() { [ "$suffix_state" -ge 1 ]; }
marker_0006() { [ "$suffix_state" -ge 2 ]; }
marker_0007() { [ "$suffix_state" -ge 3 ]; }
marker_0008() { [ "$suffix_state" -ge 4 ]; }
marker_0009() { [ "$suffix_state" -ge 5 ]; }
marker_0010() { [ "$suffix_state" -eq 6 ]; }

state() {
  local n=0
  marker_0001 && n=$((n + 1))
  marker_0002 && n=$((n + 1))
  marker_0003 && n=$((n + 1))
  marker_0004 && n=$((n + 1))
  marker_0005 && n=$((n + 1))
  marker_0006 && n=$((n + 1))
  marker_0007 && n=$((n + 1))
  marker_0008 && n=$((n + 1))
  marker_0009 && n=$((n + 1))
  marker_0010 && n=$((n + 1))
  echo "$n"
}

[ -d "$SGLANG_DIR/python/sglang" ] || {
  echo "[club-3090] no SGLang source at $SGLANG_DIR" >&2; exit 1; }

if ! suffix_state="$(detect_suffix_state)"; then
  echo "[club-3090] DFlash/Mamba suffix is partial, modified, or DOES NOT APPLY; expected v0.5.18." >&2
  echo "[club-3090] recreate the container to get a clean image tree." >&2
  exit 1
fi

if [ "${1:-}" = "--verify" ]; then
  for m in 1 2 3 4 5 6 7 8 9 10; do
    printf '%-45s ' "${PATCHES[$((m - 1))]}"
    "$(printf 'marker_%04d' "$m")" && echo "applied" || echo "NOT applied"
  done
  [ "$(state)" = "10" ] || { echo "incomplete"; exit 1; }
  echo "all ten applied"
  exit 0
fi

start_index=0
case "$(state)" in
  10)
    echo "[club-3090] patches already applied — nothing to do"
    exit 0
    ;;
  0) ;;
  *)
    # Upgrade containers with the complete old stack without reapplying it.
    if marker_0001 && marker_0002 && marker_0003; then
      start_index=3
      if marker_0004; then
        start_index=$((4 + suffix_state))
      elif [ "$suffix_state" -ne 0 ]; then
        echo "[club-3090] DFlash/Mamba installed without intact 0004 — refusing." >&2
        exit 1
      fi
    else
      # An incomplete scheduling stack cannot be safely reapplied.
      echo "[club-3090] SGLang at $SGLANG_DIR is PARTIALLY patched — refusing." >&2
      bash "$0" --verify >&2 || true
      echo "[club-3090] recreate the container to get a clean image tree." >&2
      exit 1
    fi
    ;;
esac

# Preflight every remaining patch in order before mutating any installed file.
# A bad later patch must not leave an old working stack partially upgraded.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
pending=()
for name in "${PATCHES[@]:$start_index}"; do pending+=("$HERE/$name"); done
snapshot_files "$scratch" "${pending[@]}"
for patch in "${pending[@]}"; do
  if ! git -C "$scratch" apply "$patch"; then
    echo "[club-3090] $(basename "$patch") — DOES NOT APPLY; installed files unchanged." >&2
    exit 1
  fi
done

cd "$SGLANG_DIR"
for name in "${PATCHES[@]:$start_index}"; do
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

suffix_state="$(detect_suffix_state)"
[ "$(state)" = "10" ] || {
  echo "[club-3090] patches applied but a marker is missing" >&2; exit 1; }
echo "[club-3090] patches OK — scheduling, logging, AutoRound W4A8, DFlash2 and Mamba cache fixes installed"
