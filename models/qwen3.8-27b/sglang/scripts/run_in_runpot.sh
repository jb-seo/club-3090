#!/usr/bin/env bash
# Bootstrap inside an already-running lmsysorg/sglang:v0.5.18 container.
# Keep the body in a function so `curl ... | bash` consumes it before children run.
set -euo pipefail
export PYTHONUTF8=1

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Run the Qwen3.8 SGLang compose command inside an existing RunPod container.

Usage: bash run_in_runpot.sh [--dry-run] [SGLang arguments...]
       curl -fsSL <raw-script-url> | bash -s -- [--dry-run] [arguments...]

Environment:
  CLUB3090_REPO       Clone URL (default: https://github.com/jb-seo/club-3090.git)
  CLUB3090_REF        Branch/tag to clone (default: qwen3.8-27b-sglang-mtp)
  CLUB3090_DIR        Existing/new checkout (default: /workspace/club-3090)
  WORKSPACE_DIR      Persistent workspace (default: /workspace)
  MODEL_PATH         Local model directory or Hugging Face model ID
                     Default: existing /models/target, then MODEL_DIR/TARGET_DIR,
                     then the profile's AutoRound INT4 Hugging Face repository
  MODEL_DIR          Model parent directory (default: /workspace/models)
  TARGET_DIR         Model subdirectory (default: qwen3.8-27b-autoround-int4)
  TP_SIZE / PORT      Override compose TP=2 / port=30000
  SGLANG_DIR         Image source tree (default: /sgl-workspace/sglang)
  HF_HOME            Hugging Face cache (default: /workspace/cache/huggingface)
  SGLANG_CACHE_DIR   JIT cache and L3 parent (default: /workspace/cache/sglang)
  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR
                     L3 file cache (default: SGLANG_CACHE_DIR/hicache-file)
  SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE
                     Per-rank L3 cap (default: 10G; TP=2 total: 20 GB)

HF_TOKEN is honored by SGLang/Hugging Face. Export overrides before piping,
or put `env NAME=value` on the bash side of the pipe.
Existing checkouts are reused without pulling or resetting local changes.
--dry-run may clone the recipe repo; it does not patch, load weights or start a server.
Expose the selected HTTP port in the RunPod template. The server runs in foreground.
HELP
    return
  fi

  local workspace="${WORKSPACE_DIR:-/workspace}"
  local repo_dir="${CLUB3090_DIR:-$workspace/club-3090}"
  local repo_url="${CLUB3090_REPO:-https://github.com/jb-seo/club-3090.git}"
  local repo_ref="${CLUB3090_REF:-qwen3.8-27b-sglang-mtp}"
  local runner="models/qwen3.8-27b/sglang/scripts/run_in_runpot.py"
  command -v git >/dev/null || { echo 'Missing git; use the stock SGLang image or install git.' >&2; return 1; }
  command -v python3 >/dev/null || { echo 'Missing python3; use lmsysorg/sglang:v0.5.18.' >&2; return 1; }

  if [[ ! -e "$repo_dir" ]]; then
    mkdir -p "$(dirname "$repo_dir")"
    git clone --depth 1 --single-branch --branch "$repo_ref" "$repo_url" "$repo_dir" </dev/null
  fi
  if [[ ! -f "$repo_dir/$runner" ]]; then
    echo "Missing $repo_dir/$runner. Update that checkout to the branch containing this launcher, or set CLUB3090_DIR to a new path." >&2
    return 1
  fi
  export WORKSPACE_DIR="$workspace"
  echo "[runpod] Recipe checkout: $repo_dir ($(git -C "$repo_dir" rev-parse --short HEAD))"
  exec python3 "$repo_dir/$runner" "$@" </dev/null
}

main "$@"
