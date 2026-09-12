# Qwen3.8-27B on SGLang — experimental, dual 3090

One compose: [`compose/dual/autoround-int4/mtp.yml`](compose/dual/autoround-int4/mtp.yml).
Boots, serves, and is fast enough to use. It is **not** a production path yet,
for reasons that are open questions rather than known defects — see below.

Needs thirteen vendored patches, applied in-container at startup:
[`patches/sglang-v0518-scheduling-and-logging/`](patches/sglang-v0518-scheduling-and-logging/README.md).
The image is pinned to `lmsysorg/sglang:v0.5.18` because that is what they
are cut against — `:latest` is already `v0.5.19`.

## Run inside an existing RunPod container

Use `lmsysorg/sglang:v0.5.18` with a Bash entrypoint. The launcher clones
this fork's `qwen3.8-27b-sglang-mtp` branch into `/workspace/club-3090`, applies
the thirteen patches to the image's `/sgl-workspace/sglang` source and runs
`python3 -m sglang.launch_server` in the foreground. It reads the arguments
and environment from `compose/dual/autoround-int4/mtp.yml` so list-valued
flags and future compose edits stay in sync.

Once the launcher files have been pushed to that branch:

```bash
curl -fsSL https://raw.githubusercontent.com/jb-seo/club-3090/qwen3.8-27b-sglang-mtp/models/qwen3.8-27b/sglang/scripts/run_in_runpot.sh | bash
```

The default model is the profile's AutoRound INT4 HF repository, currently
`Frozenlock/Qwen3.8-27B-int4-AutoRound`; SGLang downloads missing files using
`HF_HOME=/workspace/cache/huggingface`. An existing `/models/target/config.json`
or `/workspace/models/qwen3.8-27b-autoround-int4/config.json` takes precedence.
Set `MODEL_PATH` to use another local directory or HF model ID. JIT caches
use `/workspace/cache/sglang`.

HiCache is enabled with the stock image's built-in `file` backend. Because
Qwen3.8 uses one host/file pool per TP rank, the compose assigns **1 GB L2 RAM
and 10 GB L3 per rank**: TP=2 gives about 2 GB of host cache and a strict
20 GB disk-cache ceiling for the server. The `write_back` policy begins
GPU→L2→L3 backup when an unlocked radix leaf is selected for GPU-cache
eviction, rather than copying newly inserted cache entries immediately. L3
defaults to `/workspace/cache/sglang/hicache-file` on RunPod and to the compose's
`SGLANG_CACHE_DIR/hicache-file` on Docker. Point `WORKSPACE_DIR` at your
persistent volume if it is mounted elsewhere; existing `HF_HOME`,
`SGLANG_CACHE_DIR` and `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` win.

`SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE` is a **per-rank** override, so keep it
at `10G` for a 20 GB TP=2 total. The v0.5.18 size parser accepts `10G` (decimal)
and `10Gi` (binary), but not `10GB`. Changing `TP_SIZE` also changes the total
RAM and disk budget unless `--hicache-size` and this per-rank cap are adjusted.

Defaults retain the compose's TP=2, PCIe communication settings, built-in
EAGLE/MTP drafter and HTTP port **30000**. Expose that port in the Pod template;
there is no Docker port mapping from 8042 inside this container. GPU access
and the compose's 16 GiB shared-memory requirement belong to the container
configuration. The script checks the installed SGLang version and visible
GPU count before applying patches.

```bash
# Inspect the command first (no patch installation, weights or server).
curl -fsSL https://raw.githubusercontent.com/jb-seo/club-3090/qwen3.8-27b-sglang-mtp/models/qwen3.8-27b/sglang/scripts/run_in_runpot.sh | bash -s -- --dry-run

# Reuse the checkout with explicit model/port overrides and extra server flags.
MODEL_PATH=/workspace/models/my-autoround-model PORT=8042 \
  bash /workspace/club-3090/models/qwen3.8-27b/sglang/scripts/run_in_runpot.sh \
  --max-running-requests=4
```

For a pipeline, export overrides first or use `| env PORT=8042 bash`;
`PORT=8042 curl ... | bash` only sets the variable on curl. `TP_SIZE` can
override TP explicitly. Extra SGLang arguments are appended to the compose
arguments. `HF_TOKEN` is passed through without being printed.

Reruns reuse the checkout and the idempotent patch installer. The launcher
does not pull or reset an existing checkout: update it deliberately, or set
`CLUB3090_DIR` to a new directory. `CLUB3090_REPO` and `CLUB3090_REF` select
another clone source; `--help` lists the other settings.

Patch `0012` makes the insertion cap preserve replay coverage while retaining
the newest frontier. Append `--mamba-max-states-per-path=3` to test three
checkpoints per path; the compose currently sets two. Patch `0011` still
spreads global eviction across paths above its soft floor of two. See the
[policy and validation notes](patches/sglang-v0518-scheduling-and-logging/README.md#coverage-aware-insertion-cap-0012-2026-09-11).

Patch `0013` writes per-process KV/Mamba cache history and 30-second tree/pool
snapshots under `/tmp/sglang_kv_mamba_history`. Files are JSONL, rotate at
32 MiB with four backups, and include TP/PP rank in their names. Set
`SGLANG_CACHE_HISTORY_DIR=off` to disable it; the patch README documents the
other size, interval and CUDA-slot controls.

Validation: seven CPU-only launcher tests cover compose argument parity,
overrides, model selection, version/GPU checks, piped clone/reuse and refusing
to start after patch installation fails. No RunPod GPU model load or serving
benchmark has been run for this launcher.

## How this was arrived at

Empirically, not analytically. This subtree is one rig's trial-and-error
written down, not a config derived from understanding how SGLang schedules
hybrid-GDN models. **Do not read the flag values as recommendations.**

The distinction that matters when reading the rest of this page:

| Measured | Guessed or unknown |
|---|---|
| Decode TPS, accept-len | Why the prefix cache drops whole sessions |
| Which backend serves which layers (the engine logs it) | Where the ~2x wall-time gap against vLLM lives |
| Image preprocessing lands on GPU 0 and OOMs there | Whether any flag value here is near optimal |
| The patches do not apply to v0.5.19 | |

The numbers in "Measured" came off this rig and are not cross-validated
anywhere else.

### What was tried and abandoned

Listed so the next person does not re-run them. Each was believed at the time:

- **Mamba checkpoint eviction as the cause of the cache wipeouts.** The
  longest-held theory. `--max-mamba-cache-size` is 64 here because it was
  raised from 20 chasing it; that change did **nothing**, and the value is
  kept only because nothing since has argued for a different one. The theory
  died when the same workload reproduced the same wipeouts on vLLM.
- **A fixed ~260-320 token loss per turn read as a cache truncating to some
  internal granularity**, and matched against `mamba_track_interval=256`. It
  was a bug in the benchmark client, and the 256 was a coincidence with the
  client's own `--gen-tokens`. Three separate client bugs were found this way;
  each produced losses that looked exactly like cache behaviour.
- **LRU eviction** and **request retraction** as causes of the wipeouts. Both
  ruled out by observation, not argument — see "Open questions" below.
- **A wall-clock-driven decode share.** Worked on one rank and deadlocked TP:
  the accumulators are rank-local, they drift, and once the drift straddles
  the threshold the ranks take different branches and NCCL hangs. That is why
  `--decode-passes-per-prefill` counts passes.

The general lesson, and the reason the table above exists: a measurement bug
reproduces identically on two engines; an engine bug does not. Anything here
that has not been checked that way is a hypothesis.

## This is not the 3.6 story

[`models/qwen3.6-27b/sglang/`](../../qwen3.6-27b/sglang/README.md) is PARKED
for three reasons. Two do not apply here:

| 3.6 parking reason | Here |
|---|---|
| EAGLE-3 external drafter is slower than native MTP | N/A — this uses the **built-in MTP head**, no external drafter. That page's own numbers put native MTP ahead (121 vs 111 TPS). |
| CUTE_DSL cuda-graph capture hangs on Ampere (v0.5.12) | Does not reproduce on v0.5.18. Graphs capture and replay; decode logs `cuda graph: True`. |
| SGLang 15–18 TPS vs vLLM-MTP ~85 TPS | That 15–18 was measured with `--disable-cuda-graph` forced by the hang above. With graphs working, decode here is **~80–104 tok/s** at bs=1. |

So the 3.6 verdict does not transfer by inheritance. What does carry over is
narrower and is stated honestly below: vLLM still finishes the same
multi-session benchmark in about **half** the wall time.

## Measured on this rig

2× RTX 3090 (sm_86), TP=2, AutoRound INT4 W4A16, KV fp8_e4m3.

| | |
|---|---|
| Decode, bs=1 | ~80–104 tok/s |
| MTP accept-len | 2.3–3.4 (accept-rate 0.44–0.80) |
| Full attention | flashinfer, 16 layers |
| GDN linear attention | Triton (`TritonGDNKernel`), the remaining layers |
| KV pool | ~440K tokens — under THIS compose. It moved with `--mem-fraction-static` and the other memory knobs during development; it is a property of the settings, not of the card. |
| Context exercised | 100K+ |

Not measured: TTFT curve, quality packs, soak, multi-card. No `BENCHMARKS.md`
row is claimed.

## Open questions

These are why the status is 🧪 and not ⚠️. They are unresolved, not
characterised-and-accepted.

### 1. ~2× total wall time vs vLLM, and it is not decode

On a multi-session benchmark (10 sessions, growing contexts, compaction,
concurrency 4) SGLang takes roughly twice vLLM's total time on the same box
and the same workload. Decode TPS is at parity or better, so the gap is in
prefill.

A plausible shape, unverified: most layers of this model are **GDN**, and GDN
runs Triton here. `--attention-backend` only moves the 16 full-attention
layers. flashinfer's GDN kernels are SM90/SM100/SM103 only, so on sm_86 there
is no alternative to select. If that is where the gap lives it is not
tunable — but it has not been localised yet.

To localise it, run the client at both extremes and compare totals per engine:

```bash
# prefill-dominated
python3 tools/probe_radix.py --url <engine> --mode simulate \
    --sessions 3 --turns 30 --seed 0 --concurrency 1 --gen-tokens 1
# decode-dominated
python3 tools/probe_radix.py --url <engine> --mode simulate \
    --sessions 3 --turns 30 --seed 0 --concurrency 1 --gen-tokens 2048
```

(`--concurrency 1` takes scheduling out of it, leaving kernel speed. The
client used here is not vendored in this repo; any harness that reports
`usage.prompt_tokens_details.cached_tokens` and drives both engines the same
way will do.)

### 2. Prefix-cache wipeouts, cause unknown

In multi-session use a session that has been idle while others grew sometimes
comes back to **0%** prefix reuse — the whole prefix, not a truncated tail.
Ruled out, each by measurement rather than argument:

- **LRU eviction** — happens with plenty of KV available, early in a run.
- **Retraction** — `Retract requests` never appears in the log. (Retraction
  would explain it: the retract path releases KV with `is_insert=False`, so a
  preempted request's KV is dropped rather than cached.)
- **Mamba checkpoint eviction** — the hypothesis this subtree spent the most
  time on. `--max-mamba-cache-size` 20 → 64 changed nothing, and the same
  workload reproduces the same result **on vLLM**, whose GDN state caching is
  a different implementation.

That last point is the useful one: whatever this is, it is not SGLang-specific.
A measurement bug reproduces identically on two engines; an engine bug does
not. Three bugs in the benchmark client were found this way and fixed — the
numbers here are post-fix, and the wipeouts survived all three.

### 3. Multimodal preprocessing lands on GPU 0

The fast (CUDA) image processor runs on `cuda:{base_gpu_id}` — always GPU 0,
alongside TP rank 0's weights. It OOMs there while GPU 1 still has ~700 MB
free; the asymmetry is the tell. The compose sets
`--image-processor-backend pil` to keep preprocessing off the GPU. Fine for a
text-heavy workload; revisit before claiming the vision path.

## Knobs worth knowing

| Flag | Value here | Note |
|---|---|---|
| `--attention-backend` | `flashinfer` | Only affects the 16 full-attention layers. GDN is Triton regardless. |
| `--page-size` | `64` | Required by `extra_buffer` (FLA chunk size 64). Moves together with it. |
| `--mamba-radix-cache-strategy` | `extra_buffer` | `no_buffer` requires `page-size 1` and disables overlap scheduling. |
| `--long-prefill-token-threshold` | `2048` | Patched flag. Lets several requests be mid-prefill at once. |
| `--decode-passes-per-prefill` | `50` | Patched flag. Read off this rig's logs, not tuned. The server logs the split it achieved — tune against that. |
| `--max-mamba-cache-size` | `64` | Raised from 20 while chasing a theory that did not hold. No evidence it is the right value; no evidence it is wrong either. |
| `--chunked-prefill-size` | `4096` | Sets how long a single prefill pass blocks decode. The model's cookbook page suggests 2048; smaller chunks also mean more state checkpoints per request. |
| `--image-processor-backend` | `pil` | See open question 3. |
| `--disable-custom-all-reduce` | set | PCIe-only Ampere, no NVLink. |
| `--max-running-requests` | `1` | Development ran at 4 against a 440K-token KV pool. At full context that pool does not hold two sessions, so 4 was a testing value this hardware cannot ship. See below — it makes the two scheduling flags inert. |

## The two scheduling flags do nothing at the shipped concurrency

Worth stating plainly, because the compose otherwise reads as
self-contradictory: it vendors three patches for two flags and then sets a
concurrency at which neither flag can act.

Both divide a scheduling pass between concurrent requests. At
`--max-running-requests 1` there is never a second request to divide with —
nothing else is mid-prefill for `--long-prefill-token-threshold` to cap, and
the running batch is empty while a prefill runs, so
`--decode-passes-per-prefill` returns before deciding anything.

They are set anyway for two reasons: raising the concurrency is the first
thing anyone will try, and the flags are what the patches exist for. Whether
they help is the open question the development was about; at concurrency 1 it
is not being asked.

Raising it is bounded by KV, not by the flags. 440K tokens does not hold two
full-context sessions, which is the whole reason the shipped value is 1.

## Not tried

- Single-card (`single/`): 27B INT4 is ~18 GiB of weights against 24 GiB;
  no attempt made.
- `--enable-mixed-chunk`: refuses to coexist with speculative decoding, which
  is the whole point of this compose. It would make
  `--decode-passes-per-prefill` unnecessary — that trade is the interesting
  one if MTP is ever dropped.
- FP8 / NVFP4 weights: the sibling vLLM composes have those tiers; this one
  only has AutoRound INT4.

## Cross-links

- vLLM on this model: [`../vllm/compose/dual/autoround-int4/mtp.yml`](../vllm/compose/dual/autoround-int4/mtp.yml)
- SGLang on 3.6 (parked): [`../../qwen3.6-27b/sglang/README.md`](../../qwen3.6-27b/sglang/README.md)
- Upstream PR behind `--long-prefill-token-threshold`: [sgl-project/sglang#34623](https://github.com/sgl-project/sglang/pull/34623)
