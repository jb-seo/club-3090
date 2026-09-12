# Qwen3.8-27B — Changelog

Dated history for Qwen3.8-27B configs in this repo. Append-only — add a new entry, don't rewrite past ones.

## 2026-09-12 — Enable bounded L2/L3 HiCache for SGLang

Enable HiCache in the dual-3090 SGLang compose using v0.5.18's dependency-free
`file` backend. The TP=2 server allocates 1 GB of host cache per rank (about
2 GB total) and caps each rank's file cache at 10 GB (20 GB total), with
write-through from the small L2 staging pool. Docker persists L3 below the
existing SGLang cache mount; the RunPod launcher maps it to
`/workspace/cache/sglang/hicache-file` and exposes path/cap overrides.

CPU-only launcher tests cover compose parity, defaults and explicit HiCache
environment overrides. GPU startup, L3 hit latency and workload throughput
remain to be measured on the target rig.

## 2026-09-12 — Add persistent KV/Mamba cache history

Append `0013` after the existing twelve SGLang patches. It records correlated
KV leaf eviction, Mamba allocation/checkpoint/eviction, request cache
operations and 30-second radix-tree/pool snapshots as per-process JSONL under
`/tmp/sglang_kv_mamba_history`. Parent/depth fields preserve shared-path
structure; logs contain no token or state contents. Rotation defaults to
32 MiB with four backups, CUDA slot IDs stay disabled to avoid synchronization,
and environment variables control the path, interval, size and node bound.
I/O failure disables only the recorder. Runtime CPU cache tests and installer
fresh/upgrade/drift tests cover delivery; TP2 serving overhead remains to be
measured on the target rig.

## 2026-09-11 — Bound Mamba checkpoints by replay coverage

Append `0012` after the existing eleven patches. The per-path insertion cap
now selects old victims by whole-path maximum replay gap, then integer
quantile deviation and deterministic depth/ID ties. The newest frontier and
fork/lock/session/reuse/load-back/leaf protections survive; overflow remains
soft. The existing post-BackupKV action, Full KV and host copies, exact-demand
allocation and global fairness/thinning behavior are preserved. This release
has no Rust radix-tree backend. Pre-admission slot reuse is deferred, so a
new frontier may briefly require N+1 persistent slots.

The runtime suite passes 55 tests; five GPU backup tests skip because CUDA
cannot initialize in the local container. All 17 installer tests pass. A
24-turn CPU policy A/B at cap=3 reduces zero-hit probes from 65.8% to 17.0%
(4 sessions), 47.8% to 22.5% (8), and 33.0% to 20.8% (12), with the same
slot ceilings. High pressure still permits poor gaps and full-path loss;
these are synthetic cache-policy results, not GPU accuracy or TTFT claims.
Details and reproduction are in the patch README's validation report.

## 2026-09-11 — Do not backport superseded SGLang PR #32606

PR #32606 was evaluated for the v0.5.18 patch series and rejected after its
single commit conflicted at the exact guard changed by merged successors
#34043 and #34184. Both successors are already ancestors of v0.5.18: Mamba
tracking remains wired under speculative decoding and stale padded tracking
rows are cleared. Applying #32606 would undo the former and force affected
prefills eager. No runtime patch or installer entry was added; the upstream
tracker records the sources and removal decision.

## 2026-09-11 — Share SGLang Mamba eviction pressure across paths

Append local patch `0011`: each eviction sweep takes at most one checkpoint
per non-branching path, using the existing coverage/min-gap selector. Retain
two usable device checkpoints while other paths can contribute; when a full
sweep cannot progress, return to the true LRU tail and allow eviction below
the floor. Eviction does not refresh access recency. The exact-demand fix
and original patches `0001`–`0010` are unchanged. Upstream ancestry remains
tracked in [`docs/UPSTREAM.md`](../../docs/UPSTREAM.md#sglang-sgl-projectsglang).

The current compose's insertion cap of two is independent of this floor;
the patch README documents a launcher override to disable that cap for
fairness experiments. Seventeen new fairness tests include real partial
prefix matching and leaf eviction. Combined runtime checks: 39 passed, five
GPU tests skipped. All 14 installer tests pass, including in-place upgrades
from ten patches. GPU serving and workload cache-hit improvements have not
been measured for this policy.

## 2026-09-11 — Fix the SGLang Mamba backport's request-layout mismatch

The initial allocation-demand backport in `0007` used newer `req.kv` fields
that do not exist in v0.5.18. A fresh prefill crashed with
`AttributeError: 'NoneType' object has no attribute 'holds_mamba'`.
Append `0010` to use the same request-level fields as the release allocator;
existing nine-patch containers upgrade in place. This corrects the local
backport of the source tracked in
[`docs/UPSTREAM.md`](../../docs/UPSTREAM.md#sglang-sgl-projectsglang).

Six allocation tests with real release `Req` objects reproduce the failure
before the fix and pass afterward; nine thinning tests and 13 installer
tests also pass. GPU serving/performance have not been validated.

## 2026-09-11 — SGLang launcher for an existing RunPod container

Add `sglang/scripts/run_in_runpot.sh` for piped startup inside the stock
v0.5.18 image. It clones the recipe fork, reads the existing compose command,
applies the nine patches, and launches SGLang with persistent model/JIT caches.
Local model paths, TP, port and extra server arguments can be overridden.
Seven CPU-only launcher tests pass; RunPod GPU serving remains unvalidated.

## 2026-09-11 — SGLang v0.5.18 Mamba cache backports

Append alphabetc1's Mamba allocation-demand and coverage-thinning fixes as
`0007`–`0009` after the existing six SGLang patches. The backport preserves
the release's Python eviction loop and omits the absent Rust radix-tree
backend. Upstream sources and removal conditions are tracked in
[`docs/UPSTREAM.md`](../../docs/UPSTREAM.md#sglang-sgl-projectsglang).

The installer supports upgrades from three through eight patches and checks
overlapping patches in a temporary copy before changing installed files.
All 11 installer tests pass; host runtime unit tests are blocked at collection
by a PyTorch CUDA API mismatch. GPU serving and performance remain unmeasured.

## 2026-09-09 — SGLang v0.5.18 DFlash2 backports

Append upstream DFlash2 core (`0005`) and quantized target `lm_head` support
(`0006`) to the existing SGLang patch series, retaining upstream authors and
diffs. Existing `0001`–`0004`, AutoRound W4A8 code and the EAGLE compose are
unchanged. The installer supports upgrades and checks the overlapping DFlash
patches in reverse order in a temporary copy. Sources and removal conditions:
[`docs/UPSTREAM.md`](../../docs/UPSTREAM.md#sglang-sgl-projectsglang).

Local syntax, model-registry/config and unit checks pass; full 3090 TP2 model
load, nonzero DFlash2 acceptance, W4A8 throughput and EAGLE serving regression
remain for on-rig validation. The patch README includes the procedure and a
deterministic smoke script that checks strict accepted-draft counters.

## 2026-09-02 — dual-max: `ASYNC_SCHED=off` wired — vllm#50021 mitigation (1), drafter kept at ~0% cost

The `dual/fp8/mtp.yml` header has documented two mitigations for the [vllm#50021](https://github.com/vllm-project/vllm/pull/50021) MTP × hybrid-GDN wild write since the 2026-08-19 production crash (#1059): (1) `ASYNC_SCHED=off` → `--no-async-scheduling`, which keeps the drafter, and (2) `SPEC=off`, which drops it. Only (2) was actually plumbed — `ASYNC_SCHED` was header prose with no env passthrough and no entrypoint branch, so anyone following the header got the drafter-off path by default and paid for it in decode. This wires (1) the same way `SPEC` is wired: env passthrough + entrypoint `case`, off by default, appended to both `exec vllm serve` branches.

**Measured on-rig (2026-09-02, ref 2×3090 PCIe, v0.27.1, canonical 3 warm + 5 measured, same boot family):**

| Config | narr / **code** decode | prefill @10K / @90K | TTFT | MTP AL |
|---|--:|--:|--:|--:|
| `SPEC=off` (mitigation 2) | 44.2 / **45.1** | 1353 / 1097 | 77–84 ms | — |
| `ASYNC_SCHED=off`, MTP n=3 (mitigation 1) | **67.8 / 88.1** | 1316 / 1058 | 98–113 ms | 3.0–3.3 |
| BENCHMARKS row 2026-08-17 (async ON, MTP n=3) | 67.4 / 85.8 | 1166 / 942 | 152 ms | 2.62 |

Mitigation (1) restores **+53% prose / +95% code** decode over the drafter-off path and is at parity or better with the async-ON catalog row on every column — the header's ~0% cost claim holds. Decode CV 2.1% / 3.3%; VRAM 22.3 GB/card, 0 MiB leak; no Xid / CUDA-error signatures across bench + 8-pack + verify-full on the new config. ⚠️ This bounds the crash *mechanism* the maintainer identified (MTP + prefix-cache + async), not #50021 itself, which is still open — a 10-minute bench is not 44 h of agent traffic. If Xid 31 recurs, `SPEC=off` remains the fallback.

## 2026-08-23 — dual-fast: FlashInfer decode-buffer unpin merged (#1051) — MTP concurrency unlocked to C=32

Merged **[#1051](https://github.com/noonghunna/club-3090/pull/1051)** (thanks **@A1RM4X** — reproduced independently on-rig before promoting). New patch [`vllm-flashinfer-decode-pin`](vllm/patches/vllm-flashinfer-decode-pin/README.md) flips `pin_memory=True→False` on the `flashinfer/decode.py` workspace-buffer allocs, forcing a synchronous plan copy per step and closing the stale-plan async-copy race ([vllm#40756](https://github.com/vllm-project/vllm/issues/40756)) that crashed `vllm/qwen38-27b-dual-fast` (W4A8 + MTP n=4 + fp8 KV) with an Xid 31 VIRT_READ under concurrency. Idempotent, marker-gated, no-ops without FlashInfer, hard-fails on drift (boot-refused). Wired on `mtp.yml` alongside `vllm-gdn-mtp-async-spec-order` (the two fix **distinct** bugs: async wild-write vs decode-plan race).

**On-rig validation (2026-08-23, ref 2×3090, v0.27.1, real delivery path — switch.sh → compose mount → entrypoint install.sh):** the fix holds far past the PR's conservative c=4 claim. With `MAX_NUM_SEQS=32` it survived the full concurrency ladder **clean to C=32** (spec-ON MTP n=4): agg 71→241 tok/s (C=1→32), per-stream decode 84→18, drafter accepting (mean accept 2.85), zero crashes. Unpatched, the same slug crashes at C≥8. Distinct from DFlash2 (dual-superfast), which OOMs at C=8 regardless (VRAM, not this race). Full A/B + all four arms: `learnings/qwen3.8-27b.md` 2026-08-23.

## 2026-08-21 — DFlash2 super/ultra tiers benched (full matrix); iq4xs single-card slug; HOL flag

**The DFlash2 tier hierarchy is measured.** All six dual slugs benched fresh, same session (canonical 3 warm + 5 measured, stock `vllm/vllm-openai:v0.27.1` + the vendored [`vllm-dflash2-backport`](vllm/patches/vllm-dflash2-backport/README.md) of [vllm#52816](https://github.com/vllm-project/vllm/pull/52816)). Decode TPS, narrative / **code**:

| Slug | Drafter · KV / attn | Ctx | narr / **code** | vs base |
|---|---|--:|--:|--:|
| `dual-fast` | MTP n=4 · fp8 | 262K | 73 / **100** | — |
| `dual-superfast` | DFlash2 · fp8 / FlashInfer | 262K | 78 / **141** | **+41%** |
| `dual-ultrafast` | DFlash2 · bf16 / FA2 | ~200K | 128 / **231** | **+131%** |
| `dual-max` | MTP n=3 · fp8 | 262K | 69 / **87** | — |
| `dual-supermax` | DFlash2 · fp8 / FlashInfer | 144K | 68 / **130** | **+49%** |
| `dual-ultramax` | DFlash2 · bf16 / FA2 | 64K | 90 / **172** | **+98%** |

Two structural findings: (1) the **FA2 ⊕ fp8-KV mutual exclusion on Ampere** is what splits `super` (fp8 KV → FlashInfer → keeps 262K, ~40% slower decode) from `ultra` (bf16 KV → FA2 → fastest decode, but 2× KV so context drops); (2) the **fidelity (fp8-weight) series decodes slower than the speed (int4) series** — Ampere has no native fp8 compute, so fp8 weights upcast to fp16. Full numbers + prefill/VRAM in [`../../BENCHMARKS.md`](../../) (stack) and `learnings/qwen3.8-27b.md`.

**Single-card slug swapped `iq4nl` → `iq4xs`.** unsloth removed `Qwen3.8-27B-IQ4_NL.gguf` from the repo (2026-08-19), so `llamacpp/qwen38-27b-single-iq4nl` is dead. Replacement `llamacpp/qwen38-27b-single-iq4xs` (unsloth UD-IQ4_XS, 14.3 GB) ships **q4_0 KV at the full 262144** (halving the KV clears the q8 build's 131072 ceiling) **+ F16 vision** (mmproj-F16, `WITH_VISION=1`). Decode 61.6 narr / 71.3 code; NIAH-clean to 240,635 (91% of n_ctx); one correct image recognition. ⚠️ **q4_0 KV is below the stack serving floor** — a max-ctx / vision *exhibit*, not serving-grade; `KV_TYPE=q8_0 CTX_SIZE=131072` restores the q8@131K config. **🐣 Incubating.** (PR [#1068](https://github.com/noonghunna/club-3090/pull/1068).)

**HOL flag defaulted.** `--long-prefill-token-threshold 4096` added to all 20 vLLM composes (env `LONG_PREFILL_TOKEN_THRESHOLD`, 0 disables). Inert at the shipped `max_num_seqs=1` — preparatory for concurrency (Zylone's tip; qwen3.6 dual-fast precedent).

**Fast-tier weights swap** `Avuja` → `Frozenlock` (both AutoRound INT4) landed earlier — see [vllm#52873](https://github.com/vllm-project/vllm/issues/52873): the permanent MTP acceptance-collapse was checkpoint-specific to Avuja, not a vLLM bug (PR [#1070](https://github.com/noonghunna/club-3090/pull/1070)).

**Port hygiene.** Resolved 3 cross-model `default_port` collisions by moving the qwen3.8 side: `single-iq4xs` 8086→8090, `dual-fast` 8095→8113, `multi4-fast` 8096→8114 (nemotron/inkling/agents-a1 keep their ports). New gate `scripts/tests/test-compose-port-conflicts.sh` fails on any cross-model port overlap (aliases + same-model variants allowed); 8020/8032 allowlisted pending a separate hygiene PR.

## 2026-08-14/16 — onboarding: 5 incubating slugs + v0.27.1 pin + W4A8 default

Qwen3.8-27B onboarded as 🐣 Incubating across llama.cpp + vLLM (1/2/4/8 cards). Served from the official [FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and unsloth's [dynamic GGUFs](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF); AutoRound INT4 fast tier ships **W4A8** (int8 activations) by default. Numbers deliberately withheld at onboarding after a bench ran on a silently-degraded (pipeline-parallelism-off) config — see discussion [#993](https://github.com/noonghunna/club-3090/discussions/993) for the slugs, the sampler rows, and the traps. Same Qwen3-Next hybrid-GDN architecture as 3.5/3.6, so it inherits the [vllm#50021](https://github.com/vllm-project/vllm/pull/50021) MTP crash exposure (mitigate `SPEC=off`).
