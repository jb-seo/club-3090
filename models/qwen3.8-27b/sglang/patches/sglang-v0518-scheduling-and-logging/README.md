# `sglang-v0518-scheduling-and-logging`

Twelve patches applied to SGLang **inside the container** at startup by
`install.sh`, which the compose runs before `launch_server`. Nothing to
install on the host.

Base is **v0.5.18** (`71de97b264`). The compose pins `lmsysorg/sglang:v0.5.18`
for that reason: `:latest` is already past it — as of 2026-09-04 `:latest` and
`:v0.5.19` share a digest (`sha256:d6e72886...3eda9`).

`bash install.sh --verify` reports which of the twelve are present, changing
nothing. Rerunning `install.sh` upgrades a container with a complete prefix
of three through eleven patches by applying only the remaining patches.

## Why patches, not a fork checkout

A checkout pins what runs by commit and rebases in one operation, which is
tidier for a branch under active work. It also makes the compose depend on
reaching that fork, and hides the diff from review. The context diffs
against a tagged release are readable in a PR and apply to anyone's image.

The cost is the pin, and it is not theoretical. v0.5.19 is 794 commits past
v0.5.18 and rewrites **all four** files the scheduling/logging patches touch; `0001` does not
apply to it. `install.sh` stops there with a non-zero exit before the server
starts, rather than half-patching a tree. Re-cut the patches against the new
tag when moving the pin — do not force them.

## What is in it

| Patch | What |
|---|---|
| `0001-concurrent-chunked-prefill.patch` | Backport of upstream PR [sgl-project/sglang#34623](https://github.com/sgl-project/sglang/pull/34623) — concurrent chunked prefill via `--long-prefill-token-threshold`. Not merged upstream; the v0.5.18 rebase is [apejcic's branch](https://github.com/apejcic/sglang/tree/feat/concurrent-chunked-prefill-v0518), taken as-is. |
| `0002-attention-backend-logging.patch` | Name the resolved attention backends at boot, and say why one was rejected. Auto-selection silently fell back to triton; nothing said which backend ran or on what grounds. |
| `0003-decode-passes-per-prefill.patch` | `--decode-passes-per-prefill`, plus binding `--long-prefill-token-threshold` only when something else is waiting on the pass. |
| `0004-autoround-w4a8.patch` | Experimental AutoRound dense W4A8 through the existing GPTQMarlin pipeline: per-token INT8 activations, original signed floating group scales, FP16/BF16 outputs. Includes the standalone JIT CUDA sources. |
| `0005-add-dflash2-runtime-support.patch` | DFlash2 core: model registry entry, local convolution, candidate selector, V2 worker and config handling. Upstream `c14312a66420b75ca9a11bf1817c4db1fa26b097`. |
| `0006-support-quantized-target-lm-head-for-dflash2.patch` | DFlash2 selector support for the target's existing quantized `lm_head`, including vocabulary padding and CUDA graph admission. Upstream `1cf2b8c54d81802abc15dcf23a29b9cc687bc01e`. |
| `0007-mamba-alloc-req-slots-demand.patch` | Counts only the active and ping-pong Mamba slots the batch still needs, then evicts only the shortfall. |
| `0008-mamba-coverage-thinning.patch` | Selects cold-path checkpoint victims by coverage while preserving v0.5.18's eviction loop. |
| `0009-mamba-protect-reused-states.patch` | Marks reused Mamba states and excludes them from deeper thinning victims. |
| `0010-fix-mamba-demand-v0518-request-fields.patch` | Corrects `0007` to use v0.5.18's request-level Mamba fields; fixes the first-prefill `req.kv is None` crash. Includes regression tests using real `Req` objects. |
| `0011-mamba-inter-path-eviction-fairness.patch` | Spreads Mamba eviction across cold paths, retaining two usable checkpoints per path while possible, with a hard-pressure LRU fallback. |
| `0012-mamba-path-cap-minimax-coverage.patch` | Changes the insertion-time per-path cap to minimize the maximum replay gap, with deterministic quantile ties and protected checkpoints. Includes unit tests and a synthetic policy A/B. |

`0001`–`0004` are scoped to `python/sglang/`. `0005`/`0006` retain their
upstream tests and original format-patch author/commit metadata. Their bytes
are exactly `git format-patch -1 <upstream-sha> --stdout`; no adaptations.

Two further fork-local commits are deliberately NOT here: a Triton
kernel-load log line, and a mamba radix `evictable`/`available` counter in the
batch lines. Both were debugging instruments for the open questions in the
subtree README, not things this compose needs.

## AutoRound W4A8 experiment (`0004`)

The existing `compose/dual/autoround-int4/mtp.yml` mounts this directory and
runs `install.sh`, so it now installs W4A8 without an extra compose override
or source checkout mount. Recreate the service to load the patched code.
CUDA compilation uses SGLang's existing JIT infrastructure and requires the
image's CUDA compiler; the first load includes compilation time.

The patch contains two hooks, one Python helper and seven CUDA source files.
Device code is adapted from vLLM v0.27.1, commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, retaining source attribution.
The reference repository is not included, and no vLLM package, checkout or
shared library is needed at build time or runtime.

Scope: SM8x, symmetric AutoRound INT4, no act-order, group sizes 32/64/128,
FP16/BF16 input/output. Per-shard K must be divisible by 128 and N by 64;
unsupported shapes in otherwise eligible layers fail during preparation.
Other quantization methods and ineligible layers retain their existing path.
Eligible decode/MTP linears also use W4A8. The model-facing dtype and TP
handling stay in the existing pipeline. Group scales remain signed floating
point, with INT32 accumulation within each group and FP32 accumulation across
groups; there is no INT4 sign folding or integer group-scale encoding.

Expected preparation log:

```text
[Marlin] W4A8 prepared: weight_bits=4 activation_dtype=int8 output_dtype=torch.bfloat16 scale_encoding=signed_float arch=sm86 K=... N=... group_size=128
```

For an A/B baseline, set `ENABLED = False` in the added
`python/sglang/kernels/ops/quantization/gptq_marlin_w4a8.py` hunk of `0004`
and recreate the container from the stock image. Restore `True` and recreate
to enable W4A8 again. Editing an installed file directly causes `--verify`
to report a mismatch; there is deliberately no new server flag.

Development validation: 14 kernel/integration tests passed on RTX 4060
(SM89), including signed scales, large prefill shapes and CUDA graph replay;
targeted Compute Sanitizer memcheck/initcheck reported zero errors. The
BF16/group-128 kernel compiled for SM86 and its SASS contains INT8 IMMA.
**3090 TP2 model correctness, MTP stability and end-to-end throughput remain
unmeasured.** One initial cold-JIT comparison showed an anomalous maximum
difference (6.665 at M=2048); subsequent reference checks did not reproduce
it, and its cause remains unresolved. This is an experimental deployment.

## Upstreamability

`0001` is someone else's PR and belongs upstream on its own.

`0003` is fork-local and not offered upstream. Two things would have to be
settled first:

- The knob is in passes, not a time share, because the decision has to be a
  pure function of state every TP rank shares. Wall-clock is rank-local: an
  earlier revision measured it per rank, the accumulators drifted within
  seconds, and one rank ran prefill while another ran decode until their
  collectives deadlocked. Upstream already negotiates this class of decision
  across ranks for the prefill delayer (`negotiate_should_allow_prefill`); a
  proper version would use that seam rather than a counter.
- `--enable-mixed-chunk` solves the same problem by merging decode rows into
  the prefill batch, and is strictly better where it applies — it just
  refuses to coexist with speculative decoding. Making mixed-chunk work with
  EAGLE would make this flag unnecessary.

`0002` is small and independently upstreamable.

`0004` is a local experiment with a source-level switch, not a general
quantization configuration or an upstream proposal.

## Mamba cache backports (`0007`–`0009`)

These are the three local cherry-picks of upstream PRs #38151 and #38000,
authored by alphabetc1. Source commits, status and removal conditions are in
[`docs/UPSTREAM.md`](../../../../../../docs/UPSTREAM.md#sglang-sgl-projectsglang).
Each patch retains the author and commit metadata from its local
`git format-patch` export: `fc84595d72`, `bc8d756a54`, `a50d98ad91`.

The #38000 backport keeps the release's `while` eviction loop: when a deeper
checkpoint is thinned, it continues until the requested number of slots is
freed or a device leaf is returned to the driver. The upstream Rust radix-tree
changes and integration tests are omitted because v0.5.18 has no such backend.
Both PRs' Python unit tests are included.

PR #32606 is intentionally absent. It was closed without merge after #34043
and #34184 fixed the same prefill-graph tracking problem more completely, and
both successors already ship in v0.5.18. Applying #32606 here would restore
the obsolete speculative-decoding exclusion in `_is_mamba_track_enabled()`,
undo part of #34043 and send affected prefills through eager execution. The
source commits and ancestry check are recorded in `docs/UPSTREAM.md`.

Local validation (2026-09-11): the installer regression suite passes 11 tests,
including fresh v0.5.18 installation, upgrades from three through eight
patches, repeated installation, read-only verification and rejection of
partial, modified or out-of-order patches without changing installed files.
The Python source passes syntax checks. Runtime unit tests in the host SGLang
checkout could not collect because its PyTorch lacks
`torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool`; GPU serving and
throughput have not been validated for these backports.

### v0.5.18 compatibility correction (`0010`, 2026-09-11)

The initial `0007` backport applied cleanly but referenced a newer request
layout: `req.kv.holds_mamba` and `req.kv.mamba_ping_pong_track_buffer`.
In v0.5.18, `req.kv` starts as `None`, and even initialized `ReqKvInfo` has
neither Mamba field. The actual allocator uses `req.mamba_pool_idx` and
`req.mamba_ping_pong_track_buffer`. `0010` matches that allocator and retains
the reduced eviction demand for COW matches and continuing chunks.
The reported `NoneType ... holds_mamba` exception is independent of GPU type.

The original upstream test doubles also assumed the newer layout and cache
API. The correction constructs real release `Req` objects and uses
`cache.evict`, covering fresh requests, initialized KV info, COW, chunked
continuations, lazy/non-overlap/no-extra-buffer modes and mixed batches.
The six allocation tests reproduce the failure before the fix and pass
after it in a local SGLang v0.5.18 container using CPU tensors. The nine
Mamba thinning tests and 13 installer tests also pass. GPU serving remains
unvalidated.

`0001`–`0009` stay byte-identical. After updating the recipe checkout, rerun
`install.sh` or the RunPod launcher: a container with all nine old patches
receives only `0010`; recreating the container is not required. Restart the
SGLang process to load the corrected code. The RunPod launcher reuses its
checkout without pulling, so update it explicitly before rerunning:

```bash
git -C /workspace/club-3090 pull --ff-only
bash /workspace/club-3090/models/qwen3.8-27b/sglang/scripts/run_in_runpot.sh
```

### Inter-path Mamba eviction fairness (`0011`, 2026-09-11)

This section records the eleven-patch policy. Its insertion-cap comparison
uses the old shallow-first cap; `0012` below changes that cap to coverage
selection. The global fairness policy itself remains the same.

The coverage selector from `0008`/`0009` can repeatedly thin the same cold
path in one allocation episode. `0011` adds eviction-local LRU sweeps above
that selector. Each sweep removes at most **one checkpoint per path**, then
gives other paths a turn. A path at **two usable device checkpoints** is
skipped while another path can contribute above that floor. Additional
sweeps repeat until demand is met. If a whole sweep makes no progress, the
cursor returns to the true LRU tail and the original coverage policy may
evict below the floor, including sacrificing the coldest path entirely.

A path here is a maximal non-branching tree segment: a fork ends its incoming
segment and each child starts a separate one. Shared ancestors are not
counted repeatedly toward each child's floor. Counting uses the actual
device match validators, including the SWA window, and requires a resident
Full-KV prefix without pending load-backs. Host-only, tombstoned, pending or
otherwise unusable checkpoints and request execution buffers do not count.
Existing deeper-victim protections remain; load-back-pinned candidates are
also skipped during hard pressure. The min-gap scoring and reused-state
marking are unchanged.

No timestamp or real-node LRU refresh is introduced. Path counts and visited
nodes are bounded by the tree scanned during eviction, memoized per sweep,
and discarded when eviction ends. Ordinary allocation does no path scan.
Separate one-slot eviction episodes still begin at the true LRU tail; this
is fairness within an episode, not persistent rotation across allocations.
The floor applies to Mamba-driven eviction. Full-KV eviction and the existing
insertion-time `--mamba-max-states-per-path` cap remain independent.

**The current compose sets that insertion cap to 2**, equal to the new floor.
If every candidate path already has only two checkpoints, hard-pressure
fallback is expected. To exercise distributed thinning, use a cap above 2
or disable insertion thinning with `--mamba-max-states-per-path=-1`. After
publishing this recipe update, update the existing RunPod checkout and
restart SGLang with the appended override:

```bash
git -C /workspace/club-3090 pull --ff-only
bash /workspace/club-3090/models/qwen3.8-27b/sglang/scripts/run_in_runpot.sh \
  --mamba-max-states-per-path=-1
```

Omitting the flag from the SGLang command also defaults to `-1`. Omitting
extra launcher arguments does not remove the compose's existing `=2` flag.
The separate `--max-mamba-cache-size=32` pool limit still applies.

CPU simulation using the actual insertion-cap, eviction and prefix-match
code: four independent paths A–D, coldest first, each initially checkpointed
at 20K, 40K, 60K, 80K, 100K and 120K. Full KV is resident; there are no locks,
shared branches or reused-state markers. With eight slots available for
cached checkpoints after other uses of the pool:

| Insertion cap | Checkpoints retained on each path | Resume point for a matching 90K prefix |
|---|---|---|
| `2` | 100K, 120K | 0K |
| Omitted / `-1` | 80K, 120K | 80K |

The cap removes shallow states during insertion; without it, pressure-driven
min-gap thinning preserves wider coverage. With only four cache slots left,
hard pressure empties A and B while C and D retain their two checkpoints.
Episode boundaries also matter: from six states per path, requesting four
slots in one eviction removes one from each path (5/5/5/5 remain); requesting
one slot in each of four separate evictions removes four from A (2/6/6/6
remain). This simulation measures cache policy, not GPU serving performance.

The installer adds only `0011` to a ten-patch container. Existing patches
`0001`–`0010` retain their bytes, including the exact-demand allocation fix.
Validation: 17 new fairness tests cover multiple paths/sweeps, partial prefix
matches, real leaf-driver callbacks, hard-pressure progress, locks, session
references, reuse, forks, load-back pins and invalid checkpoint counting.
Together with the existing thinning, allocation and path-cap tests, 39 pass
and five GPU tests skip in the local v0.5.18 container. All 14 installer tests
pass, including fresh installation, upgrades from three through ten patches,
idempotence and rejection of drift. GPU serving, TP2 behavior under load and
real-workload cache-hit distributions remain unmeasured for this policy.

### Coverage-aware insertion cap (`0012`, 2026-09-11)

`--mamba-max-states-per-path=N` now bounds checkpoints after each insertion
by replay coverage. With depths 20, 40, 60 and a new frontier at 70, `N=3`
removes 60 and retains **20, 40, 70**, reducing the worst gap from the old
policy's 40 to 30. This happens during the existing insert action, even when
the global pool still has free slots.

Each replaceable old holder is scored against **all survivors** from root
depth zero to the mandatory frontier. The score is ordered by maximum gap,
total absolute deviation from uniform quantiles, deeper victim first, then
node ID. Quantile deviation uses integer scaling by the survivor count, so
the decision needs no floating-point comparison or device synchronization.
The primary objective is whole-path minimax, not #38000's local merged gap.
If multiple excess holders exist, the selector recomputes after each victim;
it is an online greedy policy over available states, not an optimizer that
can recreate previously evicted checkpoints.

The frontier, forks, locks, session references, reused states, load-back
pins and protected device leaves cannot be selected. Mandatory holders can
exceed the numeric cap. They naturally partition replay gaps, but this first
implementation scores the root-to-tail path and global quantiles; it does
not introduce a separate optimization domain at each mandatory ancestor.
Full KV, host backups, slot allocation, #38000 and inter-path fairness keep
their existing behavior. There is **no Rust radix-tree backend in v0.5.18**;
this patch targets the Python implementation actually present in this tree.

Backup ordering is unchanged: the action runs after the insert's walk-time
BackupKV, preserving its locks, and uses the existing failure-safe free
drain. A new checkpoint can temporarily require **N+1 persistent slots**
before reduction; admission-time victim-slot reuse is deferred. A full pool
may therefore still invoke global eviction before this cap can run.

For the requested three-checkpoint setting, update the recipe checkout after
publication and append the override to the launcher (the compose still uses
2). Omitting the flag directly in SGLang means unlimited; omitting launcher
arguments still inherits the compose value.

```bash
bash /workspace/club-3090/models/qwen3.8-27b/sglang/scripts/run_in_runpot.sh \
  --mamba-max-states-per-path=3
```

Enable DEBUG for
`sglang.srt.mem_cache.unified_cache.components.mamba_component` to inspect
tail ID, cap, root/frontier depths, holders, protected depths, candidate
scores `(max_gap, scaled_deviation, -depth, node_id)`, chosen victim and
before/after maximum gaps. Default logging adds no per-decision INFO output.

Validation: 16 new test methods include 182 exhaustive small-path choices,
growing frontiers, protections, soft overflow, host/Full KV preservation,
actual partial prefix matching and global pressure/fairness integration.
The combined runtime suite passes 55 tests with five GPU backup tests skipped;
the local container's CUDA initialization fails with error 500, so those
ordering tests were not executed on GPU. All 17 installer tests pass,
including fresh v0.5.18, three-through-eleven-patch upgrades and rejection
of partial/modified `0012`. Existing `0001`–`0011` remain byte-identical.
The [synthetic A/B report](validation/mamba-path-cap-2026-09-11.md) records
slot occupancy, per-session geometry, partial/zero hits and replay work.
GPU serving, numerical correctness, TP2 throughput and TTFT are unmeasured.

## DFlash2 backports (`0005` / `0006`)

Upstream sources, commit SHAs and drop conditions are tracked in
[`docs/UPSTREAM.md`](../../../../../../docs/UPSTREAM.md#sglang-sgl-projectsglang).
Authors: Zihan Zhang, Jian Chen and Liangsheng Yin (`0005`); Jimmy Shong and
LING ZHI (`0006`), with the original coauthor trailers retained in each patch.

Base: SGLang `v0.5.18`, commit
`71de97b264b04dcd514cf904003028aefe9775c8`, followed by the existing four
patches. This release already has `DFlashWorkerV2`, `DFlashDraftInputV2`,
ReplaySSM plumbing, nested `dflash_config` parsing and explicit draft
`unquant` handling. Those are prerequisites, not missing files to replace.
The release also has the newer `reserved_seq_lens_*` naming in the V2 worker;
both upstream patches apply cleanly and preserve that release fix.

`0005` modifies:

- `python/sglang/kernels/ops/speculative/dflash.py`: convolution kernel.
- `python/sglang/srt/models/dflash.py`: `DFlash2DraftModel` in `EntryClass`,
  grouped local convolution, candidate selector and token-path walk.
- `python/sglang/srt/speculative/dflash_utils.py`: convolution/selector config.
- `python/sglang/srt/speculative/dflash_worker_v2.py`: selector proposal,
  greedy/sampled verification and CUDA graph integration.
- `python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py`:
  Muse-specific layer mapping; Qwen layer IDs stay unchanged.
- `test/registered/unit/spec/test_dflash_logits.py` and
  `test/registered/unit/model_executor/model_runner_components/test_spec_aux_hidden_state.py`.

`0006` modifies only `models/dflash.py`, `speculative/dflash_worker_v2.py`
(under `python/sglang/srt/`) and `test_dflash_logits.py`. It calls the existing
`lm_head.quant_method.apply`, masks padded vocabulary rows with `-inf`, gathers
TP vocabulary candidates, and admits supported quantized heads for graph
capture. It does not change AutoRound config, Marlin dispatch, INT8 activation
quantization, or any EAGLE implementation. The BF16 draft still needs explicit
`--speculative-draft-model-quantization=unquant`: omission inherits target
quantization in this base release.

The official `z-lab/Qwen3.8-27B-DFlash2` config fetched on 2026-09-09 declares
`dflash_config.target_layer_ids=[5,19,33,47,61]`, `block_size=8`,
`conv_kernel_size=2`, `conv_group_size=16`, `selector_rank=256`, and
`selector_top_k=16`. These explicit IDs take precedence over the layer-spacing
fallback `[1,16,31,46,61]`. Block size 8 means seven proposed draft tokens and
one anchor per verification block. Both overlap and synchronous DFLASH use
the V2 worker; no `SGLANG_ENABLE_SPEC_V2` environment variable is needed.

### Local validation (2026-09-09)

- Original `0001`–`0004` patch bytes and existing W4A8 edits unchanged.
- `0005`/`0006` byte-identical to upstream `git format-patch` output; no adaptation.
- Clean v0.5.18 install, upgrades from 3/4/5 patches, repeated install and
  `--verify`, rejection of partial/modified patches without mutation: 7 tests pass.
- Actual registry resolution, imports of both V2 workers/info, official config
  as a dictionary and Transformers config, and explicit draft-unquant handling pass.
- `python3 -m compileall -q python/sglang` passes on the complete patched tree.
- Upstream DFlash2/quantized-head/layer-mapping unit tests: 12 pass on CPU.

This workstation has one RTX 4060 8GB and no target weights; its test container
also reports CUDA initialization error 500. **No GPU kernel, full draft load,
3090 TP2 acceptance, throughput or end-to-end EAGLE claim is made here.**
The user will run those checks on the serving rig; remote access is unavailable.

Reproduce installer and upstream tests (the latter need the SGLang environment
and pytest):

```bash
python3 /patches/test_install.py /path/to/sglang-git-checkout
cd /sgl-workspace/sglang
python3 -m pytest -q test/registered/unit/spec/test_dflash_logits.py \
  test/registered/unit/model_executor/model_runner_components/test_spec_aux_hidden_state.py
bash /patches/install.sh --verify
```

### Serving-rig validation: preserve EAGLE, then test DFLASH

The shipped `mtp.yml` stays on its existing EAGLE configuration. Its existing
mount and `bash /patches/install.sh` command automatically install all ten
patches in order at startup; no Docker build or image change is required.
Copy this entire patch directory to the serving checkout, then recreate the
existing service using the same environment/model/cache paths.

1. Run the EAGLE smoke below and the existing club benchmark first. Record
   target `[Marlin] W4A8 prepared: ... activation_dtype=int8` logs and peak VRAM
   per card. The user-reported comparison points are about 1700–1750 short
   prefill tok/s and 92/117 narrative/code decode tok/s, not locally remeasured.
2. For DFLASH, copy the existing compose beside `mtp.yml` as a temporary
   untracked file. Add a read-only volume for the downloaded official BF16
   `z-lab/Qwen3.8-27B-DFlash2` directory at `/models/draft`. Replace the four
   EAGLE argument entries with exactly:

   ```yaml
   - --speculative-algorithm=DFLASH
   - --speculative-draft-model-path=/models/draft
   - --speculative-draft-model-quantization=unquant
   - --speculative-dflash-block-size=8
   ```

   Remove the old `--speculative-num-draft-tokens=4`; it conflicts with block
   size 8. Keep ReplaySSM, TP2, AutoRound, dtype, FP8 KV, FlashInfer, page size,
   scheduling and memory settings as in the working compose. Recreate the
   same service with the temporary compose after stopping its EAGLE instance.
   Do not enable any `SGLANG_SIMULATE_ACC_*` benchmark controls.
3. Check boot logs for `model=DFlash2DraftModel`, `block_size=8`, capture layers
   `[5,19,33,47,61]`, and the W4A8 preparation logs. Run the DFLASH smoke below.
   Review the actual generated Python function, in addition to its counters.
4. Return to the original `mtp.yml` and repeat the EAGLE smoke/benchmark.

Run inside the appropriate already-started container (default name shown):

```bash
docker exec sglang-qwen38-27b-mtp-dual python3 /patches/smoke_spec.py \
  --algorithm EAGLE --output /tmp/eagle-smoke.json
# After starting the temporary DFLASH configuration:
docker exec sglang-qwen38-27b-mtp-dual python3 /patches/smoke_spec.py \
  --algorithm DFLASH --output /tmp/dflash2-smoke.json
docker cp sglang-qwen38-27b-mtp-dual:/tmp/dflash2-smoke.json ./dflash2-smoke.json
```

Copy each result out before recreating its container. The smoke sends one
thinking-OFF, temperature-zero request using the target tokenizer's template.
It saves output and metadata even if acceptance is zero, and requires
`spec_verify_ct > 0`, `spec_num_correct_drafts > 0`, and `spec_accept_rate > 0`.
`spec_accept_length > 0` alone is insufficient because it includes bonus tokens.
When accepted-draft count is zero, inspect config/layer IDs, tokenizer alignment,
convolution/selector loading, quantized head and ReplaySSM logs in that order.
No performance tuning is included in these patches.

## Verifying at runtime

Boot should print, per rank and per model runner (target and MTP draft):

```
Hybrid GDN model: full attention on prefill=flashinfer decode=flashinfer
(--attention-backend, allowed on this platform: unrestricted); GDN linear
attention on prefill=triton decode=triton verify=triton (--linear-attn-backend).
Hybrid attention wiring: hybrid=HybridLinearAttnBackend full=FlashInferAttnBackend
linear=GDNAttnBackend (16 full-attention layers)
```

Two `Hybrid attention wiring` lines per rank is correct, not a leak: the
target reports its real full-attention layer count, the MTP draft reports `1`
(its layer is hardcoded full-attention).

With `--decode-passes-per-prefill` set, decode passes appear between prefill
chunks and the achieved split is logged:

```
decode-share: resuming prefill after 50 decode passes (decode_time_share=0.26 ...)
```

**Both TP ranks must print the same count.** Different counts mean the ranks
have diverged, which ends in an NCCL hang with the GPUs busy and nothing else
in the log.

## Known-bad configurations

- **An image that is not v0.5.18** — `install.sh` stops with `DOES NOT APPLY`
  before the server starts. Verified against v0.5.19: `0001` is refused.
  Left to itself, a v0.5.18 diff landing in a later tree dies in the
  scheduler with an `ImportError` naming an unrelated module (`get_platform`
  moved into `runtime_context` after v0.5.18), which reads as anything but a
  version mismatch.
- **A partially patched tree** — an incomplete `0001`–`0003` stack is refused
  rather than reapplied. A complete old stack can upgrade to `0004`; a partial
  or modified `0004` fails its full-patch check. Complete `0001`–`0004` and
  `0001`–`0005` stacks can upgrade to all six. `0006` is reversed in a temporary
  copy before checking `0005`, so shared context does not cause reapplication.
  All pending patches are preflighted in a temporary copy before installation;
  partial/modified DFlash or Mamba files are refused. Recreate the container if the
  installed files no longer match the patches.
- **`--decode-passes-per-prefill` on an unpatched engine** — rejected at parse
  time. Harmless.
