# `sglang-v0518-scheduling-and-logging`

Ten patches applied to SGLang **inside the container** at startup by
`install.sh`, which the compose runs before `launch_server`. Nothing to
install on the host.

Base is **v0.5.18** (`71de97b264`). The compose pins `lmsysorg/sglang:v0.5.18`
for that reason: `:latest` is already past it — as of 2026-09-04 `:latest` and
`:v0.5.19` share a digest (`sha256:d6e72886...3eda9`).

`bash install.sh --verify` reports which of the ten are present, changing
nothing. Rerunning `install.sh` upgrades a container with a complete prefix
of three through nine patches by applying only the remaining patches.

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
