# `sglang-v0518-scheduling-and-logging`

Four patches applied to SGLang **inside the container** at startup by
`install.sh`, which the compose runs before `launch_server`. Nothing to
install on the host.

Base is **v0.5.18** (`71de97b264`). The compose pins `lmsysorg/sglang:v0.5.18`
for that reason: `:latest` is already past it — as of 2026-09-04 `:latest` and
`:v0.5.19` share a digest (`sha256:d6e72886...3eda9`).

`bash install.sh --verify` reports which of the four are present, changing
nothing. A container with all of `0001`–`0003` already applied can install
`0004` alone by rerunning `install.sh`.

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

Scoped to `python/sglang/` — the upstream backport also carries test-suite
changes the container has no use for.

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
  or modified `0004` fails its full-patch check. Recreate the container if the
  installed files no longer match the patches.
- **`--decode-passes-per-prefill` on an unpatched engine** — rejected at parse
  time. Harmless.
