# `sglang-v0518-scheduling-and-logging`

Three patches applied to SGLang **inside the container** at startup by
`install.sh`, which the compose runs before `launch_server`. Nothing to
install on the host.

Base is **v0.5.18** (`71de97b264`). The compose pins `lmsysorg/sglang:v0.5.18`
for that reason: `:latest` is already past it — as of 2026-09-04 `:latest` and
`:v0.5.19` share a digest (`sha256:d6e72886...3eda9`).

`bash install.sh --verify` reports which of the three are present, changing
nothing.

## Why patches, not a fork checkout

A checkout pins what runs by commit and rebases in one operation, which is
tidier for a branch under active work. It also makes the compose depend on
reaching that fork, and hides the diff from review. Three context diffs
against a tagged release are readable in a PR and apply to anyone's image.

The cost is the pin, and it is not theoretical. v0.5.19 is 794 commits past
v0.5.18 and rewrites **all four** files these patches touch; `0001` does not
apply to it. `install.sh` stops there with a non-zero exit before the server
starts, rather than half-patching a tree. Re-cut the patches against the new
tag when moving the pin — do not force them.

## What is in it

| Patch | What |
|---|---|
| `0001-concurrent-chunked-prefill.patch` | Backport of upstream PR [sgl-project/sglang#34623](https://github.com/sgl-project/sglang/pull/34623) — concurrent chunked prefill via `--long-prefill-token-threshold`. Not merged upstream; the v0.5.18 rebase is [apejcic's branch](https://github.com/apejcic/sglang/tree/feat/concurrent-chunked-prefill-v0518), taken as-is. |
| `0002-attention-backend-logging.patch` | Name the resolved attention backends at boot, and say why one was rejected. Auto-selection silently fell back to triton; nothing said which backend ran or on what grounds. |
| `0003-decode-passes-per-prefill.patch` | `--decode-passes-per-prefill`, plus binding `--long-prefill-token-threshold` only when something else is waiting on the pass. |

Scoped to `python/sglang/` — the upstream backport also carries test-suite
changes the container has no use for.

Two further fork-local commits are deliberately NOT here: a Triton
kernel-load log line, and a mamba radix `evictable`/`available` counter in the
batch lines. Both were debugging instruments for the open questions in the
subtree README, not things this compose needs.

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
- **A partially patched tree** — refused rather than reapplied. All three
  touch the same files, so applying over a partial state fails mid-stack and
  leaves something worse than either end. Recreate the container.
- **`--decode-passes-per-prefill` on an unpatched engine** — rejected at parse
  time. Harmless.
