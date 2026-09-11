# Mamba path-cap CPU policy A/B — 2026-09-11

Compare the eleven-patch shallow-first insertion cap with `0012` minimax
coverage. Both arms use the same #38000/global-fairness eviction code. This
is a CPU simulation of real tree operations, not a model-serving benchmark.

## Workload

- 4, 8 or 12 independent paths; 24 turns each; `cap=3`.
- Each turn adds 2,048/4,096/3,072/5,120 tokens, staggered by session; final
  context is 86,016 tokens per session.
- Total modeled Mamba pools are 16/24/32 slots, with four reserved execution
  slots. Cached-state ceilings are therefore 12/20/28. This reserve is an
  explicit simulation input, not measured #38151 request demand.
- A slot is reserved before a new frontier is committed. The insertion cap
  runs afterward, so transient allocation can trigger global eviction.
- Full KV has ample capacity. Existing Mamba-driven atomic leaf eviction can
  still delete Full KV; the next turn rebuilds that suffix. Full-KV pressure,
  host offload, shared branches, locks and reused-state marking are absent.
- From turn 4 onward, probe 25%, 50%, 75% and 90% of each session's context
  with the actual prefix matcher. These are diagnostic probes: no request
  execution, LRU refresh or reuse mark. Intermediate radix splits are real.

## Results

Pairs below are **legacy → coverage**. Partial hit means `0 < hit < query`;
zero hit means `hit=0`. Exact-hit probes account for the remainder.

| Sessions | Partial hit | Zero hit | Mean replay tokens | Final mean maximum gap | Peak cached slots |
|---:|---:|---:|---:|---:|---:|
| 4 | 31.5% → 81.5% | 65.8% → 17.0% | 20,621 → 12,515 | 76,288 → 43,264 | 12 → 12 |
| 8 | 51.0% → 75.4% | 47.8% → 22.5% | 18,305 → 15,834 | 67,200 → 59,264 | 20 → 20 |
| 12 | 65.4% → 77.4% | 33.0% → 20.8% | 20,015 → 17,072 | 73,045 → 62,379 | 28 → 28 |

| Sessions | Insertion-cap evictions | Global-pressure evictions | Full-frontier rebuilds |
|---:|---:|---:|---:|
| 4 | 58 → 58 | 26 → 26 | 0 → 0 |
| 8 | 48 → 45 | 124 → 127 | 0 → 1 |
| 12 | 41 → 43 | 219 → 217 | 2 → 2 |

For session 0 with four sessions, the final depths change from
`[77824, 80896, 86016]` to `[34816, 66560, 86016]`. Its maximum gap falls
from 77,824 to 34,816. Every inserted frontier survives its cap action and
each unprotected path has at most three states after that action.

The improvement is not universal: under the eight-session pressure case,
coverage's session 0 ends with `[6144, 86016]`, a 79,872-token gap, and the
coverage arm rebuilds one more full frontier. Global pressure, transient
slot demand and eviction-episode boundaries still matter. The cap cannot
recreate history already removed by global eviction, and fairness is local
to each eviction episode. This is not a guarantee against cold prefills.

Median CPU cap-decision times in this run were approximately 8/5/5 µs for
legacy and 20/14/13 µs for coverage (4/8/12 sessions). They include the tree
scan and synthetic tensor bookkeeping, are not stable performance numbers,
and do not predict GPU TTFT. **TTFT and GPU accuracy are unmeasured.** Replay
tokens estimate recomputation work only.

## Reproduce

After applying the twelve-patch series inside the v0.5.18 environment:

```bash
cd /sgl-workspace/sglang
PYTHONPATH=python python3 test/manual/mem_cache/benchmark_mamba_path_cap.py \
  --turns 24 --cap 3 > /tmp/mamba-path-cap-ab.json
```

The JSON includes every probe, per-turn per-session retained depths and
maximum gaps, occupancy, both eviction counts, CPU decision timing and
`ttft_ms: null`. No endpoint or weights are needed. The global eviction and
tree implementations are imported from the patched source, while the legacy
insertion-cap function is retained only within the benchmark.
