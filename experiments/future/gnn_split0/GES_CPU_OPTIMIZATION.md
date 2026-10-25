# High-performance CPU GES: verified incremental optimization

## Outcome

A separate CPU runner is implemented and small-case equivalence/recovery tests
pass. No full 279-node run was launched or existing process interrupted. Existing
`discover_ges.py`, `ges_recovery.py`, installed causal-learn and all baseline/PC/
BOSS/dataset artifacts are unchanged. Prior outputs are preserved.

The accepted optimization reduces a verified search hotspot, but measured total
wall-time improvement on these sparse subsets is **small**, not an order-of-
magnitude speedup. Full 279-node speedup, runtime and peak RAM remain unverified.

## Source and artifacts

- `scripts/causal/ges_cpu_fast.py`: reachability and subset helpers.
- `scripts/causal/discover_ges_fast.py`: isolated optimized entry point.
- `scripts/causal/validate_ges_fast.py`: existing CPDAG/export validator on new outputs.
- `scripts/causal/test_ges_cpu_fast.py`: helper and search/recovery equivalence.
- `scripts/causal/benchmark_ges_cpu_fast.py`: subprocess runtime/RAM/CPU/thread tests.
- `scripts/causal/benchmark_ges_parallel_scores.py`: ordered thread scoring microtest.
- `graphs/causal_split0/ges_cpu_fast/validation_v1/validation.json`
- `graphs/causal_split0/ges_cpu_fast/opt_v1_report/summary.json`, profile .txt/.pstats.
- `graphs/causal_split0/ges_cpu_fast/parallel_scores_v1.json`
- `graphs/causal_split0/ges_cpu_fast/cli_recovery_v1/`: actual process recovery test.

The new launcher executes a reviewed, source-anchor-checked copy of the reference
wrapper in memory, changes its output root and helper import, and fingerprints
both wrappers plus the new backend. This avoids editing the reference and keeps
its input validation, provenance, protection, exports and genuine recovery logic.
The helper overrides are confined to the compiled function's private globals,
not installed module globals or another running process.

## Profile and actual bottlenecks

80 genes, 2,000 controls, seed 20, lambda=2, one BLAS thread. Profile cumulative
times overlap and include profiler overhead; do not add them as disjoint totals.

| Function/work | Reference cumulative s | Optimized cumulative s |
| --- | --- | --- |
| Insert semi-directed path validity (228,108 calls) | 1.099 | 0.398 |
| Subset inclusion helper (269,909 calls) | 0.916 | 0.700 |
| Insert score-change evaluation (252,877 calls) | 0.798 | 0.765 |
| Delete score-change evaluation (40 calls) | 0.000488 | 0.000506 |
| Gaussian BIC actual cache misses (10,741) | 0.390 | 0.372 |
| NumPy inverse (10,661 calls) | 0.083 | 0.078 |
| DAG-to-CPDAG conversion | 1.079 | 1.058 |
| PDAG-to-DAG conversion | 0.578 | 0.567 |
| Durable checkpoint fsync (37 calls) | 2.736 | 2.743 |

Forward candidate evaluation, validity/subset work, graph conversion and durable
checkpoint I/O matter more here than matrix inversion. The benchmark had no
accepted backward deletions; backward candidate work was small, so no claim about
a difficult backward full-scale phase is justified. Cache layer .score was called
10,821 times, with 10,741 actual local scores; outer candidate cache already avoids
most repeated evaluations. Actual larger graphs can have very different costs.

CPU telemetry is sampled across the venv process tree every 50ms, retaining the
latest process user+system time. For the 80-node unprofiled pair it measured about
0.14 and 0.13 average core-equivalents (~0.59% and ~0.56% of 24 logical CPUs).
These end-to-end averages include durable-write waits/imports and can miss the
last sampling interval; they are not instantaneous kernel utilization. Peak RAM
includes Windows peak working sets and the launcher. No claim of all-core use is
made; full GES remains sequential.

## Accepted optimizations

1. Original `insert_vc2_fast` enumerates simple semi-directed paths with linked
   parent records, repeatedly traverses ancestry and slices its work list. For a
   Boolean validity predicate, visited-node reachability on the graph with blocked
   NAT internal vertices is sufficient: O(V+E) traversal and O(V) working storage.
   This does not alter candidate generation/order, ties, accepted moves or score.
2. Subset inclusion builds the required set once and applies `issubset`, rather
   than rebuilding sets/intersections for every candidate. Output dtype/order are
   retained. No unbounded new graph/query cache is introduced.
3. Existing once-computed ddof=0 covariance and both local-score caches are reused
   unchanged. No algebra rewrites, approximations, score rounding, penalty changes,
   new regularizers, parent limits or alternate stopping criteria were added.

Graph conversion and score kernels stay upstream. Full graph/history state is
still saved for exact recovery. Durable fsync and atomic checkpoint replacement
are intentionally preserved, not disabled to create a misleading speedup.

## Parallelism, threads and rejected alternatives

Ordered independent score microbenchmark, 1,600 two-parent tasks:

| Workers | Seconds | Exact score agreement |
| --- | --- | --- |
| Serial | 0.02088 | Yes |
| ThreadPool 2 | 0.04642 | Yes |
| ThreadPool 4 | 0.06194 | Yes |

Threading slowed these small scores. Full candidate multiprocessing is not
implemented: sharing/copying graph and caches on Windows, worker overhead, GIL-
heavy graph predicates and deterministic first-winner tie reduction need separate
validation. A parallel implementation would have to keep original candidate
ordinal and choose the first maximum exactly, and merge caches deterministically.
No unverified process-pool or unordered-reduction shortcut was adopted.

BLAS controls tested on optimized 80-node GES: one thread 5.968s algorithm / 7.563s
process, two threads 5.875s / 7.437s, four threads 5.953s / 7.500s. All passed exact
trajectory checks. The two-thread observation is fastest but only marginally so;
single trials cannot establish a robust best thread count. Recommend **one BLAS
thread** as the conservative verified configuration, avoiding oversubscription.
Numba/compiled graph kernels were considered but not installed or introduced:
new compilation/type-handling complexity is not justified by demonstrated large
end-to-end gains on these inputs. This is not a claim that compilation could
never help a denser full graph.

## Paired unprofiled measurements

Actual training-control data, sorted seeded column/row subsets, seed 20, float64,
2,000 identical cells, BOSS-matched lambda=2. One thread; snapshots and genuine
checkpoints enabled in both. Single trials under background system activity;
small differences should not be treated as statistically established speedups.

| Genes | Reference algorithm s | Optimized algorithm s | Ref / opt algorithm ratio | Reference / optimized process s | Peak ref / opt MiB |
| --- | --- | --- | --- | --- | --- |
| 20 | 0.203 | 0.266 | 0.76x | 1.766 / 1.766 | 188.77 / 188.15 |
| 40 | 0.672 | 0.625 | 1.08x | 2.203 / 2.156 | 189.04 / 188.36 |
| 80 | 6.078 | 5.968 | 1.02x | 7.656 / 7.563 | 188.62 / 188.72 |

The 20-node algorithm measurement was slower and end-to-end timing unchanged.
80-node process improvement was only ~1.2%. The separately profiled run showed
9.422s vs 8.407s algorithm (~1.12x), but that is profiler-instrumented, not the
production speedup estimate. Report both rather than selecting the larger number.

## Correctness and recovery

All reference/optimized 20/40/80-node runs, two/four-thread controls and profiled
controls had **exact** final scores, insertion/deletion sequences, graph histories,
skeletons and full CPDAG endpoint matrices. Gaussian score arithmetic is identical
by construction: the score implementation was not replaced. The helper test
compared 16,000 randomized small reachability cases (including cycles) and subset
cases to native helpers. A synthetic graph also matched full reference histories.

Forward and backward boundary interruption tests restored exact native scores,
moves, endpoints and graph histories. An actual newly created optimized 40-node
CLI process was terminated after its first genuine checkpoint, resumed and matched
the native 40-node reference. Completed-run protection was verified. Only this
new test process was stopped; pre-existing experiments remained untouched.

The optimized 80-node artifact validator also passes exact edge-list/sparse
adjacency reconstruction, CPDAG extension roundtrip, chain-component constraints
and PyTorch relation-format reconstruction. Actual PyG Data.validate remains
untested when PyG is absent; no dependency was installed.

Genuine loop-boundary checkpoint includes phase, CPDAG, scores, both caches,
counter/history state and RNGs. Resume skips completed accepted moves, repeating
only the unfinished candidate scan. No intra-scan resume is claimed. Checkpoints
are atomically replaced after fsync. Runtime is unlimited. Input/software/source/
thread fingerprints must match. Legacy CPU/CUDA graph-only snapshots cannot be
resumed with this runner; use a NEW run ID. Existing source-fingerprinted CPU
checkpoints are not automatically migrated or relabeled as optimized state.

## Manual full-run commands

New experiment, manually launched only:

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges_fast.py --run-id full_ges_fast_seed20 --nodes 279 --cells 10691 --seed 20 --bic-penalty 2
```

Monitor from another terminal:

```powershell
Get-Content D:\ALI\sccvae\graphs\causal_split0\ges_cpu_fast\full_ges_fast_seed20\progress.log -Tail 20 -Wait
```

After an interruption, same settings plus --resume:

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges_fast.py --run-id full_ges_fast_seed20 --nodes 279 --cells 10691 --seed 20 --bic-penalty 2 --resume
```

To deliberately start that incomplete experiment from zero, use the same command
with `--restart-stage` instead of `--resume`, only after its process stops. This is
not needed for normal recovery. Do not restart an existing CPU run just because
the new helper is available: full-scale acceleration is not established.

After completion:

```powershell
Set-Location D:\ALI\sccvae
.\.venv\Scripts\python.exe scripts\causal\validate_ges_fast.py --run-id full_ges_fast_seed20
Get-Content .\graphs\causal_split0\ges_cpu_fast\full_ges_fast_seed20\statistics.json
Get-Content .\graphs\causal_split0\ges_cpu_fast\full_ges_fast_seed20\validation.json
```

## Scientific limitations

The inherited whole-file normalize_total statistic across all 8,563 genes then
log1p is transductive preprocessing, not strictly inductive. Discovery uses only
the original 10,691 training controls and 279-gene mapping; no held-out response
was used for features, graph fitting or optimization selection. Observational
Gaussian SEM/Markov/faithfulness/sufficiency assumptions, compositionality,
latent confounding and cell/batch dependence limit causal interpretation. No
orientations were invented or unresolved edges discarded. No GNN was implemented
or trained. No full-size causal result or guaranteed time saving is claimed.
