# CUDA GES feasibility: not recommended for production

## Decision

A separate real CUDA scoring pilot was implemented. It is **slower**, and exact
CPU search-trajectory equivalence failed at 40 genes. Do not restart the 279-node
experiment on this CUDA pilot. No full discovery was launched, no existing
process was interrupted, and CPU runner/recovery sources and prior outputs were
not edited. The scientifically equivalent, faster-CUDA goal was NOT achieved.

## Implementation

`ges_cuda_backend.py` subclasses causal-learn's LocalScoreClass; covariance is
computed with the original NumPy ddof=0 float64 expression and copied once to the
RTX A4000. Empty/single-parent scores are prefetched in one GPU calculation;
larger-parent covariance submatrices and float64 inverse/residual calculations
run on CUDA on cache misses. Cache hits stay on CPU. Scalar residuals return to
CPU for the original NumPy log/penalty formula. TF32 is disabled; no float32 or
automatic CPU fallback is used. Library singular-matrix pseudoinverse and
nonpositive variance clipping policies are retained, but their GPU numerics are
not guaranteed bitwise identical. No new packages installed; PyTorch 2.4.1+cu124.

Search insertion/deletion operators, candidate order, validity checks, maxP=p,
forward/backward loops and DAG/CPDAG conversions remain upstream causal-learn
0.1.4.8. Lambda=2, matching BOSS and the CPU experiment; conventional library BIC
is lambda=0.5. Neither penalty nor features were test-tuned. Gaussian local score:
`-n/2*(1+log(sigma)) - lambda*(parent_count+1)*log(n)`.

`discover_ges_cuda.py` is intentionally restricted to <=80 genes. It verifies the
same training-control source rows/mapping, stores fingerprints and seed 20, and
uses 2,000 identical cells per CPU/CUDA subset. All CPU/GPU timings include genuine
checkpoint writes and graph export. Both variants share the PyTorch import envelope.

## Profile

Evidence: `graphs/causal_split0/ges_cuda_pilot/profile_n80_cpu_v1/profile.txt`.
An 80-gene CPU cProfile run took 9.666 profiled seconds. Cumulative times overlap:

| Operation | Cumulative seconds |
| --- | --- |
| Checkpoint callback, 37 saves | 3.507 |
| fsync alone | 3.058 |
| DAG-to-CPDAG conversion | 1.099 |
| Insert semi-directed-path check | 1.097 |
| Candidate subset inclusion | 0.924 |
| Original local score, 10,741 calls | 0.395 |

Gaussian scoring was only ~4.1% of profiled wall time. This supports a small
upper bound on benefits from accelerating only scoring, not a 279-node profile
claim. CPU candidate/subset/path checks, conversions and durable writes dominate.
Existing two-level local-score caches already suppress repeated computations.
Resident covariance/singleton prefetch reduce transfers, but adaptive larger-parent
scores still incur small-kernel synchronization per cache miss. General batching
would require carefully reorganizing candidate discovery/tie handling; not done
automatically because exact behavior is important. No claim of an exhaustive
GPU optimization search is made.

## Benchmarks

Results: `graphs/causal_split0/ges_cuda_pilot/pilot_v1_report/summary.json` and
`pilot_v2_report/summary.json`. Individual pilots capped externally at 60 seconds;
all completed. Round 1 overlapped the independent profiling pilot and CUDA recovery
tests overlapped part of round 2; therefore these are descriptive feasibility
measurements under background activity, not isolated microbenchmark estimates.
The user's existing full experiment also remained untouched. Both rounds found
CUDA slower at all three sizes. Round 2:

| Genes | CPU algorithm s | CUDA algorithm s | CPU process s | CUDA process s | CPU / CUDA process ratio |
| --- | --- | --- | --- | --- | --- |
| 20 | 0.187 | 0.468 | 3.062 | 3.375 | 0.91x |
| 40 | 0.687 | 1.109 | 3.469 | 4.141 | 0.84x |
| 80 | 6.250 | 8.500 | 9.094 | 11.563 | 0.79x |

Process time includes imports/input checks/search/checkpoints/export and the
post-run CUDA score audit. Algorithm time excludes that audit. Thus the slower
CUDA algorithm times cannot be attributed solely to the post-run audit.
CUDA peak host working sets: ~781–792 MiB, vs CPU ~512–513 MiB. PyTorch peak CUDA
allocated memory ~8.1–8.2 MiB and reserved 22 MiB; these exclude driver/context and
display allocations. The pilot is not memory-capacity limited.

## Scientific agreement and failure boundary

All 20/40/80-gene final skeletons AND full endpoint matrices match the CPU controls
in both rounds. Final scores match exactly. Directed/unresolved edge counts remain
2/0, 0/7 and 30/5 respectively. Cached GPU score comparisons against native CPU:
453, 2,324 and 10,741 checked scores; maximum absolute difference
4.547473508864641e-13. Singleton prefetch scores are included even if never queried.

However, at 40 genes the first insertion changed from CPU `31 -> 32` to CUDA
`32 -> 31`; subsequent updates and final unresolved CPDAG match. This is consistent
with floating-point near-tie behavior; it is not an invented edge correction.
Exact trajectory equivalence is therefore NOT established, and larger graphs
could diverge. No score rounding, tolerance-based tie override or forced CPU
orientation was introduced to hide the discrepancy. Endpoint agreement on small
cases is not a universal scientific equivalence proof.

## Genuine recovery

The separate pilot reuses unchanged `ges_recovery.py`: atomic flushed/fsynced
checkpoints contain phase, graph, score/new score, update/graph histories, both
score caches, counters and RNG states. GPU covariance tensors are explicitly
excluded from pickle and recreated lazily from stored CPU covariance on resume.
Interrupted candidate scans repeat; accepted moves survive. Source/input/backend/
software fingerprints must match, and active/completed run IDs are protected.

`test_ges_cuda_recovery.py` forced forward and backward-boundary interruptions;
resumed CUDA endpoints, scores and update sequences matched uninterrupted CUDA.
Evidence: `graphs/causal_split0/ges_cuda_pilot/recovery_v1/validation.json`.
This establishes genuine CUDA-to-CUDA recovery on small cases, not compatibility
with CPU checkpoints or historical graph-only snapshots. Use only trusted pickles.

## Manual pilot commands only

These are for investigating the slower experimental pilot, NOT recommended as
the 279-node replacement. No timeout is imposed by the pilot runner.

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges_cuda.py --run-id manual_cuda_ges_n40 --backend cuda --nodes 40 --cells 2000 --seed 20 --bic-penalty 2
```

Monitor in another terminal:

```powershell
Get-Content D:\ALI\sccvae\graphs\causal_split0\ges_cuda_pilot\manual_cuda_ges_n40\progress.log -Tail 20 -Wait
```

After interruption, same settings/environment plus --resume:

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges_cuda.py --run-id manual_cuda_ges_n40 --backend cuda --nodes 40 --cells 2000 --seed 20 --bic-penalty 2 --resume
```

Explicit restart (only after old process stops; discards algorithm progress):

```powershell
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges_cuda.py --run-id manual_cuda_ges_n40 --backend cuda --nodes 40 --cells 2000 --seed 20 --bic-penalty 2 --restart-stage
```

```powershell
Get-Content D:\ALI\sccvae\graphs\causal_split0\ges_cuda_pilot\manual_cuda_ges_n40\statistics.json
```

No full-scale CUDA command is provided because neither speed nor strict search
equivalence passed the stated goal. Keep the genuine-resume CPU implementation
as the production reference. Do not stop an existing CPU run just to use CUDA.

## Inherited limitations

Only training-control cells were used; no held-out responses. Original whole-file
normalize_total library-size statistics across 8,563 genes followed by log1p are
transductive preprocessing, not strictly inductive. Linear/Gaussian SEM, Markov,
faithfulness, sufficiency and acyclicity assumptions may fail under latent
confounding, cell/batch dependence and compositional effects. CPDAG arrows remain
observational equivalence-class estimates, not proven causal mechanisms. No GNN
was implemented or trained, and original graphs/results were not modified.
