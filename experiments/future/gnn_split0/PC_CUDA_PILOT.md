# CUDA PC pilot: measured feasibility and scientific validation

## Outcome

The isolated pilot implements streamed, batched float64 Fisher-Z matrix inversions
on the NVIDIA RTX A4000. PC-stable, alpha=0.05, uc_rule=0, uc_priority=2, separating
sets and endpoint orientation logic are preserved. On tested subsets all native
CPU/GPU independence decisions, endpoint matrices and separating-set structures
matched. P-values are NOT bitwise identical across hardware.

CUDA beat the scalar CPU reference, but **batched CPU was faster than CUDA** on
both tested workloads. The evidence supports batching/search-memory optimization
more strongly than GPU acceleration for this scale. Strict verification of every
GPU decision largely eliminates the runtime advantage. This is a validated small
pilot, NOT certification of full 279-node CUDA discovery or legacy migration.

No existing scripts, dependencies, datasets, checkpoints or experiments were
modified. No full 279-node search was launched. Only newly created pilot test
processes were deliberately interrupted for recovery validation.

## Files and output layout

- `scripts/causal/pc_cuda.py`: bounded query streaming, numerical backend,
  runtime instrumentation of the original skeleton loop, small-state migration.
- `scripts/causal/discover_pc_cuda.py`: manual pilot runner; deliberately limited
  to at most 80 nodes, no automatic timeout, true completed-depth recovery.
- `scripts/causal/benchmark_pc_cuda.py`: bounded 40/80-node comparisons and
  small legacy-checkpoint migration test (90-second per-child test budget).
- `scripts/causal/test_pc_cuda_numerics.py`: synthetic threshold/conditioning tests.
- `scripts/causal/test_pc_cuda_resume.py`: actual CLI interruption/recovery test.
- `graphs/causal_split0/pc_cuda_pilot/pilot_v1_report/summary.json` and
  `pilot_v2_report/summary.json`: two independent comparison rounds with timings,
  graph statistics, numerical audit, driver/version and source fingerprints.
- `graphs/causal_split0/pc_cuda_pilot/numerical_stress_v1.json`: stress results.
- `graphs/causal_split0/pc_cuda_pilot/cli_resume_v1_n40/resume_validation.json`:
  actual process interruption/recovery and completed-run protection results.

Each individual run has configuration, provenance, selected row/column/source-row
arrays, node order, progress/status logs, depth checkpoint, exact endpoint matrix,
sparse directed/undirected/skeleton matrices, edge list and statistics.
Per-attempt CI counters reset on resume; completed-depth events preserve history.

## Hardware and software

Measured 2026-10-22 on Windows: NVIDIA RTX A4000, 16,376 MiB reported VRAM,
driver 596.71, PyTorch 2.4.1+cu124, bundled CUDA runtime 12.4. `nvidia-smi` reports
driver CUDA compatibility 13.2; that is NOT the PyTorch runtime version.
CUDA availability and float64 inversion were verified on hardware. Roughly
15.9 billion bytes were available during the small GPU tests.

Existing isolated causal-learn 0.1.4.8, NumPy 1.26.4, SciPy 1.14.1, pandas 2.2.3,
psutil 7.2.2 and networkx 3.6.1 were used without installations or upgrades.
Default OMP/OPENBLAS/MKL thread limits are 1 unless already set by the environment;
their actual values, package/source/input fingerprints and Git HEAD are recorded.

## Implementation and scientific invariants

Input: `processed/gnn_split0/training_control_graph_expression.npy`,
10,691 training-control cells x 279 mapped genes, cast prepared float32 to float64
exactly as in the CPU runner. Input row identity, training-control status and gene
ordering are verified against existing metadata. No held-out response enters
discovery, feature selection or benchmark tuning. Subsets use seed 20, uniformly
sample columns then rows without replacement, sorting indices to retain ordering.
All comparisons use 2,000 identical cells and identical columns within each size.

The installed skeleton function is instrumented in memory, not edited on disk.
Its source anchors and upstream fingerprints are checked. At each stable depth,
a second iterator enumerates exactly the same (x,y,S) queries in the same order.
The original consumer loop asserts correspondence for every query. The graph
remains frozen throughout the depth, as required by PC-stable. Results are never
short-circuited after the first independent set; all original separating-set
unions/appends occur in the original code, including reverse-pair effects.

For CUDA and batched-CPU paths, deferred edge-removal storage is a set instead of
a duplicate-filled list. Stable PC already applies unique removals at the end of
the depth; this retains membership and final removal semantics while bounding
that storage by edge count. The scalar CPU reference retains the original list.
Orientation is the original UCSepset/Meek implementation, not a GPU rewrite.

Only a batch of at most 4,096 queries is held, with exact within-batch duplicate
merging. There is no historical GPU CI cache, unbounded key list or per-test GPU
allocation retained in checkpoints. Correlations are computed ON CPU with the
original `np.corrcoef(data.T)` for numerical consistency, then copied once to CUDA.
Submatrix gathering, inverse and partial-correlation arithmetic run in torch
float64 on CUDA. Original clipping, scalar math.log/math.sqrt transform and SciPy
normal CDF remain on CPU. TF32 is disabled; AMP, float32 inversions, ridge terms,
pseudoinverses and approximate CI tests are not used.

The extra `cpu-batch` control uses the same streamed queries, removal set and
vectorized p-value handling, but batched NumPy inversions. It separates batching
benefits from CUDA benefits. All paths include orientation and checkpoint I/O.

`--audit-ci` recomputes EVERY query with the native scalar Fisher-Z arithmetic and
stops on a returned independence-decision mismatch. Independently of that flag,
raw p-values within 1e-8 of alpha are explicitly rechecked on the native CPU path;
these rechecks are counted, not hidden. The heuristic guard alone is not a proof
of equivalence for arbitrarily ill-conditioned inputs. Use `--audit-ci` for strict
manual pilot validation. No CPU backend is silently selected when CUDA is absent.
Singular/nonfinite results fail rather than being regularized.

## Measured performance

Two rounds, separate process for each variant, seed 20 and batch size 4,096.
The following are means of TWO observations, not statistical confidence bounds.
40 nodes ran to natural completion at depth 12. 80 nodes were deliberately capped
at depth 2: this is a **truncated feasibility comparison**, not a completed graph.

Process end-to-end time is measured from launch through process exit, including
imports, input checks, CUDA setup, search, orientation, checkpoints and exports.
Both CPU and CUDA variants share the same pilot import envelope (including torch
import); these are not timings of an independent minimal-import CPU application.
Algorithm time starts after input/configuration setup, includes CUDA initialization
within the engine and the remaining search/checkpoint/export work.

| Nodes | Variant | Mean process seconds | Mean algorithm seconds | Maximum peak host MiB | Peak CUDA allocated/reserved MiB |
| --- | --- | --- | --- | --- | --- |
| 40 | Native scalar CPU | 7.254 | 4.361 | 526.80 | not used |
| 40 | Batched CPU | 3.316 | 0.487 | 512.16 | not used |
| 40 | CUDA, audit off | 3.769 | 0.868 | 787.77 | 18.06 / 24 |
| 40 | CUDA, full native audit | 6.763 | 3.954 | 787.92 | 18.06 / 24 |
| 80, depth <=2 | Native scalar CPU | 11.761 | 8.955 | 538.50 | not used |
| 80, depth <=2 | Batched CPU | 7.578 | 4.794 | 512.52 | not used |
| 80, depth <=2 | CUDA, audit off | 7.752 | 4.896 | 790.69 | 9.94 / 22 |
| 80, depth <=2 | CUDA, full native audit | 12.150 | 9.262 | 791.01 | 9.94 / 22 |

Audit-off CUDA process speedup versus scalar CPU: **1.92x (40), 1.52x (80)**.
Against batched CPU CUDA is approximately **14% slower (40), 2% slower (80)**.
With strict native audit: approximately 1.07x vs scalar CPU at 40 and 3% slower
at 80. CPU batching alone achieved approximately 2.19x / 1.55x process speedups.
CUDA setup/small-matrix overhead is a plausible explanation for the CPU-batch
advantage; no kernel-level profiler attribution is claimed.

Host measurements are Windows peak working set. CUDA allocated/reserved figures
are PyTorch allocator peaks, NOT the complete driver/context/display VRAM cost.
Remaining/total device memory is separately recorded; other desktop applications
also consume VRAM. CUDA had higher host RAM than the CPU controls in these pilots.

## Output and numerical equivalence

All four variants in BOTH rounds had identical full endpoint matrices and exact
separating-set structure hashes, with identical input fingerprints. Strict GPU
audit compared every queried p-value against native Fisher-Z:

| Subset | Queries audited per round | Max absolute p difference | Raw/returned decision mismatches |
| --- | --- | --- | --- |
| 40 | 74,428 | 1.3322676295501878e-15 | 0 / 0 |
| 80, depth <=2 | 97,625 | 3.552713678800501e-15 | 0 / 0 |

Closest raw p-value distances to alpha were approximately 1.086e-5 and 5.532e-6;
no threshold rechecks were needed for these data subsets.

40-node graph: 80 edges (75 directed, 5 undirected), density 0.102564, no isolates,
one component, directed part acyclic. 80-node depth-truncated graph: 303 edges
(299 directed, 4 undirected), density 0.095886, no isolates, one component.
**The 80-node directed part contains cycles in BOTH CPU and GPU output.** It is
not certified as a valid final CPDAG/DAG; it is retained unchanged as a truncated
upstream-PC equivalence result. No directions were repaired to improve appearance.

Synthetic stress: a constructed p-value within 2.64e-16 of alpha triggered the
explicit CPU threshold safeguard and returned the exact native value. Nearly
collinear covariates with correlation condition number ~4.01e12 produced p-value
differences up to 2.236e-6 (no decision changes in those two stress queries).
At condition number ~1.08e16 strict validation stopped on a native singular-matrix
error. This demonstrates why tiny observed errors on ordinary pilot subsets
cannot justify an unconditional full-graph equivalence claim.

PyTorch also cautions that CPU, GPU and batched operations need not be bitwise
identical: [official numerical-accuracy documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
That general guidance is not a claim that a newer torch version was installed;
the measured installation remains 2.4.1+cu124.

## Checkpoints and migration

Checkpoints are atomically written after each COMPLETE stable conditioning depth.
They contain CPU-portable graph/separating-set/CI/correlation state, completed
depth and pilot format tag. No CUDA tensor, transient batch or engine is pickled.
`--resume` genuinely skips completed depths; a partial depth repeats and orientation
restarts. Configuration/source/input fingerprints must match; completed outputs
and active run IDs are protected. Only trusted local pickles should be loaded.

Actual CLI recovery test: a newly created strict 40-node GPU process was terminated
after depth 0, resumed from depth 1, completed at depth 12 and matched the native
reference endpoints and separating sets exactly. Completed-run overwrite protection
was verified afterward. No pre-existing process was stopped.

Small legacy migration test (twice): the original CPU runner was interrupted after
depth 1; a NEW checkpoint preserved its skeleton and separating sets, replaced only
the memoization backend, verified identical stored input/correlation arrays, then
resumed on GPU at depth 2. Both tests reached depth 12 with exact native reference
endpoints and separating sets and no changed source checkpoint. Historical CI
cache values are unnecessary; the historical skeleton/separating sets are retained.

Actual legacy full checkpoint investigation, READ ONLY:

- Source: `graphs/causal_split0/pc/full_pc_seed20/pc_depth_checkpoint.pkl`.
- File size: 3,475,001,062 bytes. Source log reports depth **2 completed**, not 3,
  83,320,945 dictionary entries including metadata, maximum degree 154 and
  7,639 skeleton edges. Successful migration should therefore continue at depth 3.
- Full input hashes, float64 cast, stable/alpha/uc settings and package fingerprints
  match the intended pilot methodology. The checkpoint content itself was NOT
  deserialized, so its actual internal consistency/corruption status is unverified.
- Loading an ordinary pickle reconstructs its enormous dictionary BEFORE a
  subsequent cache.clear(); the historical process reached ~23.9 billion bytes.
  Clearing after loading does not make loading safe on this 32-GB machine.
- The small-state migration primitive is structurally validated, but there is NO
  automatic large-checkpoint converter/CLI and the actual legacy checkpoint has
  NOT been migrated. Safe conversion needs an isolated sufficiently high-RAM host
  (64 GB is a reasonable starting point, not a measured guarantee) or a separately
  verified streaming converter that also handles pickle memo references correctly.
- Any migration must preserve the PRE-ORIENTATION skeleton checkpoint, gene order,
  input/source fingerprints, completed depth and separating sets, and write to a
  NEW destination. A graph endpoint export after orientation is not a substitute.

Consequently there is no scientifically certified command to resume the actual
279-node legacy checkpoint on GPU yet. The pilot deliberately rejects >80 nodes.

## Exact manual PowerShell commands

Validated unrestricted 40-node pilot, strict native decision auditing, no timeout:

```powershell
Set-Location D:\ALI\sccvae
.\.venv\Scripts\python.exe -u scripts\causal\discover_pc_cuda.py --run-id manual_cuda_n40_seed20 --backend cuda --nodes 40 --cells 2000 --seed 20 --batch-size 4096 --audit-ci
```

Monitoring from another terminal:

```powershell
Set-Location D:\ALI\sccvae
Get-Content .\graphs\causal_split0\pc_cuda_pilot\manual_cuda_n40_seed20\progress.log -Tail 20 -Wait
```

After an interruption (not while active), same settings and run ID:

```powershell
Set-Location D:\ALI\sccvae
.\.venv\Scripts\python.exe -u scripts\causal\discover_pc_cuda.py --run-id manual_cuda_n40_seed20 --backend cuda --nodes 40 --cells 2000 --seed 20 --batch-size 4096 --audit-ci --resume
```

If interrupted before ANY complete depth checkpoint, explicitly restart instead:

```powershell
.\.venv\Scripts\python.exe -u scripts\causal\discover_pc_cuda.py --run-id manual_cuda_n40_seed20 --backend cuda --nodes 40 --cells 2000 --seed 20 --batch-size 4096 --audit-ci --restart-stage
```

Read saved outputs:

```powershell
Get-Content .\graphs\causal_split0\pc_cuda_pilot\manual_cuda_n40_seed20\status.json
Get-Content .\graphs\causal_split0\pc_cuda_pilot\manual_cuda_n40_seed20\statistics.json
```

Optional matched batched-CPU comparison, explicitly selected rather than fallback:

```powershell
.\.venv\Scripts\python.exe -u scripts\causal\discover_pc_cuda.py --run-id manual_cpu_batch_n40_seed20 --backend cpu-batch --nodes 40 --cells 2000 --seed 20 --batch-size 4096
```

Repeat the bounded benchmark suite with a NEW prefix:

```powershell
.\.venv\Scripts\python.exe scripts\causal\benchmark_pc_cuda.py --run-prefix my_gpu_pilot_01
```

Do not interpret these 40-node commands as a full 279-gene discovery. Audit-off
timings above correspond to omitting `--audit-ci`, but strict scientific validation
commands deliberately include it. Keep Windows awake for manual long runs.

## Limitations and recommendation

The inherited preprocessing used normalize_total across all 8,563 genes with a
whole-file library-size statistic, then log1p before graph-node selection. It is
transductive preprocessing, not strictly inductive. No test perturbation outcome
was used here. Observational Fisher-Z PC depends on linear-Gaussian CI, causal
Markov/faithfulness, sufficiency and acyclicity assumptions; latent confounding,
normalization/compositional effects, cell-state/batch dependence and measurement
error limit causal interpretation.

Keep CUDA as an optional research pilot, not the default production choice yet.
Prefer further evaluation of CPU batching, which was fastest on these workloads.
Full 279-node scientific equivalence, runtime, RAM, conditioning stability and
actual legacy-state migration remain unverified. No trustworthy full-runtime
estimate or production-readiness claim follows from these small tests.
