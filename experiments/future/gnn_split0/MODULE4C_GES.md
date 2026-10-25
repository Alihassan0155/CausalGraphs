# Module 4C — genuine GES feasibility

Implemented separately in `scripts/causal/discover_ges.py`, using installed
causal-learn 0.1.4.8 `search/ScoreBased/GES.py`. No original PC/BOSS/SCCVAE or
dataset files were edited. No 279-node discovery was launched.

## Method and reproducibility

GES greedily inserts then deletes valid CPDAG operators, unlike BOSS's order
search. Both use the same Gaussian covariance SEM-BIC local score and float64
input, with covariance ddof=0. Explicit lambda_value=2 matches the completed BOSS
experiment. The maximized local score is
`-n/2*(1+log(residual_variance)) - lambda*(parent_count+1)*log(n)`.
Conventional BIC in this library is lambda=0.5, not 2; these experiments therefore
use a stronger, BOSS-matched penalty and must be labeled accordingly. No penalty
was selected using test responses. maxP defaults to the number of variables,
matching unrestricted upstream behavior. Seed 20 and sorted seeded subsets are
recorded; core GES is deterministic subject to arithmetic and tie ordering.

Input is exactly the BOSS training-control matrix, 10,691 x 279, prepared float32
cast to float64 without further normalization or standardization. Mapping,
source-row identity, gene symbols, Ensembl IDs and expression columns are
verified. No direct held-out perturbation responses enter discovery.

The original scorer uses a pseudoinverse on singular parent covariance and clips
nonpositive residual variance to machine epsilon. This is inherited upstream
behavior, not an added regularizer; near-deterministic relations can affect results.
Saved configurations fingerprint input, software, runner, score, graph conversion
and GES sources and thread settings. Provenance records Git HEAD and command.

## Lightweight feasibility results

All cases used 2,000 training-control cells, seed 20, lambda=2, with a 90-second
external per-run benchmark budget. None timed out. The manual production runner
has no timeout. Times are single measurements, not scaling guarantees.

| Genes | Algorithm seconds | Directed | Unresolved | Edges | Isolates | Components | Peak process-tree bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.062 | 2 | 0 | 2 | 17 | 18 | 197,672,960 |
| 40 | 0.187 | 0 | 7 | 7 | 34 | 35 | 197,529,600 |
| 80 | 2.922 | 30 | 5 | 35 | 59 | 62 | 197,816,320 |

All passed exact endpoint/edge-list/sparse-adjacency reconstruction, retained
gene ordering, acyclic directed parts, chordal undirected chain components,
acyclic chain-component quotients and DAG-extension-to-CPDAG roundtrips.
PyTorch int64 edge_index plus unresolved relation labels roundtrip exactly;
PyG itself is not installed, so Data.validate is not claimed. Explicit num_nodes
preserves isolates. Do not interpret reciprocal unresolved message arcs as two
causal arrows or invent orientations.

Native versus instrumented GES on a small synthetic case gave identical endpoint
matrices, scores and forward/backward update sequences. The score formula was
independently checked against centered least-squares residual variance.

Full 279-node manual feasibility is reasonable to attempt, but neither runtime
nor RAM is predictable from these sparse small subsets. The full sample size
can support denser graphs and increase candidate-set enumeration substantially.
Search caches and upstream graph history may grow; no bounded-memory guarantee
or overnight completion estimate is asserted.

Evidence: `graphs/causal_split0/ges/feasibility.json`,
`instrumentation_validation.json`, individual `bench_*/validation.json` and logs.

## Recovery and exports

Runtime source instrumentation adds only accepted-move logging/snapshots, retaining
the installed algorithm. Five-second heartbeats report elapsed time and RSS.
Snapshots are atomic, per attempt, and contain a CPDAG and phase/score/move count.
They are NOT resumable algorithm-state checkpoints: candidate/local-score caches,
full loop state and histories are not captured. Upstream has no supported resume
interface. `--resume` explicitly errors; `--restart-stage` repeats the search from
an empty graph using verified saved inputs/configuration. Interruptions preserve
logs/snapshots, completed runs and active IDs are protected. Only trusted local
pickles should be loaded.

Final outputs under `graphs/causal_split0/ges/<run-id>/` include endpoint matrix,
directed/undirected/skeleton sparse adjacency, edge CSV, node order, complete GES
record, configuration, source/input fingerprints, status, statistics and logs.
`validate_ges.py` validates final artifacts and writes a protected validation.json.
No final 279-node graph exists until the user runs it.

The validator compares full GES to full BOSS using skeleton overlap/Jaccard,
edges/density/isolates and shared-edge orientation agreement, reversal and
directed-vs-unresolved differences. Benchmark comparisons use induced full-BOSS
subgraphs, NOT matched subset discovery runs, and are explicitly labeled so.
Actual full GES-versus-BOSS conclusions remain pending, not fabricated.

## Manual PowerShell commands

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges.py --run-id full_ges_seed20 --nodes 279 --cells 10691 --seed 20 --bic-penalty 2
```

Monitor in another terminal:

```powershell
Get-Content D:\ALI\sccvae\graphs\causal_split0\ges\full_ges_seed20\progress.log -Tail 20 -Wait
```

After an interrupted process has stopped, restart the algorithm (NOT true resume):

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges.py --run-id full_ges_seed20 --nodes 279 --cells 10691 --seed 20 --bic-penalty 2 --restart-stage
```

After completion:

```powershell
.\.venv\Scripts\python.exe scripts\causal\validate_ges.py --run-id full_ges_seed20
Get-Content .\graphs\causal_split0\ges\full_ges_seed20\statistics.json
Get-Content .\graphs\causal_split0\ges\full_ges_seed20\validation.json
```

## Scientific limitations

Inherited SCCVAE preprocessing normalizes across all 8,563 genes using the whole
file's library-size statistic, then log1p before graph selection. This is
transductive preprocessing, not strictly inductive; graph fitting still uses only
training controls. Observational Gaussian SEM, causal Markov/faithfulness,
acyclicity and causal sufficiency assumptions may fail under latent confounding,
cell/batch dependence, compositional effects and measurement error. CPDAG arrows
are equivalence-class estimates, not interventionally established mechanisms.
No GNN has been implemented or trained.
