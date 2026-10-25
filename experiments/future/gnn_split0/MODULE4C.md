# Module 4C: memory-bounded stable PC

## Scope and implementation

The original `scripts/causal/discover.py`, installed packages, existing PC/BOSS runs,
SCCVAE baseline and prepared inputs are unchanged. The new entry point is
`scripts/causal/discover_pc_memory.py`; its cache backend is `scripts/causal/pc_memory.py`.
Results are isolated under `graphs/causal_split0/pc_memory/<run-id>/`.
No full 279-variable experiment was launched during this module.

The input remains the prepared training-control matrix (10,691 cells x 279 genes).
The runner verifies its gene ordering against the correlation graph, source-row
identity, training partition and control labels. Seeded subsets use exactly the
original runner's selection order: choose columns, then rows, sort both. There is
no additional normalization, filtering, standardization or change to float64.

Scientific settings remain stable PC, Fisher-Z, alpha=0.05, uc_rule=0,
uc_priority=2 and unrestricted conditioning depth by default. `--max-depth` is
available but any resulting truncation is explicitly reported. Edge exports reuse
the original validated endpoint exporter, retaining unresolved directions and
conflict marks rather than arbitrarily orienting edges.

`MemoryFisherZ` **inherits the installed FisherZ.__call__ unchanged**. It only
replaces key formatting and memoization storage. Keys encode the canonical
unordered pair and sorted/deduplicated conditioning set as little-endian uint16
bytes. Their length encodes conditioning-set size; no probabilistic hashes or
lossy numeric representations are used. Values remain the original p-values.
This key format supports up to 65,536 variables; the present input has 279.

A configurable LRU holds at most `--cache-entries` CI entries (default 100,000);
three upstream metadata fields are retained separately. Zero disables memoization.
Evicted values are recomputed using identical upstream arithmetic. Cache counters,
live object accounting and checkpoint sizes are logged at each completed depth.
Supervised elapsed-time/memory heartbeats occur every two seconds, including
inside long conditioning depths. These heartbeats are not intra-depth checkpoints.

## Cache-growth diagnosis and alternatives

The legacy full-run log reports 83,320,945 cache entries (including metadata) after
depth 2, with a 3,475,001,062-byte checkpoint. Later interrupted attempts reported
23,932,661,760 bytes peak worker-tree memory. These are historical log/file-size
observations, not new full-run measurements; the huge pickle was NOT loaded.

The original string key already canonicalizes reversed pairs. Different
conditioning sets are distinct tests, not redundant scientific observations.
However, memoized values are not required algorithmic state. Earlier-depth tests
are not reused by later skeleton depths because conditioning-set cardinality
changes. In this configured uc_rule=0 path, orientation uses separating sets;
it does not require the entire historical p-value dictionary. Retaining all
depths also inflates pickle writes and graph deep copies during orientation.
The separating sets and stable-PC iteration logic are not pruned or modified.

Investigated choices:

| Choice | Finding |
| --- | --- |
| Bounded LRU | Implemented; predictable entry-count ceiling; may recompute evicted tests. |
| Compact keys alone | Not sufficient: controlled 100k-key accounting was 11,742,960 bytes vs 11,359,060 for strings, owing to Python mapping overhead/capacity. Shorter keys do not guarantee lower total memory. |
| No caching | Supported with zero entries and validated on a small input; may waste repeated pair tests. |
| Disk/SQLite memoization | Not implemented: potentially substantial per-test I/O and operational complexity; unnecessary for exact deterministic recomputation. |
| Float32 p-values, approximate CI, early exit after independence | Rejected: these could change scientific behavior/separating sets. |

The same controlled profile with a 1,000-entry bounded cache used 203,640 bytes
of live shallow object accounting after 100,000 insertions. This is NOT process
RSS, allocator-inclusive memory, or a full-PC benchmark.

## Measured validation

Authoritative small comparisons:
`graphs/causal_split0/memory_optimization/validation_v3/summary.json`.
Both implementations ran in separate processes on identical actual training-control
subsets, seed 20, 2,000 cells, no conditioning-depth cap. Both wrote depth checkpoints.
The optimized comparison deliberately uses **1,000 entries**, forcing eviction.
Times include discovery, orientation and depth-checkpoint I/O; they are single
measurements, not performance confidence intervals. Peak memory is Windows worker
peak working set, not system-wide RAM or GPU memory.

| Genes | Implementation | Seconds | Peak MiB | Final checkpoint bytes | CI cache entries | Edges |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | Original | 0.1019 | 158.71 | 385,711 | 1,664 | 26 |
| 20 | Bounded 1,000 | 0.1035 | 158.30 | 364,528 | 1,000 | 26 |
| 40 | Original | 4.1658 | 189.46 | 3,650,897 | 72,966 | 80 |
| 40 | Bounded 1,000 | 3.6042 | 160.62 | 736,984 | 1,000 | 80 |

For 40 genes: approximately 15.2% lower measured peak working set, 79.8% smaller
checkpoint, and 13.5% shorter measured runtime. There is no reliable speedup claim
from one trial. The 20-gene runtime was slightly slower. Endpoints and exact
separating-set structures match for BOTH sizes; natural termination depths were
6 and 12, respectively. Forty-gene endpoints contain 75 directed and 5 undirected
edges, no isolates, one component, density 0.102564, max degree 13.

At 40 genes, 74,428 CI calls produced 73,206 computations with the bounded cache
versus 72,966 distinct computations under unbounded caching: 240 additional
recomputations (~0.33%). Final live cache accounting was 174,764 vs 8,607,967 bytes;
the depth logs also retain pre-orientation sizes (deepcopy can resize mappings).

Additional checks: 72 exactly equal native/optimized p-values, 3,696 collision-free
canonical keys, zero-cache and two-entry-cache endpoint/separating-set equivalence,
forced eviction, and exact recovery after a depth-boundary interruption. A stray
partial temporary pickle is ignored; the valid atomically replaced checkpoint is
used. The inherited Fisher-Z function identity is asserted in the test suite.

The actual supervisor was also deliberately timed out on a small 40-gene run,
then resumed **from completed depth 3** and finished at depth 12. Results:
`graphs/causal_split0/pc_memory/recovery_n40_cells2000_seed20_v2/`.
Its endpoint matrix matches the original 40-gene comparison exactly.

Default-cache smoke test:
`graphs/causal_split0/pc_memory/smoke_default_n40_cells2000_seed20_v3/`.
With the production default of 100,000 entries (no eviction on this small input),
it took 4.296 seconds, peaked at 197.35 MiB, and saved a 3,559,993-byte checkpoint.
Its endpoints also match. LRU overhead can INCREASE memory on small inputs below
the bound; the main benefit is preventing unlimited growth on large searches.
Completed-run overwrite protection was exercised and correctly refused recovery
into a completed run.

Development raw files remain intact. `validation_v1/` and `validation_v2/` are
earlier passing comparison snapshots. An initial supervisor smoke run
`smoke_n20_cells2000_seed20_v1/` failed AFTER graph/statistics export because of a
duplicate logging keyword; this was fixed and is not a scientific algorithm error.
That failed attempt is retained, not relabeled as completed.

## Recovery, outputs and manual execution

Run/resume commands must repeat the same nodes, cells, seed, alpha, depth and
cache limit. Input, source-code, CI/orientation-package, version, Python and thread
environment fingerprints are checked before resuming. Completed runs cannot be
overwritten; active supervisors/workers are protected against concurrent writers.
Use only trusted local pickle checkpoints.

At each **completed stable-PC conditioning depth**, an atomic pickle stores the
graph, separating sets, CI state/correlation matrix, bounded cache and depth.
`--resume` restores that genuine state and skips completed depths. An interrupted
depth repeats; orientation restarts. `--restart-stage` instead starts the algorithm
over using the saved subset/seed. Legacy unbounded checkpoints are deliberately
not converted or consumed: start the memory-bounded runner with a new run ID.

Completed artifacts include `endpoint_matrix.npy`, sparse skeleton/directed/
undirected adjacency matrices, `edges.csv`, `node_order.csv`, `statistics.json`,
`config.json`, `provenance.json`, `status.json`, `worker.log`, `progress.log`, selected
row/column/source-row arrays and `pc_depth_checkpoint.pkl`. No timeout applies
unless explicitly requested. Ctrl+C stops this runner's worker tree; it never
targets another experiment. Windows sleep/hibernation must be disabled separately
for overnight wall-clock progress; the script does not change power settings.

```powershell
Set-Location D:\ALI\sccvae
# NEW full run, launched manually only:
.\.venv\Scripts\python.exe -u scripts\causal\discover_pc_memory.py --run-id full_pc_memory_seed20 --nodes 279 --cells 10691 --seed 20 --alpha 0.05 --cache-entries 100000

# In another terminal:
Get-Content .\graphs\causal_split0\pc_memory\full_pc_memory_seed20\progress.log -Tail 20 -Wait

# After an interruption (not while the run is active):
.\.venv\Scripts\python.exe -u scripts\causal\discover_pc_memory.py --run-id full_pc_memory_seed20 --nodes 279 --cells 10691 --seed 20 --alpha 0.05 --cache-entries 100000 --resume

# Only if no valid completed-depth checkpoint exists; starts discovery over:
.\.venv\Scripts\python.exe -u scripts\causal\discover_pc_memory.py --run-id full_pc_memory_seed20 --nodes 279 --cells 10691 --seed 20 --alpha 0.05 --cache-entries 100000 --restart-stage

Get-Content .\graphs\causal_split0\pc_memory\full_pc_memory_seed20\status.json
Get-Content .\graphs\causal_split0\pc_memory\full_pc_memory_seed20\statistics.json
```

Re-run the lightweight validation using a NEW output folder:

```powershell
.\.venv\Scripts\python.exe scripts\causal\benchmark_pc_memory.py --output graphs\causal_split0\memory_optimization\my_validation_01
```

## Scientific limitations and readiness

No scientific differences were observed on the tested inputs. This is observational
linear-Gaussian conditional-independence discovery, not proof of interventional
causal directions. Markov/faithfulness, causal sufficiency and acyclicity assumptions
can fail under hidden confounding, compositional normalization, cell-state/batch
effects and measurement error. Inherited full-file median library-size normalization
followed by log1p is transductive preprocessing, not strictly inductive; graph
fitting still uses training controls only, never held-out perturbation responses.

The optimized runner is ready for a **manual full-scale feasibility run**, based
on exact small-case and recovery checks. Full 279-node equivalence, total RAM,
runtime and completion are UNVERIFIED. The cache is bounded, not the combinatorial
search or all separating-set/graph storage. A full run can still take days or
longer, and interruption during a long depth loses that depth's progress. No
unjustified overnight completion estimate is possible from these small tests.
