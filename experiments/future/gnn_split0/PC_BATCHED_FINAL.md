# Full-size batched CPU PC: verified legacy migration and manual continuation

## Result

**Actual legacy migration SUCCEEDED. Full discovery was NOT started.**
The new run is `graphs/causal_split0/pc_batched/full_pc_seed20_migrated/`, status
`migrated_ready`. Its verified checkpoint has depth 2 complete and resumes at
**depth 3**, not depth 0. Original code, source checkpoints, scientific settings,
datasets, baseline results and existing experiments are unchanged.

| Measurement | Actual result |
| --- | --- |
| Source checkpoint | `graphs/causal_split0/pc/full_pc_seed20/pc_depth_checkpoint.pkl` |
| Source bytes | 3,475,001,062 |
| Source SHA-256, identical before/after | `4424321e8f4fdc99b571276ba615aae8c0ac3296bcc10a13d77c1ab4c3c84abc` |
| CI entries discarded | 83,320,942 (plus three cache metadata fields) |
| Original / retained pickle memo entries | 334,145,225 / 861,454 |
| Stripped intermediate bytes | 33,233,010 |
| Final lean checkpoint bytes | 9,373,578 (8.94 MiB) |
| Completed / next conditioning depth | 2 / 3 |
| Cells x graph genes | 10,691 x 279 |
| Preserved skeleton edges / maximum degree | 7,639 / 154 |
| Converter elapsed / total migration elapsed | 3.066 s / 21.875 s |
| Converter peak Windows working set | 141.33 MiB |
| Full discovery launched | No |

The converter memory measurement covers its PowerShell/.NET process tree, not the
Python migration parent's peak. The 3.5-GB pickle was NEVER passed to pickle.load.
Only the 33.23-MB stripped intermediate was deserialized. After verification,
the final checkpoint was round-tripped and matched the extracted state exactly.

Detailed machine-readable evidence is in the new run's `migration_report.json`,
`stream_profile.json`, `provenance.json` and `migration_stream.log`. Source gene
ordering and selected row/column/source-row files were copied to NEW paths and
their SHA-256 values verified against the unchanged originals.

## Implementation and safety

New source files:

- `scripts/causal/PcCacheStripper.cs`: bounded streaming protocol-5 transcode.
- `scripts/causal/strip_pc_cache.ps1`: compile with built-in Windows Add-Type.
- `scripts/causal/migrate_pc_batched.py`: read-only source validation and migration.
- `scripts/causal/pc_batched.py`: CPU-only batched CI, lean checkpoints and PC hooks.
- `scripts/causal/discover_pc_batched.py`: production-sized manual CPU runner.
- `scripts/causal/test_pc_batched.py`, `test_pc_streaming.py`,
  `test_pc_batched_cli.py`: small scientific, converter and real CLI recovery checks.

No package installation, PyTorch import, CUDA initialization or modification of
installed causal-learn was required. The .NET compiler is already provided by
Windows PowerShell; generated assembly use is local to the converter process.

The converter scans the pickle without executing it or constructing CI p-value
objects. It preserves the empty cache root and reusable metadata/global atoms,
discards CI keys/values and **densely renumbers retained memo references**. Simply
skipping values while keeping original high memo indices could allocate a huge
unpickler memo table; this implementation avoids that trap. A verified fast path
handles the standard NumPy float64 cache record, with generic parsing at frame
boundaries and other encodings. Fast and generic results were byte-identical on
the tested fixtures.

Unsupported external references into discarded cache values, unreviewed source
producers, unknown opcodes/layouts, changed input/package fingerprints and existing
destinations fail closed. A 768-MiB converter working-set guard is enforced.
Before loading the stripped result, size and retained-memo guards are checked.
Only trusted local checkpoints are supported; the stripped checkpoint still
contains normal Python objects and must not be treated as safe untrusted input.

Migration additionally verifies original float64 expression values, identical
correlations, node ordering, dimensions, undirected pre-orientation skeleton,
separating sets, and depth/edge/degree consistency with source progress records.
The final graph/separating-set fingerprints are:

- Skeleton endpoints: `e8196d50703fc74662a42c9c9770bd773fc09a0e092fb00d888411cd9609c0cb`
- Separating sets: `8723f18457812150945a651cc2c57cc886897087885f3e8b2fc0185548dbfefe`

These fingerprints compare the extracted source state to the lean checkpoint
after save/reload. The original huge pickle was intentionally not materialized
for a second independent full-object comparison; exact stripping was validated
against ordinary deserialization on small examples instead.

## Scientific behavior

Settings are unchanged: PC-stable, Fisher-Z, alpha=0.05, float64, uc_rule=0,
uc_priority=2, no background knowledge and no conditioning-depth cap. Only the
existing training-control matrix enters discovery; graph gene ordering remains
the existing 279-node ordering, not SCCVAE's internal 512-coordinate indices.

The CPU implementation uses the same NumPy float64 inverse kernel for stacked
matrices as the native scalar path, retaining native scalar sqrt/log expression
order and the original clipping and SciPy normal CDF. Identical queries are merged
only within each bounded batch. No approximations, ridge terms, pseudoinverses,
precision changes, early independence exits or separating-set pruning are used.

The original installed skeleton loop is instrumented in memory, with exact source
anchors and source fingerprints. A bounded query stream asserts correspondence
with the original consumer's (x,y,S) ordering. The stable graph remains frozen
throughout each depth. Duplicate deferred removals use a set rather than a large
list; original final unique-removal semantics and separating-set accumulation
are retained. Original UCSepset/Meek orientation code is reused unchanged.

Input, Python, software versions, BLAS thread environment, source and CI/orientation
fingerprints are enforced during resume. `--audit-ci` is available for independent
native-query checking, but must be selected when creating/migrating a run and kept
unchanged. The actual migrated run has audit disabled after exact small-case
validation; adding that flag to its resume command would correctly be rejected.

## Validation results

Authoritative outputs:

- `graphs/causal_split0/pc_batched_validation/validation_v1/summary.json`
- `graphs/causal_split0/pc_batched_validation/streaming_validation_v2/summary.json`
- `graphs/causal_split0/pc_batched/cli_validation_n40_seed20/cli_validation.json`
- `graphs/causal_split0/pc_batched/migration_check_n20_seed20/`

All cases used actual training controls, 2,000 cells, seed 20 and identical
subsets. Native reference versus batched CPU:

| Genes | Conditioning | Native p-values audited | Nonidentical p-values | Decision / endpoint / separating-set differences |
| --- | --- | --- | --- | --- |
| 20 | Natural completion, depth 6 | 1,884 | 0 | 0 / 0 / 0 |
| 40 | Natural completion, depth 12 | 74,428 | 0 | 0 / 0 / 0 |
| 80 | Explicit depth-2 feasibility cap | 97,625 | 0 | 0 / 0 / 0 |

Total initial-run comparisons: **173,937 bit-for-bit identical p-values**.
Additional post-migration/resume audits were also identical. Each small source
checkpoint was stopped after depth 1, stripped, compared to ordinary source
deserialization, migrated, and resumed from depth 2. Final endpoints and exact
separating-set structures matched native reference outputs. Original source
checkpoint hashes remained unchanged.

The optimized converter produced byte-identical output to the generic converter
for 20/40/80-node fixtures. A final one-item dictionary pickle batch (1,001 total
cache entries) was verified. A deliberately unsupported external CI-value alias
was rejected, not silently reconstructed incorrectly. Partial `.tmp` files do not
replace valid checkpoints. An unreviewed historical benchmark producer was also
correctly refused before output creation; the CLI migration fixture was then
created with the reviewed original runner rather than weakening provenance checks.

A newly created 40-node CPU CLI validation process was intentionally terminated
after completed depth 0, resumed from depth 1 and finished at depth 12 with native
endpoints, separating sets and identical audited p-values. Completed-run overwrite
protection passed. No pre-existing process was terminated.

The 80-node depth-capped case is not a full graph validation: its native and
batched oriented outputs both contain directed cycles, as previously documented
in the GPU pilot. This is an unchanged truncated equivalence fixture, not an
approved final DAG/CPDAG. No final 279-node output or full-run equivalence is claimed.

## Checkpointing, progress and recovery

After every complete stable-PC depth, the runner atomically writes, fsyncs and
replaces `pc_depth_checkpoint.pkl`. It includes graph/separating-set state,
completed depth, correlation/CI state, graph-state hashes and configuration hash.
The expression array is omitted from snapshots, then reattached only after exact
float64 input hashing and correlation checks on resume. This removes repeated
expression-array duplication without changing any expression values.

There is no historical CI cache. CI counters since migration are checkpointed;
interrupted-depth counters/state are rolled back with that depth. They are not
the number of lifetime legacy computations, and the discarded cache entry count
must not be mistaken for a legacy query count. Attempt/cumulative runtime and
progress history remain in logs/status. Every five seconds a heartbeat reports
memory, current depth/node pair and CI counts, even within a long depth.

`--resume` is genuine recovery from the latest COMPLETE depth. An interrupted
depth repeats from its start; there is no sub-depth checkpoint. Orientation is
restarted after skeleton completion. Ctrl+C records interruption; abrupt process
death leaves the previous atomically replaced checkpoint usable. Completed
experiments and active run IDs cannot be overwritten. Source/settings changes
are rejected rather than silently altering a resumed experiment.

The production runner has unlimited runtime, unless externally interrupted.
The converter itself is a linear restartable stage: an interrupted conversion
must use a NEW destination. That does not start PC search or damage its source.

## Manual commands: continue the actual 279-gene run from depth 3

No environment activation is necessary when invoking its Python executable.
These thread settings match the verified configuration and apply only to the
terminal process and its children, not global Windows settings.

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_pc_batched.py --run-id full_pc_seed20_migrated --nodes 279 --cells 10691 --seed 20 --batch-size 4096 --resume
```

Monitor in a separate PowerShell terminal:

```powershell
Set-Location D:\ALI\sccvae
Get-Content .\graphs\causal_split0\pc_batched\full_pc_seed20_migrated\progress.log -Tail 20 -Wait
```

If interrupted, use the SAME launch command above again. Do not use
`--restart-stage`, which deliberately starts from zero, and do not point the new
runner at the original unbounded checkpoint.

Status, migration proof and final results:

```powershell
Get-Content .\graphs\causal_split0\pc_batched\full_pc_seed20_migrated\status.json
Get-Content .\graphs\causal_split0\pc_batched\full_pc_seed20_migrated\migration_report.json
# This exists after full discovery and orientation actually finish:
Get-Content .\graphs\causal_split0\pc_batched\full_pc_seed20_migrated\statistics.json
```

Final exports also include endpoint matrices, directed/undirected/skeleton sparse
adjacency, edge list and the preserved node ordering. Until a manual run finishes,
the migrated checkpoint is only an intermediate undirected skeleton.

Repeat validations using NEW output/run identifiers:

```powershell
.\.venv\Scripts\python.exe scripts\causal\test_pc_batched.py --output graphs\causal_split0\pc_batched_validation\my_validation_01
.\.venv\Scripts\python.exe scripts\causal\test_pc_batched_cli.py --run-id my_cli_validation_01
```

## Remaining limitations

Cache growth and duplicate removal storage are bounded, not PC's combinatorial
search. Deep conditioning can still take a very long time, and float64 batch
matrix memory scales with batch_size x (depth+2)^2. Total full-run RAM and runtime
have not been measured; no overnight completion guarantee is justified. Keep
Windows awake for progress and preserve the configured batch size on resume.

The inherited normalize_total library-size statistic was computed over the whole
file across all 8,563 genes, followed by log1p before graph-node selection. This
preprocessing is transductive, not strictly inductive; fitting here still uses
training controls only, never held-out perturbation outcomes. Observational PC
requires Gaussian/linear CI, causal Markov/faithfulness, sufficiency and acyclicity
assumptions. Hidden confounding, compositional effects, cell state/batch dependence
and measurement error limit causal interpretation. No baseline, PC/BOSS historical
result, GNN implementation or training procedure was modified.
