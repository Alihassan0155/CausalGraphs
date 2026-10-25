# GES genuine recovery update

`discover_ges.py` now supports `--resume` using `ges_checkpoint.pkl`, not the old
graph-only snapshots. Existing live processes and historical results are untouched.
The previous MODULE4C_GES.md recovery limitation describes the old implementation;
this document supersedes it for newly launched runs using the updated code.

Checkpoints atomically flush/fsync and replace at every candidate-scan boundary:
initial forward state, after every accepted forward/backward move, and backward
phase entry. Saved state includes CPDAG, current/new score, phase, counters,
operator histories, graph histories, both score caches, scoring object/input and
random states. The next scan's candidate enumeration is recomputed, never skipped.
If interrupted inside a candidate scan, that scan repeats; completed moves remain.
There is no intra-scan recovery. Writes can add overhead as histories/caches grow.
Only trusted local pickle checkpoints may be loaded.

Configuration/input/software/source fingerprints must match. Completed/active run
IDs remain protected. Old snapshot files cannot be loaded as genuine checkpoints.
Use a NEW run ID for the first run after this update; the old full_ges_seed20
configuration fingerprints differ. This preserves all old results and configs.

Validation: `scripts/causal/test_ges_recovery.py` and
`graphs/causal_split0/ges/recovery_validation_v1/validation.json`: forced interruption
after forward move 2 and at backward entry, then checkpoint reload. Native vs
resumed graph endpoints, score, forward/backward update sequences and all returned
graph histories match exactly. Partial temporary checkpoint is ignored; incorrect
configuration fingerprint rejected. Small tests only, no full experiment launched.

New run (manual; stop any old process yourself if desired):

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges.py --run-id full_ges_resumable_seed20 --nodes 279 --cells 10691 --seed 20 --bic-penalty 2
```

After interruption, same settings plus --resume:

```powershell
Set-Location D:\ALI\sccvae
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u scripts\causal\discover_ges.py --run-id full_ges_resumable_seed20 --nodes 279 --cells 10691 --seed 20 --bic-penalty 2 --resume
```

Monitor checkpoint_saved and accepted_move events:

```powershell
Get-Content D:\ALI\sccvae\graphs\causal_split0\ges\full_ges_resumable_seed20\progress.log -Tail 20 -Wait
```

Scientific settings and original upstream search logic are unchanged; state is
restored around original forward/backward loops. No PC/BOSS artifacts were changed.
