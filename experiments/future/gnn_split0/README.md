# Modules 2 and 3: split-0 preparation and correlation graph

Configuration: `configs/gnn_split0/preparation.json`.

Run from the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -u scripts\data\prepare_gnn_split0.py
.\.venv\Scripts\python.exe scripts\data\validate_gnn_split0.py
```

The first command resumes normalization after interruption using `processed/gnn_split0/progress.json` and a flushed partial HDF5 file. Changed inputs/configuration reject reuse. Completed normalized expression is reused; derived summaries/graph are rebuilt deterministically. No PC, BOSS, GNN, or training is implemented.

Outputs: `processed/gnn_split0/` and `graphs/correlation_split0/`. Read `processed/gnn_split0/methodology_report.md` for target definitions, counts, provenance, exclusions and leakage limitations. `scripts/data/gnn_split0_reader.py` yields per-cell deltas without a second full expression matrix. The default reader retains all mapped test targets; use `baseline_compatible=True` for the documented 54-target retained-cell comparison. All 8563 genes remain outputs.

The requested baseline freeze document was absent at preparation time; the existing discrepancy audit supplies the eligibility and normalization limitations. Original whole-file normalization is transductive. Only training non-targeting rows enter Pearson correlation; the global normalization statistic still includes held-out cells. Threshold 0.30 is provisional and unoptimized.

See [MODULE4A.md](MODULE4A.md) for PC/BOSS discovery, small benchmarks, manual PowerShell commands and recovery limitations. Any future GNN training implementation must provide real checkpoints, explicit recovery semantics, and manual terminal commands; no GNN training is implemented here.

See [MODULE4C.md](MODULE4C.md) for the separate memory-bounded PC runner, measured cache/RAM comparisons, exact-output validation, checkpoint recovery and manual full-run commands. Existing PC/BOSS runners and experiment artifacts are preserved.
