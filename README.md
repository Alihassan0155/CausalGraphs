# SCCVAE research workspace

Windows / Python 3.12 / CUDA 12.4. Original SCCVAE source and preprocessing remain in `sc_causal/`; see [upstream instructions](docs/UPSTREAM_README.md).

```text
sc_causal/              Original source, GEARS baseline, gene mapping, author tables
configs/baseline/       Protocol index pointing to the actual saved run configuration
scripts/setup/          Environment installation and import verification
scripts/baseline/       Training launcher, smoke test, evaluation and audit utilities
scripts/data/           Resumable split-0 preparation, delta reader and validation
scripts/causal/         Manual PC/BOSS discovery, bounded benchmarks and validation
experiments/baseline/   Baseline run index and artifact locations
experiments/future/     Reserved documentation for future, separate experiments
data/                  Local original datasets and archives (ignored by Git)
processed/             Generated preprocessing outputs (ignored)
graphs/                Generated graph artifacts (ignored)
results/sccvae_baseline/ Historical runs, checkpoints, logs, metrics and audits
docs/                  Original upstream documentation
.venv/                 Local Python environment (ignored)
requirements.txt       Pinned direct dependencies
```

Run commands from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python .\scripts\setup\verify_setup.py
# Installation only, when needed:
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup\setup.ps1
```

The completed K562 OOD split-0 run is indexed in [experiments/baseline](experiments/baseline/README.md). Its historical artifact paths and contents are preserved. `sc_causal/h5ad_datafiles` is a Windows junction to `data/h5ad_datafiles`; keep it for original data loading.

The baseline launcher is now `scripts/baseline/start_fresh_baseline.ps1`. It **starts new training** when explicitly run and writes to `results/sccvae_baseline/runs/fresh_<timestamp>/`, with `logs/`, `provenance/`, and the original `outs/ood_split_0_full/` checkpoint/config layout. It does not resume an old run. Nothing trains on setup or import verification.

Evaluation/history/report/audit scripts under `scripts/baseline/` target the completed run and may write derived artifacts when executed; do not rerun them over frozen results casually. Existing experiment results were not regenerated during organization. Read the discrepancy audit before comparisons: retain learned-shift and test-selected results separately; preprocessing is transductive.

Future experiment definitions belong under `experiments/future/<experiment>/`, configurations under `configs/<experiment>/`, and outputs under `results/<experiment>/`. Keep them separate from `results/sccvae_baseline/`.

Modules 2/3 preparation and correlation graph are documented in [experiments/future/gnn_split0](experiments/future/gnn_split0/README.md). Generated data are in `processed/gnn_split0/` and `graphs/correlation_split0/`; these commands do not implement or train a GNN.

Module 4A PC/BOSS runners and recovery semantics are documented in [MODULE4A.md](experiments/future/gnn_split0/MODULE4A.md). Artifacts are separate under `graphs/causal_split0/`. Full 279-node searches are manual only; the benchmark script runs bounded subsets.
