# Module 4A: observational PC and BOSS graphs

Input: `processed/gnn_split0/training_control_graph_expression.npy`, 10691 training non-targeting cells x 279 genes. The runner verifies the original gene order against both the mapping and correlation-graph ordering, records source-row indices, and fingerprints inputs, pinned implementation source, configuration and software. It never loads held-out perturbation expression. Float32 prepared input is cast to float64; no additional scaling, filtering or jitter is applied.

Dependencies are isolated in `.venv/module4a_packages`: causal-learn 0.1.4.8 and momentchi2 0.1.8. Module scripts explicitly prepend this path. Baseline causal-learn 0.1.3.8 and other baseline packages remain untouched. Installation, if needed:

```powershell
cd D:\ALI\sccvae
.\.venv\Scripts\python.exe -m pip install --no-deps --target .venv\module4a_packages -r configs\module4a\requirements.txt
```

PC uses upstream stable PC, Fisher-Z alpha=0.05, uc_rule=0 and uc_priority=2, with native max_k exposed as --max-depth (-1 means unrestricted). Depth-truncated results are explicitly labeled PDAGs and are not claimed to be unrestricted PC output. Conditioning-depth boundaries are instrumented for atomic serialization of the CausalGraph, separating sets, CI cache and completed depth. Synthetic tests and a real 40-node interrupted run verified exact endpoint-matrix equality after recovery. An interrupted depth is repeated; orientation is restarted. Checkpoints are local trusted pickle files and must not be loaded from unknown sources. Input/configuration/implementation fingerprints must match.

BOSS is genuine upstream Best Order Score Search with grow-shrink trees; it is not GES or another substitute. It uses covariance-based Gaussian SEM-BIC, maximizing -n/2*(1+log(residual_variance)) - lambda*(parents+1)*log(n). lambda_value=2 is explicitly used to match this version's BOSS default; conventional BIC corresponds to 0.5 in this parameterization. One seeded search is run until no order improvement; no automatic restarts or extra BES stage. Both the representative DAG before conversion and the final upstream CPDAG are saved. The representative DAG is not uniquely identified causal truth. No genuine BOSS mid-search state recovery is implemented/exposed; --restart-stage repeats its search from saved data subsets and seed. --resume is rejected for BOSS.

Run commands (manual; no full experiment runs automatically):

```powershell
.\.venv\Scripts\python.exe -u scripts\causal\discover.py --algorithm pc --run-id full_pc_seed20 --nodes 279 --cells 10691 --alpha 0.05 --max-depth -1 --seed 20
.\.venv\Scripts\python.exe -u scripts\causal\discover.py --algorithm boss --run-id full_boss_seed20 --nodes 279 --cells 10691 --bic-penalty 2 --seed 20
```

Monitor from another terminal:

```powershell
Get-Content graphs\causal_split0\pc\full_pc_seed20\progress.log -Tail 12 -Wait
# Ctrl+C stops this log viewer only. Ctrl+C in the discovery terminal interrupts discovery.
Get-Content graphs\causal_split0\boss\full_boss_seed20\worker.log -Tail 12 -Wait
```

Recover interrupted runs using exactly the original algorithm parameters; wall-time budget may change:

```powershell
.\.venv\Scripts\python.exe -u scripts\causal\discover.py --algorithm pc --run-id full_pc_seed20 --nodes 279 --cells 10691 --alpha 0.05 --max-depth -1 --seed 20 --resume
.\.venv\Scripts\python.exe -u scripts\causal\discover.py --algorithm boss --run-id full_boss_seed20 --nodes 279 --cells 10691 --bic-penalty 2 --seed 20 --restart-stage
```

If PC has not completed even depth 0, use --restart-stage instead of --resume. Completed runs are protected even with these flags; use a new run ID for a distinct experiment. Changing the depth cap is a new experimental configuration, not continuation of the same run. Full unrestricted PC is potentially impractical; an explicitly truncated pilot can use --max-depth 2 and a separate run ID, but is not interchangeable with full PC.

Omitting --time-limit-seconds runs until completion or manual interruption. An explicit 0 is also unlimited. Positive values enable an optional wall-time budget. This does not prevent Windows sleep, shutdown or power loss. Existing processes retain the arguments with which they started; a running 3600-second invocation must stop before it can be resumed without a timeout. The reviewed prior runner's checkpoints remain compatible with this supervisor-only update; algorithm-state, package, input and scientific-setting checks remain enforced. Saved original configurations are not rewritten; each attempt records the current runner fingerprint and time budget in status/progress logs.

Outputs live in `graphs/causal_split0/<algorithm>/<run-id>/`: config.json, provenance.json, selected_rows.npy, selected_columns.npy, selected_source_rows.npy, node_order.csv, progress.log, worker.log, status.json and, on completion, statistics.json and graph files. PC additionally has pc_depth_checkpoint.pkl; BOSS saves representative_dag_* files. Atomic checkpoints protect the last completed PC depth. The supervisor samples process-tree RAM, emits heartbeats, and stops the Windows venv interpreter tree at interruption/timeout. Polling can overshoot a timeout by approximately two seconds. Windows OS peak working-set measurements are included on completed workers; monitor sampling on other systems is approximate. Process memory excludes unrelated applications.

Endpoint matrix[i,j] is the endpoint at node i: tail=-1, arrow=1, circle=2 (additional native codes are preserved). Thus i -> j has matrix[i,j]=-1 and matrix[j,i]=1. directed_adjacency[i,j]=1 denotes i -> j. Skeleton and undirected adjacencies are separate; no undirected or circle endpoints are silently converted to directed edges. PC bidirected conflicts are labeled as conflicts, not inferred latent confounding. Directed cycles and conflicts are reported explicitly. A returned PC graph is a CPDAG candidate; this validation is not a formal proof of complete equivalence-class correctness.

Scientific limitations: observational data require causal Markov/faithfulness and causal sufficiency; linear-Gaussian assumptions motivate Fisher-Z/SEM-BIC. Hidden confounding, batch/cell-state structure, dependent cells, compositional normalization, measurement error and finite-sample multiple testing can invalidate interpretations. These are candidate observational graphs, not validated interventions. The 279-node vocabulary includes held-out target identities from annotations. Original whole-file median normalization includes held-out library-size information; this transductive dependence is inherited even though only training-control rows enter discovery. No test outcomes are used for algorithm/feature/hyperparameter selection.

Reliable implementation references:
- https://causal-learn.readthedocs.io/en/latest/search_methods_index/Permutation-based%20causal%20discovery%20methods/BOSS.html
- https://github.com/cmu-phil/boss
- https://github.com/py-why/causal-learn

See `graphs/causal_split0/feasibility_report.md` for measured small benchmarks and limitations of extrapolation. No full 279-node discovery or GNN training is part of these benchmarks.
