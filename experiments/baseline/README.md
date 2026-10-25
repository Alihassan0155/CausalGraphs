# SCCVAE baseline artifact index

Artifacts retain their historical layout to preserve logged paths and provenance. No duplicate checkpoints or predictions were created during organization. Paths below start at `results/sccvae_baseline/` unless stated otherwise.

| Artifact | Location |
|---|---|
| Completed 100-epoch run | `fresh_20261021_214807/` |
| Training log | `fresh_20261021_214807/training.log` |
| Environment / commit / GPU | `fresh_20261021_214807/{packages,git_commit,gpu,source_diff}.txt` |
| Checkpoints, saved config, W&B logs | `fresh_20261021_214807/outs/ood_split_0_full/` |
| Metrics, predictions, history and report | `fresh_20261021_214807/evaluation/` |
| Supplemental discrepancy audit | `fresh_20261021_214807/evaluation/discrepancy_audit/` |
| Earlier interrupted run | `outs/ood_split_0_full/` |
| Earlier unsuccessful launcher run | `fresh_20261021_203831/` |
| Historical smoke artifacts and console logs | Files directly in the baseline result root |

The selected checkpoint is `best_val_mmd.pth`, epoch index 56. Table-compatible evaluation uses 54 targets (INTS2 excluded); raw predictions contain 55 mapped targets. Preserve both no-test-shift-selection and original test-selected results. See the supplemental audit for transductive preprocessing and comparability limitations.

New baseline runs use `results/sccvae_baseline/runs/fresh_<timestamp>/`: logs in `logs/`, environment/provenance in `provenance/`, model artifacts and saved config in `outs/ood_split_0_full/`, and subsequent evaluation in `evaluation/`. Existing completed runs are not moved to this new layout.
