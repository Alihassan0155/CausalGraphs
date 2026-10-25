"""Validate outputs and render a concise measured-results report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sccvae_baseline/fresh_20261021_214807/evaluation'
audit=json.loads((OUT/'audit.json').read_text())
summary=pd.read_csv(OUT/'summary.csv')
assert len(summary)==24, 'Both branches must finish before final reporting'
assert np.isfinite(summary['mean']).all()
history=pd.read_csv(OUT/'training_validation_history.csv')
vals=history.dropna(subset=['elbo_mmd val']).reset_index(drop=True)
assert len(vals)==100 and np.isfinite(vals['elbo_mmd val']).all()
assert vals['elbo_mmd val'].idxmin()==56
vals.insert(0,'epoch_index',range(100))
vals.to_csv(OUT/'validation_by_epoch.csv',index=False)
errs=np.load(OUT/'original_test_selected/errs_hard_redo_mse.npy')
audit['shift_candidate_test_mse_shape']=list(errs.shape)
audit['shift_choice_counts']={str(s):int((np.argmin(errs,axis=0)==i).sum()) for i,s in enumerate([0,0.1])}
audit['limitations'] += [
 'No test-based shift tuning is not a proof of strict end-to-end leakage freedom: original normalize_total uses default whole-file median before splitting; normalization includes test-cell library sizes.',
 'Original test sampler iterates a Python set; seed=20 alone does not fix hash ordering, so stochastic predictions can differ across processes.',
 'Original selection candidate generation has no explicit seed; evaluation harness seeds all RNGs to 20 and records this adaptation.',
 'Training/validation objective MMD differs from evaluation MMD; their numerical values are not directly comparable.',
 'Published reference is the author repository table, not an independently verified paper-table transcription.']
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
lines=['# SCCVAE K562 OOD split 0: measured reproduction report','',
 'Training completed all 100 configured epochs with exit code 0. Latest recovery state records 100 completed epochs; checkpoint_100.pth and last_model.pth exist. All 100 validation MMD+KL entries are finite.',
 '',f"Training runtime including preprocessing/checkpoint I/O: {audit['training_elapsed_seconds']/3600:.3f} hours. Evaluation runtime: {audit['evaluation_elapsed_seconds']/60:.2f} minutes.",
 '', '## Methods','',
 'Repository commit: 3fe62d4e8b17e36197888421ed443cab0633fe46. Windows; NVIDIA RTX A4000 (16 GB), PyTorch 2.4.1+cu124; float64. Full upper-triangular SCM, latent dimension 512; PCA perturbation dimension 50; two 1024-unit encoder/decoder hidden layers. Original K562 OOD split 0; normalize_total then log1p; original gene mapping. Batch size 32; Adam lr=0.0005, eps=0.01, weight decay=1e-6; original per-parameter gradient clipping norm 1; beta linearly increases 0 to 1 over 100 epochs, gamma=10, control reconstruction scale=0.25, no scheduler. Seed 20; see determinism limitations below.',
 '', 'Original validation behavior was retained, including training mode, gradient tracking, and stochastic control sampling. Added checkpoint I/O did not change the model or loss. Checkpoints were selected by validation MMD+KL, not training loss.',
 '', f"Selected checkpoint: best_val_mmd.pth, zero-based epoch 56 (57th epoch), validation MMD+KL={vals.loc[56,'elbo_mmd val']:.12f}. SHA256: {audit['checkpoint_sha256']}.",
 '', '## Test-selection audit','',
 'Original select_model_shift_values.py explicitly loads modes=[test], predicts candidate shifts {0,0.1}, computes MSE against those test outcomes, and chooses a shift separately for each perturbation. These test-selected results are an oracle-style reconstruction of the released protocol, not an unbiased held-out generalization estimate. The other branch retains learned shifts and does not select on test outcomes. Original whole-file normalization remains a separate leakage concern.',
 '',f"Original candidate selection covered {errs.shape[1]} perturbations; choices: {audit['shift_choice_counts']}. Official table exclusions leave 54 perturbations for reported metric means. Batch size 32 and original dropped remainders preserved.",
 '', '## Measured split-0 results','',
 '| Gene set | Metric | No test-shift tuning | Original test-selected | Author Full, matched perturbations |',
 '|---|---|---:|---:|---:|']
for group in ['All genes','Top 50']:
 for metric in summary.metric.unique():
    sub=summary[(summary.genes==group)&(summary.metric==metric)].set_index('protocol')
    lines.append(f"| {group} | {metric} | {sub.loc['no_test_shift_selection','mean']:.8f} | {sub.loc['original_test_selected','mean']:.8f} | {sub.loc['original_test_selected','reference_matched_full_mean']:.8f} |")
lines += ['', 'Metrics are exact released repository implementations: MSE of perturbation means, multi-kernel MMD, Pearson correlation of mean profiles, direction fractions, and the repository-defined energy statistic. Fixed top-50 indices preserved; these are not newly computed test differential-expression genes.',
 '', '## Reference comparison','',
 'Reference: https://github.com/uhlerlab/sccvae/tree/master/sc_causal/visualize_paper_tables_and_figures/tables . Compare Full only; Causal denotes a different sparse graph. Author OOD CSVs concatenate five split tables. Matched reference values above restrict rows by perturbation identity, but identical cell subsets, inference RNG, and artifact provenance cannot be established. summary.csv also stores the pooled five-split mean as context only. Single-split means and variation across perturbations are not five-split means or uncertainty across seeds. No claim of exact published reproduction is made.',
 '', '## Limitations','']
lines += ['- '+item for item in audit['limitations']]
lines += ['', '## Artifacts','', 'summary.csv; per-perturbation CSVs for all 24 protocol/gene-set/metric combinations; both prediction H5ADs; original candidate predictions and shift-selected checkpoint; audit.json; training_validation_history.csv; validation_by_epoch.csv. Original training checkpoints and source files were preserved.']
(OUT/'reproduction_report.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines[:10]))
