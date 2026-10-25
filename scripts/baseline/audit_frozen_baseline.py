"""Read-only discrepancy checks; write only new discrepancy_audit artifacts."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import anndata as ad
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'sc_causal'))
from causal.metrics import predict_pearsonr, mse
OUT=ROOT/'results/sccvae_baseline/fresh_20261021_214807/evaluation'
DEST=OUT/'discrepancy_audit'; DEST.mkdir(exist_ok=True)
names={'untuned':OUT/'no_test_shift_selection_y.h5ad','selected':OUT/'original_test_selected_y.h5ad',
 'candidate0':OUT/'original_test_selected/cache/prediction_0_hard.h5ad',
 'candidate01':OUT/'original_test_selected/cache/prediction_0.1_hard.h5ad'}
files={k:ad.read_h5ad(p,backed='r') for k,p in names.items()}
def genes(a): return sorted(set(a.obs.loc[a.obs['y']=='ground truth','ptb']))
eligible=sorted(pd.read_csv(OUT/'no_test_shift_selection_MSE_All genes.csv').PTBs)
mapped=genes(files['untuned'])
result={'prediction_target_sets':{k:genes(a) for k,a in files.items()},'eligible_count':len(eligible),
 'mapped_prediction_count':len(mapped),'excluded_from_metrics':sorted(set(mapped)-set(eligible))}
assert eligible==sorted(pd.read_csv(OUT/'original_test_selected_MSE_All genes.csv').PTBs)
mapping=pd.read_csv(ROOT/'sc_causal/train_util_files/new_estimated_dag_gene.csv').set_index('gene')['idx']
saved=OUT.parent/'outs/ood_split_0_full'
original=torch.load(saved/'best_val_mmd.pth',map_location='cpu',weights_only=False)['model_state']
shifted=torch.load(OUT/'original_test_selected/best_val_mmd_shiftselect_hard.pth',map_location='cpu',weights_only=False)['model_state']
result['changed_checkpoint_keys']=[k for k in original if not torch.equal(original[k],shifted[k])]
errs=np.load(OUT/'original_test_selected/errs_hard_redo_mse.npy')
rows=[]
for j,g in enumerate(mapped):
 row={'gene':g,'eligible':g in eligible,'learned_shift':float(original['s'][mapping[g]]),
      'selected_shift':float(shifted['s'][mapping[g]])}
 targets={}
 for label,a in files.items():
  ids=np.flatnonzero((a.obs.ptb==g)&a.obs['y'].isin(['predicted','ground truth']))
  part=a[ids,:].to_memory()
  true=np.asarray(part.X[part.obs['y']=='ground truth'])
  pred=np.asarray(part.X[part.obs['y']=='predicted'])
  targets[label]=true
  row[label+'_n_true']=len(true); row[label+'_n_pred']=len(pred)
  row[label+'_target_sha256']=hashlib.sha256(true.tobytes()).hexdigest()
  row[label+'_mse']=float(mse(part,g))
  row[label+'_pearson']=float(predict_pearsonr(part,g))
 row['targets_exact_equal']=all(np.array_equal(targets['untuned'],t) for t in targets.values())
 row['candidate_errors_match_saved']=bool(np.allclose([row['candidate0_mse'],row['candidate01_mse']],errs[:,j],rtol=1e-12,atol=1e-12))
 row['selected_shift_correct']=bool(row['selected_shift']==[0.,.1][int(np.argmin(errs[:,j]))])
 rows.append(row)
 print(g,flush=True)
frame=pd.DataFrame(rows); frame.to_csv(DEST/'per_target_checks.csv',index=False)
e=frame[frame.eligible]
result.update({'all_targets_exact_equal':bool(frame.targets_exact_equal.all()),
 'saved_candidate_errors_verified':bool(frame.candidate_errors_match_saved.all()),
 'selected_shifts_verified':bool(frame.selected_shift_correct.all()),
 'counts_match':bool((frame.untuned_n_true==frame.selected_n_true).all() and (frame.untuned_n_pred==frame.selected_n_pred).all()),
 'eligible_means':{c:float(e[c].mean()) for c in ['untuned_mse','selected_mse','candidate0_mse','candidate01_mse','untuned_pearson']},
 'cached_candidate_minimum_mean':float(e[['candidate0_mse','candidate01_mse']].min(axis=1).mean()),
 'learned_shift_range':[float(frame.learned_shift.min()),float(frame.learned_shift.max())],
 'selected_minus_cached_min_mean':float((e.selected_mse-e[['candidate0_mse','candidate01_mse']].min(axis=1)).mean())})
# Verify reference identity and aggregation without claiming that table rows encode splits.
references=[]
for p in sorted((ROOT/'sc_causal/visualize_paper_tables_and_figures/tables').glob('ood_*.csv')):
 ref=pd.read_csv(p); matched=ref[ref.PTBs.isin(eligible)]
 references.append({'file':p.name,'rows':len(ref),'unique_targets':ref.PTBs.nunique(),'matched_rows':len(matched),
 'Full_pooled_mean':float(ref.Full.mean()),'Full_matched_mean':float(matched.Full.mean())})
pd.DataFrame(references).to_csv(DEST/'author_full_reference.csv',index=False)
result['reference_note']='OOD author tables concatenate five splits and contain no split-ID column; matched target identity is not proof of identical cell subsets.'
(DEST/'checks.json').write_text(json.dumps(result,indent=2))
for a in files.values(): a.file.close()
print(json.dumps({k:v for k,v in result.items() if k!='prediction_target_sets'},indent=2))
