"""Audited evaluation of the completed full-graph SCCVAE run."""
import os, sys, json, time, hashlib, random, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / 'sc_causal')
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import causal.metrics as metrics
from causal.run import get_data
from causal.model import CausalVAE_Gaussian
from causal.inference import compute_predictions
from causal.select_model_shift_values import model_select_inference, model_select_errors

RUN = ROOT / 'results/sccvae_baseline/fresh_20261021_214807'
SAVED = RUN / 'outs/ood_split_0_full'
OUT = RUN / 'evaluation'
OUT.mkdir(exist_ok=True)
START = time.time()
assert torch.cuda.is_available(), 'CUDA required'
torch.set_default_dtype(torch.float64)
def seed():
    torch.manual_seed(20); torch.cuda.manual_seed_all(20); np.random.seed(20); random.seed(20)
seed()
config = json.loads((SAVED/'config.json').read_text())
assert config['mode']=='full' and config['split_num']==0 and config['num_epochs']==100
state = torch.load(SAVED/'latest_recovery.pth', map_location='cpu', weights_only=False)
assert state['completed_epochs']==100
audit = {'completed_epochs':state['completed_epochs'], 'training_elapsed_seconds':state['elapsed_seconds'],
         'best_history':state['best_history'], 'config':config, 'checkpoint':'best_val_mmd.pth',
         'checkpoint_sha256':hashlib.sha256((SAVED/'best_val_mmd.pth').read_bytes()).hexdigest(),
         'selection_audit':'Original selects per-test-perturbation shift in [0,0.1] by test MSE. Test-informed/oracle; not leakage-free.',
         'adaptations':['Metric device cuda:1 -> cuda:0 in memory only', 'torch.no_grad for inference; no numerical/model changes'],
         'limitations':['Training seeds NumPy/Torch but not Python random/hash iteration; exact independent-run determinism not guaranteed.',
                        'Original validation remains in training mode and tracks gradients; preserved during training.',
                        'Full graph is the Full/SCCVAE variant, not the sparse Causal variant.',
                        'Fixed top-50 gene indices are repository-defined, not newly selected DE genes.']}
del state; gc.collect()
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
# Correct only the device literal, preserving every metric calculation.
source = (ROOT/'sc_causal/causal/metrics.py').read_text()
assert source.count('"cuda:1"')==1
exec(compile(source.replace('"cuda:1"','"cuda:0"'),str(ROOT/'sc_causal/causal/metrics.py'),'exec'), metrics.__dict__)
datasets, loaders, sizes = get_data(32,False,'pca',50,True,False,False,modes=['test'],ptbbatch=True,split_num=0)
net = CausalVAE_Gaussian(8563,50,512,[1024,1024],[1024,1024],'train_util_files/B_512_upper_triangular.npy','full','cuda:0','pca')
net.load_state_dict(torch.load(SAVED/'best_val_mmd.pth',weights_only=False)['model_state'])
net.to('cuda:0')
EXCLUDE = {'nontargeting','non-targeting','nontargeting_ref','INTS2','POLR2B','SMN2','DDX47','TREX2','PHB','BGLAP','ANAPC2','ZNF687'}
FUNCS = {'MSE':metrics.mse_per_ptb,'MMD':metrics.mmd_per_ptb,'PearsonR':metrics.pearsonr_per_ptb,
         'Frac Correct Direction':metrics.frac_correct_direction_per_ptb,'Frac Changed Direction':metrics.frac_changed_direction_per_ptb,
         'Energy Distance':metrics.energy_per_ptb}
summaries=[]
def evaluate(label, checkpoint):
    seed()
    path=OUT/f'{label}_y.h5ad'
    if not path.exists():
        with torch.no_grad():
            variables=compute_predictions(str(checkpoint),['hard'],'cuda:0',setting='full',ptb_type='pca',ptb_dim=50,
                data_mode='test',ood_split=0,data=(datasets['test'],loaders['test']),net=net)
        variables['y'].write_h5ad(path)
        del variables; gc.collect(); torch.cuda.empty_cache()
    adata=sc.read_h5ad(path)
    ptbs=sorted(set(adata.obs.loc[adata.obs['y']=='ground truth','ptb'])-EXCLUDE)
    for top, group in [(False,'All genes'),(True,'Top 50')]:
        for name,func in FUNCS.items():
            print(label,group,name,flush=True)
            seed()
            values=func(adata,ptbs=ptbs,marker_genes=top)
            frame=pd.DataFrame({'PTBs':list(values),'value':list(values.values())})
            frame.to_csv(OUT/f'{label}_{name}_{group}.csv',index=False)
            ref=pd.read_csv(ROOT/f'sc_causal/visualize_paper_tables_and_figures/tables/ood_{name}_{group}.csv')
            common=frame.merge(ref,on='PTBs')
            summaries.append({'protocol':label,'genes':group,'metric':name,'n_ptbs':len(frame),
                'mean':float(frame.value.mean()),'std_across_ptbs':float(frame.value.std()),
                'reference_all_five_splits_full_mean':float(ref.Full.mean()),
                'matched_ptbs':len(common),'reference_matched_full_mean':float(common.Full.mean()),
                'measured_matched_mean':float(common.value.mean())})
            pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    del adata; gc.collect()
evaluate('no_test_shift_selection',SAVED/'best_val_mmd.pth')
# Original selection outputs are isolated from the training checkpoints.
import shutil
ORACLE=OUT/'original_test_selected'; ORACLE.mkdir(exist_ok=True)
if not (ORACLE/'best_val_mmd.pth').exists(): shutil.copy2(SAVED/'best_val_mmd.pth',ORACLE/'best_val_mmd.pth')
seed()
net.load_state_dict(torch.load(SAVED/'best_val_mmd.pth',weights_only=False)['model_state'])
with torch.no_grad():
    model_select_inference(str(ORACLE),datasets['test'],net,50,'pca','hard')
    model_select_errors(str(ORACLE),datasets['test'],net,'hard')
evaluate('original_test_selected',ORACLE/'best_val_mmd_shiftselect_hard.pth')
audit['evaluation_elapsed_seconds']=time.time()-START
audit['test_dataset_size']=sizes['test']
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
table=pd.DataFrame(summaries)
report='# SCCVAE K562 OOD split 0 reproduction\n\n'
report+='Training completed 100 epochs (exit code 0). Selected best_val_mmd.pth, epoch index 56 (57th epoch), using validation MMD+KL.\n\n'
report+='Original architecture: full upper triangular SCM, z=512, PCA perturbation dimension=50, encoder/decoder two 1024-unit layers, float64, Adam lr=5e-4, eps=0.01, weight decay=1e-6, batch=32, seed=20. Original preprocessing and validation behavior preserved.\n\n'
report+='The original per-perturbation shift selection minimizes test MSE over {0,0.1}; these results are test-informed and cannot be claimed as leakage-free generalization. The no_test_shift_selection branch uses learned shifts without test-outcome selection.\n\n'
report+=table.to_csv(index=False)+'\n\nReference values come from the checked-in author result tables (Full column). The all-table mean pools five splits; matched perturbations restrict reference rows to this evaluation set but do not establish identical cells or random seeds. No five-split reproduction claim is made.\n\n'
report+='Limitations: '+ '; '.join(audit['limitations'])+'\n'
(OUT/'reproduction_report.md').write_text(report)
print('EVALUATION COMPLETE',OUT,flush=True)
