"""Run original SCCVAE with additional checkpoint I/O; source files stay intact."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / 'sc_causal')
sys.path.insert(0, str(ROOT / 'sc_causal'))
os.environ.setdefault('WANDB_MODE', 'offline')

import torch
assert torch.cuda.is_available(), 'CUDA required: stopping.'
probe = torch.ones(2, device='cuda:0', dtype=torch.float64).sum().item()
assert probe == 2.0
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('Free/total GPU bytes:', torch.cuda.mem_get_info(), flush=True)

import causal.train as training

def save_recovery(scope, filename):
    import random
    import numpy as np
    import time
    net = scope['net']
    checkpoint = {
        'model_state': net.state_dict(),
        'optimizer_state': scope['optimizer'].state_dict(),
        'completed_epochs': scope.get('epoch', -1) + 1,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(),
        'numpy_rng': np.random.get_state(),
        'python_rng': random.getstate(),
        'beta_schedule': scope['beta_schedule'],
        'gamma_schedule': scope['gamma_schedule'],
        'best_history': {k: v for k, v in scope.items() if k.startswith('best_')},
        'elapsed_seconds': time.monotonic() - START,
    }
    if scope['scheduler'] is not None:
        checkpoint['scheduler_state'] = scope['scheduler'].state_dict()
    target = Path(scope['savedir']) / filename
    temporary = target.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, target)
    print('Saved:', target, 'completed epochs:', checkpoint['completed_epochs'], flush=True)

import time
START = time.monotonic()
source_path = ROOT / 'sc_causal' / 'causal' / 'train.py'
source = source_path.read_text()
anchor = '    for epoch in range(num_epochs):'
assert source.count(anchor) == 1
source = source.replace(anchor, "    save_recovery(locals(), 'initial_state.pth')\n" + anchor)
anchor = '            wandb.log(log)'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "\n            if phase == phases[-1]:\n                save_recovery(locals(), 'latest_recovery.pth')")
training.__dict__['save_recovery'] = save_recovery
exec(compile(source, str(source_path), 'exec'), training.__dict__)
runpy.run_path(str(ROOT / 'sc_causal' / 'causal' / 'run.py'), run_name='__main__')
