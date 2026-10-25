"""CUDA smoke test using the original SCCVAE classes and full OOD split 0."""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

os.chdir(Path(__file__).resolve().parents[2] / "sc_causal")
sys.path.insert(0, os.getcwd())

torch.set_default_dtype(torch.float64)
torch.manual_seed(20)
np.random.seed(20)
random.seed(20)
torch.cuda.manual_seed_all(20)

from causal.run import get_data
from causal.model import CausalVAE_Gaussian

assert torch.cuda.is_available(), "CUDA is unavailable; refusing to continue."
device = torch.device("cuda:0")
assert torch.get_default_dtype() == torch.float64

datasets, loaders, sizes = get_data(
    batch_size=32,
    nontargeting=False,
    ptb_type="pca",
    ptb_dim=50,
    ood=True,
    finetune=False,
    marker=False,
    modes=["train", "val"],
    ptbbatch=True,
    split_num=0,
    tiny=False,
    parent_dir=False,
)

model = CausalVAE_Gaussian(
    exp_dim=8563,
    ptb_dim=50,
    z_dim=512,
    enc_hidden=[1024, 1024],
    dec_hidden=[1024, 1024],
    B_filename="train_util_files/B_512_upper_triangular.npy",
    device=device,
    mode="full",
    ptb_encode="pca",
    parent_dir=False,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, eps=0.01, weight_decay=1e-6)
batch = next(iter(loaders["train"]))
y, p, p1h, gene = batch
y, p, p1h = y.to(device), p.to(device), p1h.to(device)

model.train()
loss, train_losses = model.step(y, p, p1h, gene[0], optimizer, torch.tensor(0.0), torch.tensor(10.0), True, 0.25)
assert np.isfinite(loss)

model.eval()
with torch.no_grad():
    val_batch = next(iter(loaders["val"]))
    vy, vp, vp1h, _ = val_batch
    outs = model(vy.to(device), vp.to(device), vp1h.to(device))
    val_losses = model.loss_function(vy.to(device), outs, torch.tensor(0.0), torch.tensor(10.0), 0.25)
    val_loss = float(val_losses["loss"])
assert np.isfinite(val_loss)

out = Path("../results/sccvae_baseline")
out.mkdir(parents=True, exist_ok=True)
checkpoint = out / "cuda_smoke_checkpoint.pth"
model.save(checkpoint, optimizer=optimizer)
reloaded = CausalVAE_Gaussian(
    exp_dim=8563, ptb_dim=50, z_dim=512, enc_hidden=[1024, 1024], dec_hidden=[1024, 1024],
    B_filename="train_util_files/B_512_upper_triangular.npy", device=device,
    mode="full", ptb_encode="pca", parent_dir=False,
).to(device)
state = torch.load(checkpoint, map_location=device)
reloaded.load_state_dict(state["model_state"])
assert all(torch.isfinite(p).all() for p in reloaded.parameters())

report = {
    "device": torch.cuda.get_device_name(0),
    "cuda_runtime": torch.version.cuda,
    "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
    "default_dtype": str(torch.get_default_dtype()),
    "dataset_sizes": sizes,
    "train_loss": loss,
    "validation_loss": val_loss,
    "checkpoint": str(checkpoint),
    "status": "PASS",
}
(out / "cuda_smoke_test.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
