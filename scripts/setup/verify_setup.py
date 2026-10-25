"""Verify the SCCVAE research environment without loading datasets or training."""
import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "anndata": "anndata",
    "scanpy": "scanpy",
    "torch": "torch",
    "causallearn": "causal-learn (PC/BOSS)",
    "wandb": "wandb",
}

print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
print(f"Executable: {sys.executable}")
failed = []
for module, package in MODULES.items():
    try:
        loaded = importlib.import_module(module)
        print(f"OK  {package}: {getattr(loaded, '__version__', 'imported')}")
    except Exception as exc:  # report all failures before exiting
        failed.append((package, repr(exc)))
        print(f"ERR {package}: {exc}")

try:
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch device check: {torch.tensor([1.0, 2.0]).device}")
except Exception:
    pass

for folder in ("data", "processed", "graphs", "results"):
    path = ROOT / folder
    print(f"Folder {folder}: {'OK' if path.is_dir() else 'MISSING'}")

if failed:
    print("\nVerification failed. Review the import errors above.")
    raise SystemExit(1)
print("\nVerification passed: imports work and the environment is CPU-compatible.")
