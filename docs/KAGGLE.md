# Kaggle GPU execution

Kaggle is the execution environment for CUDA/PyTorch work. A T4 x2 session presents **two separate CUDA devices**; their VRAM is not a single combined 30 GB allocation. Multi-GPU execution must be implemented explicitly later.

## Minimal preflight

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> /kaggle/working/GaussTwin
cd /kaggle/working/GaussTwin
git rev-parse HEAD
python scripts/kaggle_environment_check.py
python scripts/kaggle_gpu_test.py
```

The check reports per-device total, allocated, reserved, and, when CUDA supports it, free memory. The smoke test makes only a four-element tensor allocation on each device.

## Future dependency setup

Only after the preflight passes and a gsplat/PyTorch/CUDA compatibility decision is recorded:

```bash
python -m pip install -r requirements/kaggle-gpu.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

Do not run training yet. Keep data and outputs in Kaggle-managed storage or an approved artifact store, never Git.
