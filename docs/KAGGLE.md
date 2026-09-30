# Kaggle GPU execution

Kaggle is the GPU execution environment. Lightning remains development-only (code, documentation, tests, configuration, VS Code/Codex, and Git), while GitHub synchronizes reviewed source between Lightning and Kaggle.

## Validated baseline environment

The `baseline/000_gsplat_environment` validation passed at commit `784b78b`:

- Python 3.12.13; PyTorch 2.10.0+cu128; CUDA runtime/toolkit 12.8
- GCC 11.4.0; Ninja 1.13.0; gsplat 1.5.3 (pinned baseline)
- 2 × Tesla T4, with 14.56 GiB per device

Kaggle exposes two independent T4 CUDA devices. **2 × T4 is not one 29.12 GiB GPU.** Initial experiments use one GPU at a time; multi-GPU execution is a future explicit experiment. The first gsplat CUDA extension initialization can take several minutes (approximately 310.65 seconds in this validation).

## Minimal preflight

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> /kaggle/working/GaussTwin
cd /kaggle/working/GaussTwin
git rev-parse HEAD
python scripts/kaggle_environment_check.py
python scripts/kaggle_gpu_test.py
python scripts/gsplat_environment_check.py
python scripts/gsplat_smoke_test.py --device cuda:0
python scripts/gsplat_smoke_test.py --device cuda:1
```

The environment check reports total-across-devices separately from per-device memory. The gsplat smoke test is a tiny one-device synthetic rasterization, not training.

## Future dependency setup

The validated Kaggle-only dependency set is pinned in `requirements/kaggle-gpu.txt`:

```bash
python -m pip install -r requirements/kaggle-gpu.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

Do not run training yet. Keep data and outputs in Kaggle-managed storage or an approved artifact store, never Git.
