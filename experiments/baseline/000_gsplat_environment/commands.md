# Validation commands

Run these commands from the GaussTwin repository in the validated Kaggle environment.

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvcc --version
gcc --version
ninja --version
python -c "import gsplat; print(gsplat.__version__)"
python scripts/kaggle_gpu_test.py
python scripts/gsplat_smoke_test.py --device cuda:0
python scripts/gsplat_smoke_test.py --device cuda:1
```

`kaggle_gpu_test.py` performs the independent GPU 0 and GPU 1 CUDA smoke tests. The two `gsplat_smoke_test.py` commands perform the independent GPU 0 and GPU 1 rasterization checks.
