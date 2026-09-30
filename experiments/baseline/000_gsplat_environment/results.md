# Results

Experiment identifier: `baseline/000_gsplat_environment`  
Repository commit: `784b78b`

- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- CUDA runtime/toolkit: 12.8
- gsplat: 1.5.3
- Hardware: 2 × Tesla T4, 14.56 GiB per device
- GPU 0 and GPU 1 CUDA smoke tests: pass
- GPU 0 and GPU 1 gsplat rasterization: pass
- First gsplat CUDA extension compilation: approximately 310.65 seconds

The successful synthetic rasterization produced `render` shape `[1, 512, 512, 3]` and `alpha` shape `[1, 512, 512, 1]`. These are validation outputs only; no 3DGS training or dataset processing occurred.
