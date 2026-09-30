# 000 — gsplat environment validation

This reproducibility record captures the successful Kaggle validation of the pinned `gsplat==1.5.3` baseline. It establishes only the GPU software and hardware precondition for later work; it is not a 3DGS training run and uses no dataset.

The validation ran at repository commit `784b78b` on two independent Tesla T4 CUDA devices. Initial GaussTwin experiments use one device at a time. The two 14.56 GiB devices are not one 29.12 GiB CUDA device; multi-GPU execution remains a future explicit experiment.
