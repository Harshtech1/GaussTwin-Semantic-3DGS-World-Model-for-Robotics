# GaussTwin-001 — reproducible 3DGS baseline

GaussTwin-001 is the first real GaussTwin reconstruction pipeline. It takes a precomputed COLMAP sparse model plus RGB images, initializes a fixed Gaussian set from sparse points, and optimizes RGB reconstruction with `gsplat==1.5.3`.

This baseline uses exactly one CUDA device at a time (default `cuda:0`). It does not combine the two T4 GPUs, download data, run COLMAP, densify/prune Gaussians, or implement semantic features.
