# Experiments

Experiment groups are `baseline`, `semantic`, `spatial`, and `robotics`. Every run record must include the Git commit, resolved configuration, random seed, environment report, dataset manifest, command, metrics, and artifact location.

## GaussTwin-001: reproducible 3DGS baseline

GaussTwin-001 is the first reconstruction experiment. It consumes a pre-existing COLMAP sparse model and source images, initializes one Gaussian per retained sparse point, and optimizes RGB reconstruction with gsplat on **one CUDA device** (default `cuda:0`). It does not run COLMAP, download data, use a second T4, or implement semantic features.

Use `configs/reconstruction.yaml` as the T4-conscious starting point: 640-pixel maximum image dimension, one view per iteration, at most 100,000 initial points, 3,000 iterations, and checkpoints/renders every 500 iterations. Adjust only through a committed experiment configuration.

The protocol is:

1. Place a COLMAP-style scene under a configured `reconstruction.scene_root`.
2. Validate it with `python scripts/prepare_scene.py --config configs/reconstruction.yaml --scene-root <scene>`.
3. Run `python scripts/train_3dgs.py --config configs/reconstruction.yaml --scene-root <scene> --device cuda:0` on Kaggle.
4. Preserve the final metrics JSON, selected renders, checkpoint, resolved config, Git commit, GPU name, peak VRAM, duration, Gaussian count, and validation PSNR/SSIM.

PSNR and a lightweight differentiable SSIM are implemented with PyTorch. LPIPS is deliberately not included: it requires an additional learned metric dependency and is reported as unavailable until it is separately approved.
