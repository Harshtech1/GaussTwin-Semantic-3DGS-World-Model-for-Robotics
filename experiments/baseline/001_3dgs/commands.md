# Kaggle commands

```bash
cd /kaggle/working/GaussTwin
python scripts/prepare_scene.py --config experiments/baseline/001_3dgs/config.yaml
python scripts/train_3dgs.py --config experiments/baseline/001_3dgs/config.yaml --device cuda:0
python scripts/evaluate.py --config experiments/baseline/001_3dgs/config.yaml --scene-root /kaggle/input/<your-colmap-scene> --checkpoint /kaggle/working/gausstwin/outputs/checkpoints/001_3dgs/iter_003000.pt --device cuda:0
```

Use `cuda:1` only as a separate one-GPU experiment. There is no multi-GPU launch command in GaussTwin-001.
