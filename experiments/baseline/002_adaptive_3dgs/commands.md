# Kaggle protocol

```bash
cd /kaggle/working/GaussTwin
python scripts/prepare_scene.py --config experiments/baseline/002_adaptive_3dgs/config.yaml
python scripts/train_3dgs_002.py --config experiments/baseline/002_adaptive_3dgs/config.yaml --device cuda:0
```

Use a new output directory for every run. The trainer refuses to overwrite same-iteration checkpoints or evaluation files. Run `cuda:1` only as an independent single-GPU replication.
