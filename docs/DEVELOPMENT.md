# Development environments

GaussTwin deliberately separates development from GPU execution.

| Environment | Responsibility | Must not do |
| --- | --- | --- |
| Lightning AI Studio | CPU-only code, documentation, unit tests, configuration, VS Code/Codex work, and Git | GPU training, CUDA installation, large models |
| GitHub | Versioned source-of-truth synchronization between Lightning and Kaggle | Store credentials, datasets, checkpoints, or renders |
| Kaggle | CUDA/PyTorch execution through the remote Jupyter kernel and future gsplat/3DGS workloads | Replace the reviewed source-of-truth workflow |

## Lightning workflow

```bash
cd GaussTwin
python scripts/environment_check.py --config configs/base.yaml
python -m unittest discover -s tests -v
git status
```

Commit focused changes and push them to GitHub. Kaggle must pull the same commit before a GPU experiment. Do not install `requirements/kaggle-gpu.txt` in Lightning.
