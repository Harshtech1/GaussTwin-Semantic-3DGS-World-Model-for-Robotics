# Kaggle GPU workflow

GPU execution is planned for a Kaggle notebook using its Jupyter server or a remote Jupyter kernel. The intended target is two NVIDIA T4 GPUs; availability remains a property of the selected Kaggle session and must be verified rather than assumed.

## Setup in a Kaggle GPU session

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> /kaggle/working/GaussTwin
cd /kaggle/working/GaussTwin
python scripts/environment_check.py --config configs/kaggle_t4.yaml
python -m pip install -r requirements/kaggle-gpu.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

Verify that the environment check reports CUDA availability, two GPUs, their names, and memory before running future training. Do not replace Kaggle's PyTorch build unless a documented gsplat compatibility issue requires it.

## Data and outputs

Attach datasets using Kaggle Datasets and configure paths in a local experiment YAML. Write transient results under `/kaggle/working`; publish large outputs as a private Kaggle Dataset or other artifact store. Never commit tokens or large artifacts to Git.

Training commands are intentionally not provided yet because the training implementation is planned, not complete.
