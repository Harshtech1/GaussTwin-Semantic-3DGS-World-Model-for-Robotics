# Development workflow

GaussTwin uses a three-environment workflow:

1. Develop, document, and run CPU-safe checks in Lightning AI Studio.
2. Commit and push reviewed source to GitHub, the source of truth.
3. Pull the same revision into Kaggle for GPU-only experiments; bring code, configs, and small metrics back through Git, but store large artifacts separately.

## Lightning CPU development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/dev.txt
python -m pip install -e .
make check test
```

Do not install the Kaggle requirements, select a GPU machine, or run training here. Never commit `.env`, Kaggle tokens, raw/processed data, checkpoints, or renders.

## Change discipline

- Branch from an up-to-date GitHub revision.
- Keep experiments reproducible through committed YAML configuration.
- Record the Git commit and environment report with future experiment metadata.
- Run CPU tests before push and GPU smoke tests on Kaggle before long runs.
