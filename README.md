# GaussTwin

GaussTwin is a research scaffold for a semantic 3D Gaussian world model for robotics. It is designed to connect camera reconstruction, 3D Gaussian splatting, open-vocabulary semantics, scene graphs, spatial queries, and robot-aware digital twins.

> **Project status:** architecture and development tooling only. Training, rendering, semantic fusion, robotics integration, and dynamic/4D Gaussian modeling are planned and are not yet implemented.

## Compute model

- **Lightning AI Studio (CPU):** persistent development, configuration, documentation, and lightweight tests.
- **GitHub:** source of truth and transfer mechanism.
- **Kaggle T4 x2:** future GPU experiments through a Kaggle Jupyter server/kernel.

Never add credentials, datasets, checkpoints, or generated renders to version control. See [DEVELOPMENT.md](DEVELOPMENT.md) and [KAGGLE.md](KAGGLE.md).

## Quick start

```bash
cd GaussTwin
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/environment_check.py --config configs/base.yaml
make test
```

The package supports Python 3.11 and newer. GPU dependencies are intentionally separate; do not install `requirements/kaggle-gpu.txt` in the Lightning CPU workspace.

## Planned pipeline

```text
RGB/RGB-D -> reconstruction -> Gaussian scene -> semantic features
          -> scene graph -> language/spatial queries -> robot reasoning
```

Each pipeline script currently validates configuration and reports its planned status without starting expensive work.

## License

Apache-2.0. Research dependencies and future datasets retain their own licenses.
