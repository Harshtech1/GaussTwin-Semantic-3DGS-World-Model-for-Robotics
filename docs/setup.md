# Setup

Use Python 3.11 or newer. For CPU development, create a virtual environment and install `requirements/dev.txt`, followed by an editable package install. Run `make check test`.

The base dependency is PyYAML. PyTorch is optional for CPU-side environment detection: when absent, the checker reports it as not installed. GPU dependencies belong exclusively to `requirements/kaggle-gpu.txt`; see `KAGGLE.md`.
