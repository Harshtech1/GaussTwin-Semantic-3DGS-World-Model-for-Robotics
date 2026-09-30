.PHONY: install install-dev check test lint

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e '.[dev]'

check:
	python scripts/environment_check.py --config configs/base.yaml

test:
	python -m unittest discover -s tests -v

lint:
	ruff check src scripts tests
