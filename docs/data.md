# Data policy

No dataset is bundled or downloaded by setup. Put immutable source captures in `data/raw/` and derived inputs in `data/processed/`; both directories are ignored except for their marker files. Document provenance, license, checksum, coordinate conventions, camera calibration, and processing commands for each future dataset.

Prefer small, versioned manifests over committed media. Configure alternate roots in YAML instead of editing code.
