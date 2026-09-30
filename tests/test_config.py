from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gausstwin.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_base_runtime_contract(self):
        config = load_config(ROOT / "configs/base.yaml")
        self.assertEqual(config["runtime"]["device"], "auto")
        self.assertEqual(config["runtime"]["expected_gpu_count"], 2)
        self.assertEqual(config["runtime"]["backend"], "kaggle")

    def test_inheritance_merges_nested_values(self):
        config = load_config(ROOT / "configs/kaggle_t4.yaml")
        self.assertEqual(config["project"]["name"], "GaussTwin")
        self.assertEqual(config["runtime"]["device"], "cuda")
        self.assertEqual(config["runtime"]["expected_gpu_count"], 2)
        self.assertEqual(config["runtime"]["mixed_precision"], "fp16")
        self.assertEqual(config["runtime"]["memory"]["per_gpu_batch_size"], 1)
        self.assertIn("checkpoints", config["paths"])

    def test_invalid_device_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "runtime:\n  device: quantum\n  expected_gpu_count: 0\n  backend: local\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
