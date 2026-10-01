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
        self.assertEqual(config["python"]["version"], "3.12")
        self.assertEqual(config["pytorch"]["version"], "2.10.0+cu128")
        self.assertEqual(config["cuda"]["runtime"], "12.8")
        self.assertEqual(config["gsplat"]["version"], "1.5.3")
        self.assertEqual(config["hardware"]["gpu_count"], 2)
        self.assertEqual(config["hardware"]["gpu_memory_gib"], 14.56)

    def test_invalid_device_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "runtime:\n  device: quantum\n  expected_gpu_count: 0\n  backend: local\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_reconstruction_baseline_defaults(self):
        config = load_config(ROOT / "configs/reconstruction.yaml")
        self.assertEqual(config["reconstruction"]["image_resolution"], 640)
        self.assertEqual(config["reconstruction"]["max_points"], 100000)
        self.assertEqual(config["training"]["device"], "cuda:0")
        self.assertEqual(config["training"]["iterations"], 3000)

    def test_adaptive_reconstruction_defaults(self):
        config = load_config(ROOT / "configs/reconstruction_002.yaml")
        self.assertEqual(config["training"]["device"], "cuda:0")
        self.assertEqual(config["refinement"]["warmup_iterations"], 500)
        self.assertEqual(config["refinement"]["max_gaussian_count"], 60000)


if __name__ == "__main__":
    unittest.main()
