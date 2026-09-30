from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gausstwin.utils.environment import collect_environment, format_environment


class EnvironmentTests(unittest.TestCase):
    def test_report_has_stable_schema(self):
        report = collect_environment()
        self.assertTrue(
            {"python_version", "os", "pytorch_version", "cuda_available",
             "cuda_runtime_version", "gpu_count", "gpus"}.issubset(report)
        )
        self.assertEqual(report["gpu_count"], len(report["gpus"]))
        self.assertIn("total_vram_bytes", report)

    def test_cpu_report_is_truthful(self):
        report = collect_environment()
        if not report["cuda_available"]:
            self.assertEqual(report["gpu_count"], 0)
            self.assertEqual(report["gpus"], [])
            self.assertIn("GPUs: none attached", format_environment(report))


if __name__ == "__main__":
    unittest.main()
