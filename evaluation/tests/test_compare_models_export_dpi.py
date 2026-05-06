import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compare_models import EXPORT_DPI


class CompareModelsExportDpiTests(unittest.TestCase):
    def test_compare_models_exports_use_300_dpi(self):
        self.assertEqual(300, EXPORT_DPI)


if __name__ == "__main__":
    unittest.main()
