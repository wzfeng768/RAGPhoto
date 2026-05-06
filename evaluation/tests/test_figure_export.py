import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from figure_export import save_figure_dual_format


class SaveFigureDualFormatTests(unittest.TestCase):
    def test_png_output_also_creates_tiff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sample.png"
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            written_paths = save_figure_dual_format(fig, output_path)

            self.assertEqual(
                [output_path, output_path.with_suffix(".tiff")],
                written_paths,
            )
            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.with_suffix(".tiff").exists())


if __name__ == "__main__":
    unittest.main()
