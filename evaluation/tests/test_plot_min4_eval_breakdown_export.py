import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT_DIR / "Vis"

for path in (ROOT_DIR, VIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_min4_eval import generate_correctness_breakdown_plots


class CorrectnessBreakdownExportTests(unittest.TestCase):
    def test_by_method_plots_also_export_structured_data(self):
        records = [
            {
                "model": "Model A",
                "mode": "direct_llm",
                "category": "Device engineering",
                "difficulty": "easy",
                "reasoning": "single-hop",
                "answer_correctness": 0.2,
                "question_id": "q1",
            },
            {
                "model": "Model A",
                "mode": "direct_llm",
                "category": "Material design",
                "difficulty": "hard",
                "reasoning": "multi-hop",
                "answer_correctness": 0.4,
                "question_id": "q2",
            },
            {
                "model": "Model B",
                "mode": "direct_llm",
                "category": "Device engineering",
                "difficulty": "easy",
                "reasoning": "single-hop",
                "answer_correctness": 0.3,
                "question_id": "q3",
            },
            {
                "model": "Model B",
                "mode": "direct_llm",
                "category": "Material design",
                "difficulty": "hard",
                "reasoning": "multi-hop",
                "answer_correctness": 0.6,
                "question_id": "q4",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)

            generate_correctness_breakdown_plots(
                records=records,
                out_dir=out_dir,
                selected_models=["Model A", "Model B"],
                selected_methods=["direct_llm"],
                selected_breakdowns=["category"],
                analysis_metric="avg_correctness",
                correct_threshold=0.8,
            )

            base_path = out_dir / "correctness_breakdown" / "by_method" / "direct" / "category_avg_correctness"
            self.assertTrue(base_path.with_suffix(".png").exists())
            self.assertTrue(base_path.with_suffix(".json").exists())
            self.assertTrue(base_path.with_suffix(".csv").exists())

            with open(base_path.with_suffix(".json"), "r", encoding="utf-8") as file:
                payload = json.load(file)

            self.assertEqual("category", payload["dimension"])
            self.assertEqual("avg_correctness", payload["analysis_metric"])
            self.assertEqual(["Material design", "Device engineering"], payload["ordered_groups"])
            self.assertEqual(2, len(payload["series"]))
            self.assertEqual("Model B (direct)", payload["series"][0]["label"])
            self.assertEqual(0.6, payload["series"][0]["groups"]["Material design"]["value"])

            with open(base_path.with_suffix(".csv"), "r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(4, len(rows))
            self.assertEqual("Model B (direct)", rows[0]["series_label"])
            self.assertEqual("Material design", rows[0]["group"])

    def test_by_method_exports_also_create_one_master_table(self):
        records = [
            {
                "model": "Model A",
                "mode": "direct_llm",
                "category": "Device engineering",
                "difficulty": "easy",
                "reasoning": "single-hop",
                "answer_correctness": 0.2,
                "question_id": "q1",
            },
            {
                "model": "Model B",
                "mode": "direct_llm",
                "category": "Device engineering",
                "difficulty": "easy",
                "reasoning": "single-hop",
                "answer_correctness": 0.3,
                "question_id": "q2",
            },
            {
                "model": "Model A",
                "mode": "agentic_with_kg",
                "category": "Device engineering",
                "difficulty": "easy",
                "reasoning": "single-hop",
                "answer_correctness": 0.8,
                "question_id": "q3",
            },
            {
                "model": "Model B",
                "mode": "agentic_with_kg",
                "category": "Device engineering",
                "difficulty": "easy",
                "reasoning": "single-hop",
                "answer_correctness": 0.9,
                "question_id": "q4",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)

            generate_correctness_breakdown_plots(
                records=records,
                out_dir=out_dir,
                selected_models=["Model A", "Model B"],
                selected_methods=["direct_llm", "agentic_with_kg"],
                selected_breakdowns=["category", "difficulty"],
                analysis_metric="avg_correctness",
                correct_threshold=0.8,
            )

            base_path = out_dir / "correctness_breakdown" / "by_method" / "master_breakdown_table_avg_correctness"
            self.assertTrue(base_path.with_suffix(".json").exists())
            self.assertTrue(base_path.with_suffix(".csv").exists())

            with open(base_path.with_suffix(".json"), "r", encoding="utf-8") as file:
                payload = json.load(file)

            self.assertEqual("avg_correctness", payload["analysis_metric"])
            self.assertEqual(["direct", "with_kg"], payload["methods"])
            self.assertEqual(["category", "difficulty"], payload["dimensions"])
            self.assertEqual(8, len(payload["rows"]))

            with open(base_path.with_suffix(".csv"), "r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(8, len(rows))
            self.assertEqual(
                {
                    "method",
                    "dimension",
                    "series_label",
                    "series_color",
                    "group",
                    "value",
                    "count",
                    "correct_count",
                    "correct_rate",
                    "avg_correctness",
                },
                set(rows[0].keys()),
            )


if __name__ == "__main__":
    unittest.main()
