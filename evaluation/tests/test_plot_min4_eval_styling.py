import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT_DIR / "Vis"

for path in (ROOT_DIR, VIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_min4_eval import (
    CORRECTNESS_FIGSIZE,
    EXPORT_DPI,
    OVERVIEW_FIGSIZE,
    create_agentic_vs_direct_charts,
    create_grouped_bar_chart,
    create_correctness_breakdown_chart,
    create_simple_correctness_chart,
)


class PlotMin4EvalStylingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def _latest_figure(self):
        figure_numbers = plt.get_fignums()
        self.assertTrue(figure_numbers)
        return plt.figure(figure_numbers[-1])

    def test_correctness_chart_canvas_is_slightly_narrower_but_same_height(self):
        self.assertEqual((11.8, 6.3), CORRECTNESS_FIGSIZE)

    def test_simple_correctness_chart_uses_same_canvas_size_as_other_overview_plots(self):
        self.assertEqual(OVERVIEW_FIGSIZE, (12, 6))

    def test_plot_exports_use_300_dpi(self):
        self.assertEqual(300, EXPORT_DPI)

    def test_simple_correctness_chart_enlarges_text_and_markers_without_resizing_canvas(self):
        all_data = {
            "Model A": {
                "categories": {
                    "all": {
                        "modes": {
                            "direct_llm": {"ragas_metrics": {"answer_correctness": 0.51}},
                            "agentic_no_kg": {"ragas_metrics": {"answer_correctness": 0.62}},
                            "agentic_with_kg": {"ragas_metrics": {"answer_correctness": 0.74}},
                        }
                    }
                }
            },
            "Model B": {
                "categories": {
                    "all": {
                        "modes": {
                            "direct_llm": {"ragas_metrics": {"answer_correctness": 0.47}},
                            "agentic_no_kg": {"ragas_metrics": {"answer_correctness": 0.59}},
                            "agentic_with_kg": {"ragas_metrics": {"answer_correctness": 0.69}},
                        }
                    }
                }
            },
            "Model C": {
                "categories": {
                    "all": {
                        "modes": {
                            "direct_llm": {"ragas_metrics": {"answer_correctness": 0.44}},
                            "agentic_no_kg": {"ragas_metrics": {"answer_correctness": 0.57}},
                            "agentic_with_kg": {"ragas_metrics": {"answer_correctness": 0.66}},
                        }
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "correctness_simple.png"
            with patch("plot_min4_eval.plt.close"):
                create_simple_correctness_chart(all_data, "all", str(output_path))

            fig = self._latest_figure()
            ax = fig.axes[0]
            legend = ax.get_legend()
            plotted_lines = [line for line in ax.lines if line.get_label() in all_data]
            marker_sizes = [line.get_markersize() for line in plotted_lines]
            scatter_sizes = [collection.get_sizes()[0] for collection in ax.collections if len(collection.get_offsets())]

            self.assertAlmostEqual(OVERVIEW_FIGSIZE[0], fig.get_size_inches()[0], places=2)
            self.assertAlmostEqual(OVERVIEW_FIGSIZE[1], fig.get_size_inches()[1], places=2)
            self.assertEqual(17, ax.yaxis.label.get_fontsize())
            self.assertTrue(all(label.get_fontsize() == 15 for label in ax.get_xticklabels()))
            self.assertTrue(all(label.get_fontsize() == 14 for label in ax.get_yticklabels()))
            self.assertIsNotNone(legend)
            self.assertTrue(all(text.get_fontsize() == 14 for text in legend.get_texts()))
            self.assertTrue(all(size == 11.0 for size in marker_sizes))
            self.assertTrue(all(size == 120 for size in scatter_sizes))
            self.assertTrue(output_path.exists())

    def test_simple_correctness_chart_exports_same_pixel_size_as_other_overview_plots(self):
        all_data = {
            "Model A": {
                "categories": {
                    "all": {
                        "modes": {
                            "direct_llm": {"ragas_metrics": {"answer_correctness": 0.51, "answer_similarity": 0.62, "answer_relevancy": 0.66, "faithfulness": 0.7, "context_recall": 0.73, "context_precision": 0.75}},
                            "agentic_no_kg": {"ragas_metrics": {"answer_correctness": 0.62, "answer_similarity": 0.7, "answer_relevancy": 0.72, "faithfulness": 0.78, "context_recall": 0.8, "context_precision": 0.82}},
                            "agentic_with_kg": {"ragas_metrics": {"answer_correctness": 0.74, "answer_similarity": 0.79, "answer_relevancy": 0.81, "faithfulness": 0.84, "context_recall": 0.86, "context_precision": 0.88}},
                        }
                    }
                }
            },
            "Model B": {
                "categories": {
                    "all": {
                        "modes": {
                            "direct_llm": {"ragas_metrics": {"answer_correctness": 0.47, "answer_similarity": 0.58, "answer_relevancy": 0.61, "faithfulness": 0.67, "context_recall": 0.69, "context_precision": 0.71}},
                            "agentic_no_kg": {"ragas_metrics": {"answer_correctness": 0.59, "answer_similarity": 0.67, "answer_relevancy": 0.69, "faithfulness": 0.74, "context_recall": 0.77, "context_precision": 0.79}},
                            "agentic_with_kg": {"ragas_metrics": {"answer_correctness": 0.69, "answer_similarity": 0.76, "answer_relevancy": 0.78, "faithfulness": 0.82, "context_recall": 0.84, "context_precision": 0.86}},
                        }
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            grouped_path = out_dir / "metrics_comparison.png"
            simple_path = out_dir / "correctness_simple.png"

            create_grouped_bar_chart(all_data, "all", "agentic_with_kg", str(grouped_path))
            create_simple_correctness_chart(all_data, "all", str(simple_path))

            with Image.open(grouped_path) as grouped_image, Image.open(simple_path) as simple_image:
                self.assertEqual(grouped_image.size, simple_image.size)
                self.assertGreaterEqual(round(grouped_image.info.get("dpi", (0, 0))[0]), 300)
                self.assertGreaterEqual(round(simple_image.info.get("dpi", (0, 0))[0]), 300)

    def test_metrics_comparison_chart_enlarges_labels_legend_and_value_text(self):
        all_data = {
            "Model A": {
                "categories": {
                    "all": {
                        "modes": {
                            "agentic_with_kg": {
                                "ragas_metrics": {
                                    "answer_correctness": 0.74,
                                    "answer_similarity": 0.79,
                                    "answer_relevancy": 0.81,
                                    "faithfulness": 0.84,
                                    "context_recall": 0.86,
                                    "context_precision": 0.88,
                                }
                            }
                        }
                    }
                }
            },
            "Model B": {
                "categories": {
                    "all": {
                        "modes": {
                            "agentic_with_kg": {
                                "ragas_metrics": {
                                    "answer_correctness": 0.69,
                                    "answer_similarity": 0.76,
                                    "answer_relevancy": 0.78,
                                    "faithfulness": 0.82,
                                    "context_recall": 0.84,
                                    "context_precision": 0.86,
                                }
                            }
                        }
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics_comparison.png"
            with patch("plot_min4_eval.plt.close"):
                create_grouped_bar_chart(all_data, "all", "agentic_with_kg", str(output_path))

            fig = self._latest_figure()
            ax = fig.axes[0]
            legend = ax.get_legend()
            text_sizes = [text.get_fontsize() for text in ax.texts]

            self.assertEqual(15, ax.yaxis.label.get_fontsize())
            self.assertTrue(all(label.get_fontsize() == 13 for label in ax.get_xticklabels()))
            self.assertTrue(all(label.get_fontsize() == 13 for label in ax.get_yticklabels()))
            self.assertIsNotNone(legend)
            self.assertTrue(all(text.get_fontsize() == 13 for text in legend.get_texts()))
            self.assertTrue(text_sizes)
            self.assertTrue(all(size == 10 for size in text_sizes))
            self.assertTrue(output_path.exists())

    def test_agentic_vs_direct_charts_enlarge_labels_legend_and_annotations(self):
        all_data = {
            "Model A": {
                "categories": {
                    "all": {
                        "modes": {
                            "direct_llm": {"ragas_metrics": {"answer_correctness": 0.51}},
                            "agentic_no_kg": {"ragas_metrics": {"answer_correctness": 0.62}},
                            "agentic_with_kg": {"ragas_metrics": {"answer_correctness": 0.74}},
                        }
                    }
                }
            },
            "Model B": {
                "categories": {
                    "all": {
                        "modes": {
                            "direct_llm": {"ragas_metrics": {"answer_correctness": 0.47}},
                            "agentic_no_kg": {"ragas_metrics": {"answer_correctness": 0.59}},
                            "agentic_with_kg": {"ragas_metrics": {"answer_correctness": 0.69}},
                        }
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            with patch("plot_min4_eval.plt.close"):
                create_agentic_vs_direct_charts(all_data, "all", out_dir)

            figures = [plt.figure(number) for number in plt.get_fignums()]
            no_kg_figures = []
            with_kg_figures = []

            for fig in figures:
                ax = fig.axes[0]
                legend = ax.get_legend()
                if legend is None:
                    continue
                legend_labels = tuple(text.get_text() for text in legend.get_texts())
                if legend_labels == ("Direct LLM", "Agentic (No KG)"):
                    no_kg_figures.append(fig)
                elif legend_labels == ("Direct LLM", "Agentic (With KG)"):
                    with_kg_figures.append(fig)

            self.assertEqual(2, len(no_kg_figures))
            self.assertEqual(2, len(with_kg_figures))

            no_kg_max_text_sizes = []
            with_kg_max_text_sizes = []
            for fig in no_kg_figures + with_kg_figures:
                ax = fig.axes[0]
                legend = ax.get_legend()
                text_sizes = [text.get_fontsize() for text in ax.texts]

                self.assertEqual(15, ax.yaxis.label.get_fontsize())
                self.assertTrue(all(label.get_fontsize() == 12 for label in ax.get_xticklabels()))
                self.assertTrue(all(label.get_fontsize() == 13 for label in ax.get_yticklabels()))
                self.assertTrue(all(text.get_fontsize() == 13 for text in legend.get_texts()))
                self.assertTrue(text_sizes)
                self.assertTrue(min(text_sizes) >= 12)
                legend_labels = tuple(text.get_text() for text in legend.get_texts())
                if legend_labels == ("Direct LLM", "Agentic (No KG)"):
                    no_kg_max_text_sizes.append(max(text_sizes))
                else:
                    with_kg_max_text_sizes.append(max(text_sizes))

            self.assertTrue(any(size >= 13 for size in no_kg_max_text_sizes))
            self.assertTrue(any(size >= 13 for size in with_kg_max_text_sizes))

            self.assertTrue((out_dir / "agentic_no_kg_vs_direct.png").exists())
            self.assertTrue((out_dir / "agentic_with_kg_vs_direct.png").exists())

    def test_breakdown_chart_enlarges_text_and_markers_without_resizing_canvas(self):
        series_list = [
            {
                "label": "Model A (with kg)",
                "color": "#E63946",
                "summary": {
                    "Material design": {
                        "avg_correctness": 0.82,
                        "correct_rate": 0.75,
                        "count": 4,
                        "correct_count": 3,
                    },
                    "Device engineering": {
                        "avg_correctness": 0.68,
                        "correct_rate": 0.5,
                        "count": 4,
                        "correct_count": 2,
                    },
                },
            },
            {
                "label": "Model B (with kg)",
                "color": "#264653",
                "summary": {
                    "Material design": {
                        "avg_correctness": 0.79,
                        "correct_rate": 0.5,
                        "count": 4,
                        "correct_count": 2,
                    },
                    "Device engineering": {
                        "avg_correctness": 0.71,
                        "correct_rate": 0.5,
                        "count": 4,
                        "correct_count": 2,
                    },
                },
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "category_avg_correctness.png"
            with patch("plot_min4_eval.plt.close"):
                created = create_correctness_breakdown_chart(
                    series_list=series_list,
                    dimension="category",
                    analysis_metric="avg_correctness",
                    output_path=output_path,
                    title="With KG by Category",
                )

            fig = self._latest_figure()
            ax = fig.axes[0]
            legend = ax.get_legend()
            plotted_lines = [line for line in ax.lines if line.get_label() in {"Model A (with kg)", "Model B (with kg)"}]
            marker_sizes = [line.get_markersize() for line in plotted_lines]
            scatter_sizes = [collection.get_sizes()[0] for collection in ax.collections if len(collection.get_offsets())]

            self.assertTrue(created)
            self.assertAlmostEqual(CORRECTNESS_FIGSIZE[0], fig.get_size_inches()[0], places=2)
            self.assertAlmostEqual(CORRECTNESS_FIGSIZE[1], fig.get_size_inches()[1], places=2)
            self.assertEqual(17, ax.yaxis.label.get_fontsize())
            self.assertTrue(all(label.get_fontsize() == 14 for label in ax.get_xticklabels()))
            self.assertTrue(all(label.get_fontsize() == 14 for label in ax.get_yticklabels()))
            self.assertIsNotNone(legend)
            self.assertTrue(all(text.get_fontsize() == 14 for text in legend.get_texts()))
            self.assertTrue(all(size == 10.5 for size in marker_sizes))
            self.assertTrue(all(size == 115 for size in scatter_sizes))
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
