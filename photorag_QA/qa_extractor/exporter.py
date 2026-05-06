"""Export QA pairs to various formats."""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from .config import Config
from .stage2_generator import GenerationResult, QAPair


class QAExporter:
    """Export QA pairs to various formats."""

    def __init__(self, config: Config):
        self.config = config

    def _load_all_results(self, qa_dir: Path) -> list[GenerationResult]:
        """Load all generation results from directory."""
        results = []
        for json_file in qa_dir.glob("*.json"):
            try:
                result = GenerationResult.load(json_file)
                results.append(result)
            except Exception:
                continue
        return results

    def _collect_all_qa_pairs(
        self, results: list[GenerationResult]
    ) -> list[dict]:
        """Collect all QA pairs with IDs."""
        all_pairs = []
        qa_id = 1

        for result in results:
            for qa in result.qa_pairs:
                qa_dict = qa.to_dict()
                qa_dict["id"] = f"qa_{qa_id:04d}"
                all_pairs.append(qa_dict)
                qa_id += 1

        return all_pairs

    def _calculate_stats(
        self, qa_pairs: list[dict], results: list[GenerationResult]
    ) -> dict:
        """Calculate statistics for the export."""
        category_counts = Counter(qa["category"] for qa in qa_pairs)
        difficulty_counts = Counter(qa["difficulty"] for qa in qa_pairs)
        reasoning_counts = Counter(qa["reasoning_type"] for qa in qa_pairs)

        return {
            "total_qa_pairs": len(qa_pairs),
            "total_papers": len(results),
            "avg_qa_per_paper": len(qa_pairs) / len(results) if results else 0,
            "category_distribution": dict(category_counts),
            "difficulty_distribution": dict(difficulty_counts),
            "reasoning_type_distribution": dict(reasoning_counts),
        }

    def export_json(
        self,
        qa_pairs: list[dict],
        output_path: Path,
        meta: Optional[dict] = None,
    ) -> None:
        """Export QA pairs to JSON format."""
        output = {
            "meta": meta or {},
            "qa_pairs": qa_pairs,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def export_jsonl(
        self,
        qa_pairs: list[dict],
        output_path: Path,
    ) -> None:
        """Export QA pairs to JSONL format (one JSON object per line)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    def export_by_category(
        self,
        qa_pairs: list[dict],
        output_dir: Path,
        format: str = "json",
    ) -> dict[str, int]:
        """Export QA pairs split by category."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group by category
        by_category: dict[str, list[dict]] = {}
        for qa in qa_pairs:
            category = qa["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(qa)

        # Export each category
        category_counts = {}
        for category, pairs in by_category.items():
            # Sanitize category name for filename
            filename = category.lower().replace(" & ", "_").replace(" ", "_")

            if format == "jsonl":
                output_path = output_dir / f"{filename}.jsonl"
                self.export_jsonl(pairs, output_path)
            else:
                output_path = output_dir / f"{filename}.json"
                self.export_json(pairs, output_path)

            category_counts[category] = len(pairs)

        return category_counts

    def export_all(
        self,
        qa_dir: Path,
        output_path: Path,
        format: str = "json",
        split_by_category: bool = False,
    ) -> dict:
        """Export all QA pairs with optional category split."""
        # Load all results
        results = self._load_all_results(qa_dir)

        if not results:
            raise ValueError(f"No QA results found in {qa_dir}")

        # Collect all QA pairs
        qa_pairs = self._collect_all_qa_pairs(results)

        # Calculate statistics
        stats = self._calculate_stats(qa_pairs, results)

        # Create metadata
        meta = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": self.config.llm.model,
            "total_papers": len(results),
            "total_qa_pairs": len(qa_pairs),
            "categories": len(self.config.categories),
            "statistics": stats,
        }

        # Export main file
        if format == "jsonl":
            self.export_jsonl(qa_pairs, output_path)
            # Also save meta separately for JSONL
            meta_path = output_path.with_suffix(".meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        else:
            self.export_json(qa_pairs, output_path, meta=meta)

        # Export by category if requested
        if split_by_category:
            category_dir = output_path.parent / "qa_by_category"
            self.export_by_category(qa_pairs, category_dir, format=format)

        return {
            "total_qa_pairs": len(qa_pairs),
            "total_papers": len(results),
            "output_path": str(output_path),
            "statistics": stats,
        }


def create_summary_report(
    output_dir: Path,
    qa_pairs: list[dict],
    results: list[GenerationResult],
    config: Config,
) -> str:
    """Create a human-readable summary report."""
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    category_counts = Counter(qa["category"] for qa in qa_pairs)
    difficulty_counts = Counter(qa["difficulty"] for qa in qa_pairs)
    reasoning_counts = Counter(qa["reasoning_type"] for qa in qa_pairs)

    report_lines = [
        "# QA Extraction Summary Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Total Papers Processed**: {len(results)}",
        f"- **Total QA Pairs Generated**: {len(qa_pairs)}",
        f"- **Average QA per Paper**: {len(qa_pairs) / len(results):.1f}" if results else "- **Average QA per Paper**: N/A",
        "",
        "## Category Distribution",
        "",
    ]

    for category, count in category_counts.most_common():
        pct = (count / len(qa_pairs) * 100) if qa_pairs else 0
        report_lines.append(f"- {category}: {count} ({pct:.1f}%)")

    report_lines.extend([
        "",
        "## Difficulty Distribution",
        "",
    ])

    for difficulty, count in difficulty_counts.most_common():
        pct = (count / len(qa_pairs) * 100) if qa_pairs else 0
        report_lines.append(f"- {difficulty}: {count} ({pct:.1f}%)")

    report_lines.extend([
        "",
        "## Reasoning Type Distribution",
        "",
    ])

    for reasoning, count in reasoning_counts.most_common():
        pct = (count / len(qa_pairs) * 100) if qa_pairs else 0
        report_lines.append(f"- {reasoning}: {count} ({pct:.1f}%)")

    report_lines.extend([
        "",
        "## Configuration",
        "",
        f"- Model: {config.llm.model}",
        f"- Temperature: {config.llm.temperature}",
        f"- Target QA per Paper: {config.qa_settings.min_qa_per_paper}-{config.qa_settings.max_qa_per_paper}",
        "",
    ])

    report = "\n".join(report_lines)

    # Save report
    report_path = stats_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return str(report_path)
