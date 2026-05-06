"""Enhanced stats command with visual charts."""

from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..stage1_extractor import ExtractionResult
from ..stage2_generator import GenerationResult
from ..ui.themes import get_theme, Icons
from ..ui.panels import CategoryChart, ResultsSummary


def stats_command(output_dir: str, console: Console, detailed: bool = False) -> None:
    """Show detailed statistics about generated QA pairs."""
    theme = get_theme()
    output_path = Path(output_dir)

    qa_dir = output_path / "qa_pairs"
    knowledge_dir = output_path / "knowledge"

    if not qa_dir.exists():
        console.print(f"[{theme.error}]{Icons.ERROR} No QA pairs found in {output_dir}[/]")
        return

    # Load all results
    qa_results = []
    for json_file in qa_dir.glob("*.json"):
        try:
            qa_results.append(GenerationResult.load(json_file))
        except Exception:
            pass

    knowledge_results = []
    if knowledge_dir.exists():
        for json_file in knowledge_dir.glob("*.json"):
            try:
                knowledge_results.append(ExtractionResult.load(json_file))
            except Exception:
                pass

    if not qa_results:
        console.print(f"[{theme.warning}]{Icons.WARNING} No QA pairs found[/]")
        return

    # Calculate statistics
    total_qa = sum(len(r.qa_pairs) for r in qa_results)
    total_knowledge = sum(len(r.knowledge_points) for r in knowledge_results)
    cross_doc_qa = 0

    # Separate cross-doc results
    regular_results = []
    for r in qa_results:
        if r.paper_id == "cross_doc":
            cross_doc_qa = len(r.qa_pairs)
        else:
            regular_results.append(r)

    # Distributions
    category_counts = Counter()
    difficulty_counts = Counter()
    reasoning_counts = Counter()
    qa_per_paper = []

    for result in regular_results:
        qa_per_paper.append(len(result.qa_pairs))
        for qa in result.qa_pairs:
            category_counts[qa.category] += 1
            difficulty_counts[qa.difficulty] += 1
            reasoning_counts[qa.reasoning_type] += 1

    # === Header ===
    console.print()
    console.rule(f"[{theme.title}]QA Extraction Statistics[/]")
    console.print()

    # === Summary Panel ===
    summary = ResultsSummary(
        papers_processed=len(regular_results),
        knowledge_points=total_knowledge,
        qa_pairs=total_qa - cross_doc_qa,
        cross_doc_qa=cross_doc_qa,
    )
    console.print(summary)
    console.print()

    # === Per-Paper Stats ===
    if qa_per_paper:
        avg_qa = sum(qa_per_paper) / len(qa_per_paper)
        min_qa = min(qa_per_paper)
        max_qa = max(qa_per_paper)

        stats_table = Table.grid(padding=(0, 4))
        stats_table.add_column()
        stats_table.add_column(justify="right")
        stats_table.add_column()
        stats_table.add_column(justify="right")

        stats_table.add_row(
            f"[{theme.text_dim}]Avg QA/Paper[/]",
            f"[{theme.primary}]{avg_qa:.1f}[/]",
            f"[{theme.text_dim}]Min/Max[/]",
            f"[{theme.primary}]{min_qa}/{max_qa}[/]",
        )

        console.print(Panel(
            stats_table,
            title=f"[{theme.title}]Per-Paper Statistics[/]",
            border_style=theme.border,
            padding=(0, 1),
        ))
        console.print()

    # === Category Distribution ===
    if category_counts:
        chart = CategoryChart(dict(category_counts), "Category Distribution")
        console.print(chart)
        console.print()

    # === Difficulty Distribution ===
    if difficulty_counts:
        chart = CategoryChart(dict(difficulty_counts), "Difficulty Distribution")
        console.print(chart)
        console.print()

    # === Reasoning Type Distribution ===
    if reasoning_counts:
        chart = CategoryChart(dict(reasoning_counts), "Reasoning Type Distribution")
        console.print(chart)
        console.print()

    # === Detailed Per-Paper Table ===
    if detailed and regular_results:
        paper_table = Table(
            title="Per-Paper Breakdown",
            title_style=theme.title,
            border_style=theme.border,
            show_header=True,
            header_style="bold",
        )
        paper_table.add_column("#", justify="right", style=theme.text_dim, width=4)
        paper_table.add_column("Paper Title", style=theme.text, max_width=40)
        paper_table.add_column("QA", justify="right", style=theme.primary)
        paper_table.add_column("Categories", style=theme.text_dim)

        for i, result in enumerate(regular_results[:20], 1):
            cats = Counter(qa.category for qa in result.qa_pairs)
            top_cats = ", ".join(c[:15] for c, _ in cats.most_common(2))

            title = result.paper_title[:38] if len(result.paper_title) > 38 else result.paper_title
            if len(result.paper_title) > 38:
                title += "..."

            paper_table.add_row(
                str(i),
                title,
                str(len(result.qa_pairs)),
                top_cats,
            )

        if len(regular_results) > 20:
            paper_table.add_row("...", f"({len(regular_results) - 20} more papers)", "", "")

        console.print(paper_table)
        console.print()
