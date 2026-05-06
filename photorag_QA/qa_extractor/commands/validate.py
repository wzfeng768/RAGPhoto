"""Validate command - check output quality and errors."""

from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..stage1_extractor import ExtractionResult
from ..stage2_generator import GenerationResult
from ..ui.themes import get_theme, Icons
from ..ui.panels import CategoryChart


def validate_command(output_dir: str, console: Console, fix: bool = False) -> None:
    """Validate output files and report issues."""
    theme = get_theme()
    output_path = Path(output_dir)

    knowledge_dir = output_path / "knowledge"
    qa_dir = output_path / "qa_pairs"

    issues = []
    warnings = []

    # === Check Knowledge Files ===
    knowledge_results = []
    knowledge_errors = []

    if knowledge_dir.exists():
        for f in sorted(knowledge_dir.glob("*.json")):
            try:
                result = ExtractionResult.load(f)
                if "error" in result.token_usage:
                    knowledge_errors.append({
                        "file": f.name,
                        "error": result.token_usage.get("error", "Unknown error"),
                        "path": f,
                    })
                elif len(result.knowledge_points) == 0:
                    knowledge_errors.append({
                        "file": f.name,
                        "error": "No knowledge points extracted",
                        "path": f,
                    })
                else:
                    knowledge_results.append(result)
            except Exception as e:
                knowledge_errors.append({
                    "file": f.name,
                    "error": f"Failed to load: {str(e)[:50]}",
                    "path": f,
                })
    else:
        issues.append("Knowledge directory not found")

    # === Check QA Files ===
    qa_results = []
    qa_errors = []

    if qa_dir.exists():
        for f in sorted(qa_dir.glob("*.json")):
            if f.stem == "cross_doc":
                continue
            try:
                result = GenerationResult.load(f)
                if "error" in result.token_usage:
                    qa_errors.append({
                        "file": f.name,
                        "error": result.token_usage.get("error", "Unknown error"),
                        "path": f,
                    })
                elif len(result.qa_pairs) == 0:
                    qa_errors.append({
                        "file": f.name,
                        "error": "No QA pairs generated",
                        "path": f,
                    })
                else:
                    qa_results.append(result)
            except Exception as e:
                qa_errors.append({
                    "file": f.name,
                    "error": f"Failed to load: {str(e)[:50]}",
                    "path": f,
                })

    # === Summary Panel ===
    console.print()

    summary_table = Table.grid(padding=(0, 3))
    summary_table.add_column()
    summary_table.add_column(justify="right")
    summary_table.add_column()
    summary_table.add_column(justify="right")

    summary_table.add_row(
        f"[{theme.text}]Knowledge Files[/]",
        f"[{theme.primary}]{len(knowledge_results)}[/] OK",
        f"[{theme.text}]QA Files[/]",
        f"[{theme.primary}]{len(qa_results)}[/] OK",
    )
    summary_table.add_row(
        f"[{theme.text}]Knowledge Errors[/]",
        f"[{theme.error}]{len(knowledge_errors)}[/]" if knowledge_errors else f"[{theme.success}]0[/]",
        f"[{theme.text}]QA Errors[/]",
        f"[{theme.error}]{len(qa_errors)}[/]" if qa_errors else f"[{theme.success}]0[/]",
    )

    total_knowledge = sum(len(r.knowledge_points) for r in knowledge_results)
    total_qa = sum(len(r.qa_pairs) for r in qa_results)

    summary_table.add_row(
        f"[{theme.text}]Total Knowledge[/]",
        f"[{theme.primary}]{total_knowledge}[/] points",
        f"[{theme.text}]Total QA[/]",
        f"[{theme.primary}]{total_qa}[/] pairs",
    )

    console.print(Panel(
        summary_table,
        title=f"[{theme.title}]Validation Summary[/]",
        border_style=theme.border,
        padding=(0, 1),
    ))

    # === Error Details ===
    if knowledge_errors or qa_errors:
        console.print()

        error_table = Table(
            title="Files with Errors",
            title_style=theme.error,
            border_style=theme.border,
            show_header=True,
            header_style="bold",
        )
        error_table.add_column("Type", style=theme.text_dim, width=10)
        error_table.add_column("File", style=theme.text)
        error_table.add_column("Error", style=theme.error)

        for err in knowledge_errors[:10]:  # Limit to 10
            error_table.add_row(
                "Knowledge",
                err["file"][:30],
                err["error"][:40] + "..." if len(err["error"]) > 40 else err["error"],
            )

        for err in qa_errors[:10]:
            error_table.add_row(
                "QA",
                err["file"][:30],
                err["error"][:40] + "..." if len(err["error"]) > 40 else err["error"],
            )

        if len(knowledge_errors) + len(qa_errors) > 20:
            error_table.add_row(
                "...",
                f"({len(knowledge_errors) + len(qa_errors) - 20} more)",
                "",
            )

        console.print(error_table)

    # === Category Balance Check ===
    if knowledge_results:
        console.print()

        category_counts = Counter()
        for result in knowledge_results:
            for kp in result.knowledge_points:
                category_counts[kp.category] += 1

        # Check for missing or underrepresented categories
        expected_categories = [
            "Materials Design & Synthesis",
            "Performance Metrics",
            "Structure-Property Relationships",
            "Device Architecture & Physics",
            "Processing & Fabrication",
            "Characterization Methods",
            "Stability & Degradation",
            "Computational & Machine Learning",
        ]

        missing = [c for c in expected_categories if category_counts.get(c, 0) == 0]
        if missing:
            warnings.append(f"Missing categories: {', '.join(missing[:3])}" + ("..." if len(missing) > 3 else ""))

        chart = CategoryChart(dict(category_counts), "Knowledge Distribution")
        console.print(chart)

    # === QA Difficulty/Reasoning Balance ===
    if qa_results:
        console.print()

        difficulty_counts = Counter()
        reasoning_counts = Counter()

        for result in qa_results:
            for qa in result.qa_pairs:
                difficulty_counts[qa.difficulty] += 1
                reasoning_counts[qa.reasoning_type] += 1

        diff_chart = CategoryChart(dict(difficulty_counts), "Difficulty Distribution")
        console.print(diff_chart)

        console.print()
        reason_chart = CategoryChart(dict(reasoning_counts), "Reasoning Type Distribution")
        console.print(reason_chart)

    # === Warnings ===
    if warnings:
        console.print()
        for warning in warnings:
            console.print(f"[{theme.warning}]{Icons.WARNING} {warning}[/]")

    # === Fix Option ===
    if fix and (knowledge_errors or qa_errors):
        console.print()
        console.print(f"[{theme.info}]{Icons.INFO} Removing {len(knowledge_errors) + len(qa_errors)} error files for re-processing...[/]")

        for err in knowledge_errors:
            try:
                err["path"].unlink()
                console.print(f"  [{theme.text_dim}]Removed:[/] {err['file']}")
            except Exception:
                pass

        for err in qa_errors:
            try:
                err["path"].unlink()
                console.print(f"  [{theme.text_dim}]Removed:[/] {err['file']}")
            except Exception:
                pass

        console.print()
        console.print(f"[{theme.success}]{Icons.SUCCESS} Run the pipeline again to re-process these files.[/]")

    # === Final Status ===
    console.print()
    if not knowledge_errors and not qa_errors and not warnings:
        console.print(f"[{theme.success}]{Icons.SUCCESS} All files validated successfully![/]")
    elif knowledge_errors or qa_errors:
        console.print(f"[{theme.warning}]{Icons.WARNING} Found issues. Use --fix to remove error files for re-processing.[/]")

    console.print()
