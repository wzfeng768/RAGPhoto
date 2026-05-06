"""Status command - show current pipeline status."""

from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..checkpoint import CheckpointManager
from ..stage1_extractor import ExtractionResult
from ..stage2_generator import GenerationResult
from ..ui.themes import get_theme, Icons


def status_command(output_dir: str, console: Console) -> None:
    """Show current pipeline status and checkpoint information."""
    theme = get_theme()
    output_path = Path(output_dir)

    # Check directories
    knowledge_dir = output_path / "knowledge"
    qa_dir = output_path / "qa_pairs"

    # Load checkpoint
    checkpoint_manager = CheckpointManager(output_path)
    checkpoint = checkpoint_manager.load()

    # Count files and results
    knowledge_files = list(knowledge_dir.glob("*.json")) if knowledge_dir.exists() else []
    qa_files = [f for f in qa_dir.glob("*.json") if f.stem != "cross_doc"] if qa_dir.exists() else []
    cross_doc_file = qa_dir / "cross_doc.json" if qa_dir.exists() else None

    # Load and analyze results
    total_knowledge = 0
    total_qa = 0
    errors_knowledge = 0
    errors_qa = 0

    for f in knowledge_files:
        try:
            result = ExtractionResult.load(f)
            if "error" in result.token_usage:
                errors_knowledge += 1
            else:
                total_knowledge += len(result.knowledge_points)
        except Exception:
            errors_knowledge += 1

    for f in qa_files:
        try:
            result = GenerationResult.load(f)
            if "error" in result.token_usage:
                errors_qa += 1
            else:
                total_qa += len(result.qa_pairs)
        except Exception:
            errors_qa += 1

    cross_doc_qa = 0
    if cross_doc_file and cross_doc_file.exists():
        try:
            result = GenerationResult.load(cross_doc_file)
            cross_doc_qa = len(result.qa_pairs)
        except Exception:
            pass

    # Build status table
    table = Table.grid(padding=(0, 2))
    table.add_column("Stage", style=theme.text)
    table.add_column("Progress", style=theme.text)
    table.add_column("Files", justify="right")
    table.add_column("Results", justify="right")
    table.add_column("Errors", justify="right")

    # Header
    table.add_row(
        Text("Stage", style="bold"),
        Text("Progress", style="bold"),
        Text("Files", style="bold"),
        Text("Results", style="bold"),
        Text("Errors", style="bold"),
    )
    table.add_row("─" * 15, "─" * 12, "─" * 8, "─" * 12, "─" * 8)

    # Stage 1: Extract
    extract_status = _get_stage_status("extract", checkpoint, len(knowledge_files), errors_knowledge)
    table.add_row(
        "Extract",
        extract_status,
        f"{len(knowledge_files)}",
        f"{total_knowledge} points",
        f"[{theme.error}]{errors_knowledge}[/]" if errors_knowledge > 0 else f"[{theme.success}]0[/]",
    )

    # Stage 2: Generate
    generate_status = _get_stage_status("generate", checkpoint, len(qa_files), errors_qa)
    table.add_row(
        "Generate",
        generate_status,
        f"{len(qa_files)}",
        f"{total_qa} pairs",
        f"[{theme.error}]{errors_qa}[/]" if errors_qa > 0 else f"[{theme.success}]0[/]",
    )

    # Stage 3: Cross-Doc
    cross_doc_status = f"[{theme.success}]{Icons.COMPLETE} Done[/]" if cross_doc_qa > 0 else f"[{theme.text_dim}]{Icons.PENDING} Pending[/]"
    table.add_row(
        "Cross-Doc",
        cross_doc_status,
        "1" if cross_doc_qa > 0 else "0",
        f"{cross_doc_qa} pairs",
        f"[{theme.success}]0[/]",
    )

    # Panel
    console.print()
    console.print(Panel(
        table,
        title=f"[{theme.title}]Pipeline Status[/]",
        border_style=theme.border,
        padding=(1, 2),
    ))

    # Checkpoint info
    if checkpoint:
        info_lines = []
        info_lines.append(f"[{theme.text_dim}]Last Stage:[/]  [{theme.primary}]{checkpoint.stage}[/]")
        info_lines.append(f"[{theme.text_dim}]Last Update:[/] [{theme.text}]{checkpoint.timestamp}[/]")
        info_lines.append(f"[{theme.text_dim}]Checkpoint:[/]  [{theme.text}]{output_path / '.checkpoint.json'}[/]")

        if checkpoint.token_stats:
            tokens = checkpoint.token_stats.get("usage", {}).get("total_tokens", 0)
            cost = checkpoint.token_stats.get("estimated_cost_usd", 0)
            info_lines.append(f"[{theme.text_dim}]Tokens Used:[/] [{theme.primary}]{tokens:,}[/]")
            info_lines.append(f"[{theme.text_dim}]Est. Cost:[/]   [{theme.success}]${cost:.2f}[/]")

        console.print()
        console.print(Panel(
            "\n".join(info_lines),
            title=f"[{theme.title}]Checkpoint Info[/]",
            border_style=theme.border,
            padding=(0, 1),
        ))

    # Warnings
    if errors_knowledge > 0 or errors_qa > 0:
        console.print()
        console.print(f"[{theme.warning}]{Icons.WARNING} {errors_knowledge + errors_qa} file(s) with errors. Run 'validate' for details.[/]")

    console.print()


def _get_stage_status(stage: str, checkpoint, file_count: int, error_count: int) -> str:
    """Get status display for a stage."""
    theme = get_theme()

    if checkpoint and checkpoint.stage in ["complete"]:
        return f"[{theme.success}]{Icons.COMPLETE} Done[/]"
    elif checkpoint and checkpoint.stage == stage:
        return f"[{theme.warning}]{Icons.IN_PROGRESS} In Progress[/]"
    elif file_count > 0 and error_count == 0:
        return f"[{theme.success}]{Icons.COMPLETE} Done[/]"
    elif file_count > 0 and error_count > 0:
        return f"[{theme.warning}]{Icons.WARNING} Partial[/]"
    else:
        return f"[{theme.text_dim}]{Icons.PENDING} Pending[/]"
