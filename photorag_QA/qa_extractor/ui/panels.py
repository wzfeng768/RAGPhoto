"""Reusable panel components for QA Extractor CLI."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn

from .themes import Theme, get_theme, Icons


@dataclass
class ConfigPanel:
    """Configuration display panel."""

    model: str = ""
    input_dir: str = ""
    output_dir: str = ""
    resume: bool = True
    file_count: int = 0

    def __rich__(self) -> Panel:
        """Render as Rich Panel."""
        theme = get_theme()

        table = Table.grid(padding=(0, 2))
        table.add_column(style=theme.text_dim)
        table.add_column(style=theme.text)

        table.add_row("Model:", f"[{theme.primary}]{self.model}[/]")
        table.add_row("Input:", f"{self.input_dir} ({self.file_count} files)")
        table.add_row("Output:", self.output_dir)
        table.add_row("Resume:", f"[{theme.success}]enabled[/]" if self.resume else f"[{theme.warning}]disabled[/]")

        return Panel(
            table,
            title=f"[{theme.title}]Configuration[/]",
            border_style=theme.border,
            padding=(0, 1),
        )


@dataclass
class TokenPanel:
    """Token usage and cost tracking panel."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    request_count: int = 0

    def __rich__(self) -> Panel:
        """Render as Rich Panel."""
        theme = get_theme()

        table = Table.grid(padding=(0, 2))
        table.add_column(style=theme.text_dim)
        table.add_column(justify="right", style=theme.text)

        table.add_row(f"{Icons.ARROW_RIGHT} Prompt:", f"{self.prompt_tokens:,} tokens")
        table.add_row(f"{Icons.ARROW_RIGHT} Completion:", f"{self.completion_tokens:,} tokens")
        table.add_row(f"{Icons.ARROW_RIGHT} Total:", f"[{theme.primary}]{self.total_tokens:,}[/] tokens")
        table.add_row(f"{Icons.ARROW_RIGHT} Est. Cost:", f"[{theme.success}]${self.estimated_cost:.2f}[/]")

        return Panel(
            table,
            title=f"[{theme.title}]Token Usage[/]",
            border_style=theme.border,
            padding=(0, 1),
        )


@dataclass
class ProgressPanel:
    """Multi-stage progress panel."""

    stages: dict = field(default_factory=dict)
    # stages format: {"stage_name": {"current": 0, "total": 10, "status": "in_progress"}}

    def __rich__(self) -> Panel:
        """Render as Rich Panel."""
        theme = get_theme()

        lines = []
        for name, data in self.stages.items():
            current = data.get("current", 0)
            total = data.get("total", 0)
            status = data.get("status", "pending")

            # Build progress bar
            if total > 0:
                pct = current / total
                bar_width = 20
                filled = int(pct * bar_width)
                bar = Icons.BAR_FULL * filled + Icons.BAR_EMPTY * (bar_width - filled)

                # Color based on status
                if status == "complete":
                    bar_style = theme.success
                    icon = Icons.SUCCESS
                elif status == "in_progress":
                    bar_style = theme.progress_current
                    icon = Icons.SPINNER
                else:
                    bar_style = theme.text_dim
                    icon = Icons.PENDING

                line = Text()
                line.append(f"{icon} ", style=bar_style)
                line.append(f"{name:<25}", style=theme.text)
                line.append(f"[{bar_style}]{bar}[/]  ")
                line.append(f"{current}/{total}", style=theme.text_dim)
                line.append(f"  {pct*100:>3.0f}%", style=bar_style)
            else:
                line = Text()
                line.append(f"{Icons.PENDING} ", style=theme.text_dim)
                line.append(f"{name:<25}", style=theme.text_dim)
                line.append("waiting...", style=theme.text_dim)

            lines.append(line)

        return Panel(
            Group(*lines),
            title=f"[{theme.title}]Progress[/]",
            border_style=theme.border,
            padding=(0, 1),
        )


@dataclass
class TaskPanel:
    """Current task display panel."""

    filename: str = ""
    title: str = ""
    status: str = "idle"
    status_message: str = ""

    def __rich__(self) -> Panel:
        """Render as Rich Panel."""
        theme = get_theme()

        if not self.filename:
            content = Text("Waiting for task...", style=theme.text_dim)
        else:
            lines = []

            # Filename
            line1 = Text()
            line1.append(f"{Icons.PAPER} ", style=theme.primary)
            line1.append("Processing: ", style=theme.text_dim)
            line1.append(self.filename[:60], style=theme.text)
            if len(self.filename) > 60:
                line1.append("...", style=theme.text_dim)
            lines.append(line1)

            # Title
            if self.title:
                line2 = Text()
                line2.append("   Title: ", style=theme.text_dim)
                display_title = self.title[:55] if len(self.title) > 55 else self.title
                line2.append(display_title, style=theme.text)
                if len(self.title) > 55:
                    line2.append("...", style=theme.text_dim)
                lines.append(line2)

            # Status
            line3 = Text()
            line3.append("   Status: ", style=theme.text_dim)

            status_icon = {
                "idle": Icons.PENDING,
                "processing": Icons.SPINNER,
                "calling_api": Icons.SPINNER,
                "success": Icons.SUCCESS,
                "error": Icons.ERROR,
            }.get(self.status, Icons.PENDING)

            status_style = {
                "idle": theme.text_dim,
                "processing": theme.warning,
                "calling_api": theme.info,
                "success": theme.success,
                "error": theme.error,
            }.get(self.status, theme.text_dim)

            line3.append(f"{status_icon} ", style=status_style)
            line3.append(self.status_message or self.status, style=status_style)
            lines.append(line3)

            content = Group(*lines)

        return Panel(
            content,
            title=f"[{theme.title}]Current Task[/]",
            border_style=theme.border,
            padding=(0, 1),
        )


class ActivityLog:
    """Scrolling activity log panel."""

    def __init__(self, max_lines: int = 6):
        self.max_lines = max_lines
        self.entries: deque = deque(maxlen=max_lines)

    def add(self, message: str, level: str = "info") -> None:
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.entries.append((timestamp, level, message))

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()

    def __rich__(self) -> Panel:
        """Render as Rich Panel."""
        theme = get_theme()

        if not self.entries:
            content = Text("No activity yet...", style=theme.text_dim)
        else:
            lines = []
            for timestamp, level, message in self.entries:
                line = Text()
                line.append(f"{timestamp} ", style=theme.text_dim)
                line.append("│ ", style=theme.border)

                icon = {
                    "info": Icons.INFO,
                    "success": Icons.SUCCESS,
                    "warning": Icons.WARNING,
                    "error": Icons.ERROR,
                }.get(level, Icons.BULLET)

                style = {
                    "info": theme.text,
                    "success": theme.success,
                    "warning": theme.warning,
                    "error": theme.error,
                }.get(level, theme.text)

                line.append(f"{icon} ", style=style)
                line.append(message[:60], style=theme.text)
                if len(message) > 60:
                    line.append("...", style=theme.text_dim)
                lines.append(line)

            content = Group(*lines)

        return Panel(
            content,
            title=f"[{theme.title}]Activity Log[/]",
            border_style=theme.border,
            padding=(0, 1),
        )


@dataclass
class ResultsSummary:
    """Results summary panel for completion display."""

    papers_processed: int = 0
    knowledge_points: int = 0
    qa_pairs: int = 0
    cross_doc_qa: int = 0
    duration_seconds: float = 0
    total_tokens: int = 0
    estimated_cost: float = 0

    def __rich__(self) -> Panel:
        """Render as Rich Panel."""
        theme = get_theme()

        # Left side - counts
        left = Table.grid(padding=(0, 3))
        left.add_column()
        left.add_column(justify="right")

        left.add_row(
            f"[{theme.text}]{Icons.PAPER} Papers Processed[/]",
            f"[{theme.primary}]{self.papers_processed}[/]"
        )
        left.add_row(
            f"[{theme.text}]{Icons.KNOWLEDGE} Knowledge Points[/]",
            f"[{theme.primary}]{self.knowledge_points}[/]"
        )
        left.add_row(
            f"[{theme.text}]{Icons.QA} QA Pairs Generated[/]",
            f"[{theme.primary}]{self.qa_pairs}[/]"
        )
        left.add_row(
            f"[{theme.text}]{Icons.LINK} Cross-Doc QAs[/]",
            f"[{theme.primary}]{self.cross_doc_qa}[/]"
        )

        return Panel(
            left,
            title=f"[{theme.title}]Results Summary[/]",
            border_style=theme.border,
            padding=(0, 1),
        )


class CategoryChart:
    """Bar chart for category distribution."""

    def __init__(self, data: dict[str, int], title: str = "Category Distribution"):
        self.data = data
        self.title = title

    def __rich__(self) -> Panel:
        """Render as Rich Panel."""
        theme = get_theme()

        if not self.data:
            return Panel(
                Text("No data available", style=theme.text_dim),
                title=f"[{theme.title}]{self.title}[/]",
                border_style=theme.border,
            )

        total = sum(self.data.values())
        max_val = max(self.data.values()) if self.data else 1
        bar_width = 20

        lines = []
        for name, count in sorted(self.data.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            bar_len = int((count / max_val) * bar_width) if max_val > 0 else 0

            bar = Icons.BAR_FULL * bar_len + Icons.BAR_EMPTY * (bar_width - bar_len)

            line = Text()
            # Truncate long names
            display_name = name[:22] if len(name) > 22 else name
            line.append(f"{display_name:<22} ", style=theme.text)
            line.append(f"[{theme.bar_fill}]{bar}[/] ")
            line.append(f"{count:>3}", style=theme.primary)
            line.append(f"  ({pct:>5.1f}%)", style=theme.text_dim)
            lines.append(line)

        return Panel(
            Group(*lines),
            title=f"[{theme.title}]{self.title}[/]",
            border_style=theme.border,
            padding=(0, 1),
        )
