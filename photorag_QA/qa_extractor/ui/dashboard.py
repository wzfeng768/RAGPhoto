"""Live dashboard for QA Extractor pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Callable

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .themes import Theme, get_theme, Icons
from .panels import (
    ConfigPanel,
    TokenPanel,
    ProgressPanel,
    TaskPanel,
    ActivityLog,
    ResultsSummary,
    CategoryChart,
)
from .banner import print_banner, print_completion_banner


class Dashboard:
    """Live dashboard for pipeline execution."""

    def __init__(
        self,
        console: Console,
        theme: Theme | None = None,
        config_info: dict | None = None,
    ):
        self.console = console
        self.theme = theme or get_theme()
        self.config_info = config_info or {}

        # Initialize panels
        self.config_panel = ConfigPanel(
            model=config_info.get("model", ""),
            input_dir=config_info.get("input", ""),
            output_dir=config_info.get("output", ""),
            resume=config_info.get("resume", True),
            file_count=config_info.get("file_count", 0),
        )

        self.token_panel = TokenPanel()
        self.progress_panel = ProgressPanel(stages={
            "Stage 1: Extract Knowledge": {"current": 0, "total": 0, "status": "pending"},
            "Stage 2: Generate QA": {"current": 0, "total": 0, "status": "pending"},
            "Stage 3: Cross-Doc QA": {"current": 0, "total": 1, "status": "pending"},
        })
        self.task_panel = TaskPanel()
        self.activity_log = ActivityLog(max_lines=5)

        self._live: Optional[Live] = None

    def _build_layout(self) -> Layout:
        """Build the dashboard layout."""
        layout = Layout()

        # Top row: Config and Token panels side by side
        top_row = Layout(name="top")
        top_row.split_row(
            Layout(self.config_panel, name="config"),
            Layout(self.token_panel, name="tokens"),
        )

        # Middle: Progress
        progress_row = Layout(self.progress_panel, name="progress", size=6)

        # Current task
        task_row = Layout(self.task_panel, name="task", size=6)

        # Activity log
        log_row = Layout(self.activity_log, name="log", size=9)

        layout.split_column(
            top_row,
            progress_row,
            task_row,
            log_row,
        )
        layout["top"].size = 7

        return layout

    def start(self) -> None:
        """Start the live dashboard."""
        print_banner(self.console, self.theme)
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live dashboard."""
        if self._live:
            self._live.stop()
            self._live = None

    def refresh(self) -> None:
        """Refresh the dashboard display."""
        if self._live:
            self._live.update(self._build_layout())

    # Update methods
    def update_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float,
        request_count: int = 0,
    ) -> None:
        """Update token usage panel."""
        self.token_panel.prompt_tokens = prompt_tokens
        self.token_panel.completion_tokens = completion_tokens
        self.token_panel.total_tokens = total_tokens
        self.token_panel.estimated_cost = estimated_cost
        self.token_panel.request_count = request_count
        self.refresh()

    def update_progress(
        self,
        stage: str,
        current: int,
        total: int,
        status: str = "in_progress",
    ) -> None:
        """Update progress for a stage."""
        stage_map = {
            "extract": "Stage 1: Extract Knowledge",
            "generate": "Stage 2: Generate QA",
            "cross_doc": "Stage 3: Cross-Doc QA",
        }
        stage_name = stage_map.get(stage, stage)

        if stage_name in self.progress_panel.stages:
            self.progress_panel.stages[stage_name] = {
                "current": current,
                "total": total,
                "status": status,
            }
        self.refresh()

    def update_task(
        self,
        filename: str = "",
        title: str = "",
        status: str = "processing",
        status_message: str = "",
    ) -> None:
        """Update current task panel."""
        self.task_panel.filename = filename
        self.task_panel.title = title
        self.task_panel.status = status
        self.task_panel.status_message = status_message
        self.refresh()

    def log(self, message: str, level: str = "info") -> None:
        """Add an entry to the activity log."""
        self.activity_log.add(message, level)
        self.refresh()

    def set_stage_total(self, stage: str, total: int) -> None:
        """Set the total count for a stage."""
        stage_map = {
            "extract": "Stage 1: Extract Knowledge",
            "generate": "Stage 2: Generate QA",
            "cross_doc": "Stage 3: Cross-Doc QA",
        }
        stage_name = stage_map.get(stage, stage)

        if stage_name in self.progress_panel.stages:
            self.progress_panel.stages[stage_name]["total"] = total
        self.refresh()

    def __enter__(self) -> "Dashboard":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


def print_results_summary(
    console: Console,
    papers: int,
    knowledge: int,
    qa_pairs: int,
    cross_doc: int,
    tokens: int,
    cost: float,
    duration: float,
    category_data: dict[str, int] | None = None,
    difficulty_data: dict[str, int] | None = None,
    theme: Theme | None = None,
) -> None:
    """Print the final results summary."""
    theme = theme or get_theme()

    # Completion banner
    print_completion_banner(console, theme)
    console.print()

    # Results summary
    results = ResultsSummary(
        papers_processed=papers,
        knowledge_points=knowledge,
        qa_pairs=qa_pairs,
        cross_doc_qa=cross_doc,
        total_tokens=tokens,
        estimated_cost=cost,
        duration_seconds=duration,
    )
    console.print(results)
    console.print()

    # Category distribution
    if category_data:
        chart = CategoryChart(category_data, "Category Distribution")
        console.print(chart)
        console.print()

    # Token/cost summary
    token_table = Table.grid(padding=(0, 4))
    token_table.add_column()
    token_table.add_column()
    token_table.add_column()
    token_table.add_column()

    token_table.add_row(
        f"[{theme.text_dim}]Total Tokens[/]",
        f"[{theme.primary}]{tokens:,}[/]",
        f"[{theme.text_dim}]Duration[/]",
        f"[{theme.primary}]{duration:.1f}s[/]",
    )
    token_table.add_row(
        f"[{theme.text_dim}]Est. Cost[/]",
        f"[{theme.success}]${cost:.2f}[/]",
        f"[{theme.text_dim}]Avg/Paper[/]",
        f"[{theme.primary}]{duration/papers:.1f}s[/]" if papers > 0 else "[dim]N/A[/]",
    )

    console.print(Panel(
        token_table,
        title=f"[{theme.title}]Token Usage & Cost[/]",
        border_style=theme.border,
        padding=(0, 1),
    ))
    console.print()
