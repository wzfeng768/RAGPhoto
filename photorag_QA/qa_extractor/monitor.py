"""Progress monitoring and statistics tracking."""

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .config import Config
from .llm_client import TokenStats


@dataclass
class StageProgress:
    """Progress tracking for a pipeline stage."""

    name: str
    total: int = 0
    completed: int = 0
    current_item: str = ""
    items_per_category: dict = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0
        return (self.completed / self.total) * 100


@dataclass
class PipelineStats:
    """Statistics for the entire pipeline."""

    stage1_progress: StageProgress = field(
        default_factory=lambda: StageProgress(name="Knowledge Extraction")
    )
    stage2_progress: StageProgress = field(
        default_factory=lambda: StageProgress(name="QA Generation")
    )
    token_stats: TokenStats = field(default_factory=TokenStats)
    total_knowledge_points: int = 0
    total_qa_pairs: int = 0
    category_distribution: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    errors: list = field(default_factory=list)

    def get_elapsed_time(self) -> str:
        """Get formatted elapsed time."""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


class ProgressMonitor:
    """Monitor and display pipeline progress using Rich."""

    def __init__(self, config: Config):
        self.config = config
        self.stats = PipelineStats()
        self.console = Console()
        self._live: Optional[Live] = None
        self._progress: Optional[Progress] = None
        self._stage1_task: Optional[TaskID] = None
        self._stage2_task: Optional[TaskID] = None

    def _create_progress(self) -> Progress:
        """Create Rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
        )

    def _create_config_panel(self) -> Panel:
        """Create configuration info panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model", self.config.llm.model)
        table.add_row("API", self.config.llm.base_url[:40] + "..." if len(self.config.llm.base_url) > 40 else self.config.llm.base_url)
        table.add_row("Temperature", str(self.config.llm.temperature))
        table.add_row("Target QA/Paper", f"{self.config.qa_settings.min_qa_per_paper}-{self.config.qa_settings.max_qa_per_paper}")
        table.add_row("Categories", str(len(self.config.categories)))

        return Panel(table, title="[bold]Configuration", border_style="blue")

    def _create_token_panel(self) -> Panel:
        """Create token usage panel."""
        stats = self.stats.token_stats
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Prompt Tokens", f"{stats.total_usage.prompt_tokens:,}")
        table.add_row("Completion Tokens", f"{stats.total_usage.completion_tokens:,}")
        table.add_row("Total Tokens", f"{stats.total_usage.total_tokens:,}")
        table.add_row("Est. Cost", f"${stats.estimate_cost():.2f}")
        table.add_row("Rate", f"{stats.get_rate():,.0f} tok/min")

        return Panel(table, title="[bold]Token Usage", border_style="green")

    def _create_stats_panel(self) -> Panel:
        """Create statistics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Knowledge Points", f"{self.stats.total_knowledge_points:,}")
        table.add_row("QA Pairs", f"{self.stats.total_qa_pairs:,}")
        table.add_row("Elapsed Time", self.stats.get_elapsed_time())

        if self.stats.errors:
            table.add_row("Errors", f"[red]{len(self.stats.errors)}[/red]")

        return Panel(table, title="[bold]Statistics", border_style="yellow")

    def _create_category_panel(self) -> Panel:
        """Create category distribution panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="white", justify="right")

        for category in self.config.categories:
            count = self.stats.category_distribution.get(category, 0)
            short_name = category.split(" & ")[0][:20]
            table.add_row(short_name, str(count))

        return Panel(table, title="[bold]Categories", border_style="magenta")

    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=4),
        )

        # Header
        layout["header"].update(
            Panel(
                Text("PhotoRAG QA Extractor", style="bold white", justify="center"),
                border_style="bright_blue",
            )
        )

        # Body split into left and right
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split(
            Layout(name="progress", size=8),
            Layout(name="tokens"),
        )

        layout["right"].split(
            Layout(name="config"),
            Layout(name="stats"),
        )

        return layout

    def start(self, total_papers: int) -> None:
        """Start the progress monitor."""
        self.stats = PipelineStats()
        self.stats.stage1_progress.total = total_papers
        self.stats.stage2_progress.total = total_papers

        self._progress = self._create_progress()
        self._stage1_task = self._progress.add_task(
            "Stage 1: Extracting Knowledge", total=total_papers
        )
        self._stage2_task = self._progress.add_task(
            "Stage 2: Generating QA Pairs", total=total_papers
        )

    def update_stage1(
        self,
        completed: int,
        current_item: str = "",
        knowledge_count: int = 0,
        category_counts: Optional[dict] = None,
    ) -> None:
        """Update Stage 1 progress."""
        self.stats.stage1_progress.completed = completed
        self.stats.stage1_progress.current_item = current_item
        self.stats.total_knowledge_points += knowledge_count

        if category_counts:
            for cat, count in category_counts.items():
                self.stats.category_distribution[cat] = (
                    self.stats.category_distribution.get(cat, 0) + count
                )

        if self._progress and self._stage1_task is not None:
            self._progress.update(self._stage1_task, completed=completed)

    def update_stage2(
        self,
        completed: int,
        current_item: str = "",
        qa_count: int = 0,
    ) -> None:
        """Update Stage 2 progress."""
        self.stats.stage2_progress.completed = completed
        self.stats.stage2_progress.current_item = current_item
        self.stats.total_qa_pairs += qa_count

        if self._progress and self._stage2_task is not None:
            self._progress.update(self._stage2_task, completed=completed)

    def update_tokens(self, token_stats: TokenStats) -> None:
        """Update token statistics."""
        self.stats.token_stats = token_stats

    def add_error(self, error: str) -> None:
        """Record an error."""
        self.stats.errors.append(error)

    def get_progress(self) -> Progress:
        """Get the Rich progress object for display."""
        if self._progress is None:
            self._progress = self._create_progress()
        return self._progress

    def print_summary(self) -> None:
        """Print final summary."""
        self.console.print()
        self.console.rule("[bold blue]Pipeline Complete")
        self.console.print()

        # Summary table
        table = Table(title="Summary", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Papers Processed", f"{self.stats.stage1_progress.completed}")
        table.add_row("Knowledge Points Extracted", f"{self.stats.total_knowledge_points:,}")
        table.add_row("QA Pairs Generated", f"{self.stats.total_qa_pairs:,}")
        table.add_row("Total Tokens Used", f"{self.stats.token_stats.total_usage.total_tokens:,}")
        table.add_row("Estimated Cost", f"${self.stats.token_stats.estimate_cost():.2f}")
        table.add_row("Total Time", self.stats.get_elapsed_time())

        if self.stats.errors:
            table.add_row("Errors", f"[red]{len(self.stats.errors)}[/red]")

        self.console.print(table)
        self.console.print()

        # Category distribution
        if self.stats.category_distribution:
            cat_table = Table(title="Category Distribution", show_header=True)
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="white", justify="right")

            for category, count in sorted(
                self.stats.category_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                cat_table.add_row(category, str(count))

            self.console.print(cat_table)
