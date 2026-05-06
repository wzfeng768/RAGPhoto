"""ASCII banners and branding for QA Extractor CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

from .themes import Theme, get_theme, Icons


# ASCII art banner
BANNER_ART = r"""
 ██████╗  █████╗     ███████╗██╗  ██╗████████╗██████╗
██╔═══██╗██╔══██╗    ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗
██║   ██║███████║    █████╗   ╚███╔╝    ██║   ██████╔╝
██║▄▄ ██║██╔══██║    ██╔══╝   ██╔██╗    ██║   ██╔══██╗
╚██████╔╝██║  ██║    ███████╗██╔╝ ██╗   ██║   ██║  ██║
 ╚══▀▀═╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
"""

BANNER_SMALL = r"""
╔═╗ ╔═╗  ╔═╗─╗ ╦╔╦╗╦═╗
║═╬╗╠═╣  ║╣ ╔╩╦╝ ║ ╠╦╝
╚═╝╚╩ ╩  ╚═╝╩ ╚═ ╩ ╩╚═
"""

VERSION = "0.2.0"
TAGLINE = "PhotoRAG Knowledge Extraction Pipeline"


def print_banner(
    console: Console,
    theme: Theme | None = None,
    show_config: bool = False,
    config_info: dict | None = None,
) -> None:
    """Print the application banner."""
    theme = theme or get_theme()

    # Build banner text
    banner_text = Text()

    # Add ASCII art
    for line in BANNER_ART.strip().split("\n"):
        banner_text.append(line + "\n", style=theme.primary)

    # Add tagline
    banner_text.append("\n")
    banner_text.append(TAGLINE, style=f"bold {theme.text}")
    banner_text.append("\n")
    banner_text.append(f"v{VERSION}", style=theme.text_dim)

    # Center and wrap in panel
    centered = Align.center(banner_text)
    panel = Panel(
        centered,
        border_style=theme.border,
        padding=(1, 2),
    )

    console.print(panel)

    # Show config info if provided
    if show_config and config_info:
        _print_config_summary(console, config_info, theme)


def _print_config_summary(
    console: Console,
    config_info: dict,
    theme: Theme,
) -> None:
    """Print configuration summary below banner."""
    console.print()

    lines = [
        f"[{theme.text_dim}]{Icons.CONFIG}[/] Model:  [{theme.primary}]{config_info.get('model', 'N/A')}[/]",
        f"[{theme.text_dim}]{Icons.FOLDER}[/] Input:  [{theme.text}]{config_info.get('input', 'N/A')}[/]",
        f"[{theme.text_dim}]{Icons.FOLDER}[/] Output: [{theme.text}]{config_info.get('output', 'N/A')}[/]",
    ]

    for line in lines:
        console.print(f"  {line}")

    console.print()


def print_completion_banner(
    console: Console,
    theme: Theme | None = None,
) -> None:
    """Print completion banner."""
    theme = theme or get_theme()

    text = Text()
    text.append(f"{Icons.SUCCESS} ", style=theme.success)
    text.append("Pipeline Complete", style=f"bold {theme.success}")

    panel = Panel(
        Align.center(text),
        border_style=theme.success,
        padding=(0, 2),
    )

    console.print()
    console.print(panel)


def print_error_banner(
    console: Console,
    message: str,
    theme: Theme | None = None,
) -> None:
    """Print error banner."""
    theme = theme or get_theme()

    text = Text()
    text.append(f"{Icons.ERROR} ", style=theme.error)
    text.append("Error: ", style=f"bold {theme.error}")
    text.append(message, style=theme.text)

    panel = Panel(
        text,
        border_style=theme.error,
        title="[bold red]Error[/]",
        padding=(0, 1),
    )

    console.print(panel)


def print_warning_banner(
    console: Console,
    message: str,
    theme: Theme | None = None,
) -> None:
    """Print warning banner."""
    theme = theme or get_theme()

    text = Text()
    text.append(f"{Icons.WARNING} ", style=theme.warning)
    text.append(message, style=theme.text)

    panel = Panel(
        text,
        border_style=theme.warning,
        title="[bold yellow]Warning[/]",
        padding=(0, 1),
    )

    console.print(panel)
