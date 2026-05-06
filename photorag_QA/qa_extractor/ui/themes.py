"""Color themes and styles for QA Extractor CLI."""

from dataclasses import dataclass
from rich.style import Style
from rich.theme import Theme as RichTheme


@dataclass
class Theme:
    """Color theme for CLI components."""

    # Primary colors
    primary: str = "cyan"
    secondary: str = "blue"
    accent: str = "magenta"

    # Status colors
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    info: str = "cyan"

    # Text colors
    text: str = "white"
    text_dim: str = "dim white"
    text_muted: str = "bright_black"

    # Panel colors
    border: str = "bright_black"
    title: str = "bold cyan"

    # Progress colors
    progress_complete: str = "green"
    progress_remaining: str = "bright_black"
    progress_current: str = "yellow"

    # Chart colors
    bar_fill: str = "cyan"
    bar_empty: str = "bright_black"

    def to_rich_theme(self) -> RichTheme:
        """Convert to Rich Theme."""
        return RichTheme({
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "info": self.info,
            "text": self.text,
            "text.dim": self.text_dim,
            "text.muted": self.text_muted,
            "border": self.border,
            "title": self.title,
            "progress.complete": self.progress_complete,
            "progress.remaining": self.progress_remaining,
            "progress.current": self.progress_current,
            "bar.fill": self.bar_fill,
            "bar.empty": self.bar_empty,
        })


# Predefined themes
THEMES = {
    "default": Theme(),
    "ocean": Theme(
        primary="dodger_blue2",
        secondary="deep_sky_blue1",
        accent="turquoise2",
        bar_fill="dodger_blue2",
    ),
    "forest": Theme(
        primary="green3",
        secondary="dark_sea_green",
        accent="spring_green2",
        bar_fill="green3",
    ),
    "sunset": Theme(
        primary="orange1",
        secondary="dark_orange",
        accent="gold1",
        bar_fill="orange1",
    ),
}


def get_theme(name: str = "default") -> Theme:
    """Get a theme by name."""
    return THEMES.get(name, THEMES["default"])


# Status icons
class Icons:
    """Unicode icons for CLI display."""

    # Status
    SUCCESS = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"
    PENDING = "○"
    IN_PROGRESS = "◐"
    COMPLETE = "●"
    SPINNER = "⏳"

    # Items
    PAPER = "📄"
    KNOWLEDGE = "🧠"
    QA = "❓"
    LINK = "🔗"
    FOLDER = "📁"
    CONFIG = "⚙"

    # Actions
    ARROW_RIGHT = "▸"
    ARROW_DOWN = "▾"
    BULLET = "•"

    # Progress
    BAR_FULL = "█"
    BAR_HALF = "▓"
    BAR_EMPTY = "░"


# Box drawing characters
class Box:
    """Box drawing characters for panels."""

    # Single line
    HORIZONTAL = "─"
    VERTICAL = "│"
    TOP_LEFT = "┌"
    TOP_RIGHT = "┐"
    BOTTOM_LEFT = "└"
    BOTTOM_RIGHT = "┘"

    # Rounded corners
    ROUND_TOP_LEFT = "╭"
    ROUND_TOP_RIGHT = "╮"
    ROUND_BOTTOM_LEFT = "╰"
    ROUND_BOTTOM_RIGHT = "╯"

    # T-junctions
    T_RIGHT = "├"
    T_LEFT = "┤"
    T_DOWN = "┬"
    T_UP = "┴"
    CROSS = "┼"
