"""Utility functions for QA Extractor."""

import hashlib
import re
from pathlib import Path
from typing import Optional


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Sanitize a string to be used as a filename."""
    # Remove special characters
    sanitized = re.sub(r"[^\w\s-]", "", name)
    # Replace whitespace with underscores
    sanitized = re.sub(r"[\s]+", "_", sanitized)
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def generate_paper_id(file_path: Path) -> str:
    """Generate a unique paper ID from file path."""
    name = file_path.stem
    sanitized = sanitize_filename(name)
    return sanitized


def generate_hash_id(content: str, length: int = 8) -> str:
    """Generate a short hash ID from content."""
    hash_obj = hashlib.md5(content.encode())
    return hash_obj.hexdigest()[:length]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_title_from_markdown(content: str) -> Optional[str]:
    """Extract the title from markdown content."""
    lines = content.split("\n")
    for line in lines[:20]:  # Check first 20 lines
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def clean_markdown_content(content: str) -> str:
    """Clean markdown content for processing."""
    # Remove excessive whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Remove image markdown but keep alt text
    content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[Figure: \1]", content)

    # Remove HTML comments
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

    return content.strip()


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (approximately 4 chars per token)."""
    return len(text) // 4


def format_number(num: int) -> str:
    """Format a number with thousand separators."""
    return f"{num:,}"


def format_cost(cost: float) -> str:
    """Format cost in USD."""
    return f"${cost:.2f}"
