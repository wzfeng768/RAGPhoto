"""
UI Utilities for Agentic RAG v2
Helper functions for rendering and formatting
"""

import re
from typing import Dict, Any


def format_latex_for_display(text: str) -> str:
    r"""
    Convert LaTeX math expressions for proper display in Streamlit
    
    Handles:
    - Display math: \[ ... \] -> $$ ... $$
    - Inline math: \( ... \) -> $ ... $
    - Already formatted: $...$ and $$...$$ (no change)
    
    Args:
        text: Text containing LaTeX expressions
        
    Returns:
        Formatted text ready for HTML display with MathJax
    """
    if not text:
        return text
    
    # Convert display math \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # Convert inline math \( ... \) to $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    # Handle bold math symbols: \mathbf{PCE} stays as is
    # MathJax will handle it
    
    return text


def format_document_content(content: str, max_length: int = 300) -> str:
    """
    Format document content for display
    
    Args:
        content: Raw document content
        max_length: Maximum length to display
        
    Returns:
        Formatted content
    """
    if not content:
        return ""
    
    # Truncate if too long
    if len(content) > max_length:
        content = content[:max_length] + "..."
    
    # Format LaTeX
    content = format_latex_for_display(content)
    
    return content


def highlight_query_terms(text: str, query: str) -> str:
    """
    Highlight query terms in text
    
    Args:
        text: Text to highlight in
        query: Query string
        
    Returns:
        Text with highlighted terms
    """
    words = query.lower().split()
    
    for word in words:
        if len(word) > 3:  # Only highlight words > 3 chars
            pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
            text = pattern.sub(r'<mark>\1</mark>', text)
    
    return text


def format_metric_value(value: Any, metric_type: str = "auto") -> str:
    """
    Format metric values for display
    
    Args:
        value: Metric value
        metric_type: Type of metric (auto, percentage, time, count)
        
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    if metric_type == "percentage" or (metric_type == "auto" and 0 <= value <= 1):
        return f"{value * 100:.1f}%"
    elif metric_type == "time":
        if value < 60:
            return f"{value:.1f}s"
        else:
            return f"{value / 60:.1f}m"
    elif metric_type == "count":
        return f"{int(value):,}"
    else:
        return f"{value:.2f}"


def create_progress_bar_html(value: float, max_value: float = 1.0, 
                             color: str = "#3b82f6", height: str = "8px") -> str:
    """
    Create HTML progress bar
    
    Args:
        value: Current value
        max_value: Maximum value
        color: Progress bar color
        height: Bar height
        
    Returns:
        HTML string
    """
    percentage = min(100, (value / max_value) * 100)
    
    return f"""
    <div style="width: 100%; background: #e2e8f0; border-radius: 4px; height: {height};">
        <div style="width: {percentage}%; background: {color}; border-radius: 4px; height: 100%; transition: width 0.3s;"></div>
    </div>
    """


def get_quality_color(score: float) -> str:
    """
    Get color based on quality score
    
    Args:
        score: Quality score (0-1)
        
    Returns:
        Color hex code
    """
    if score >= 0.8:
        return "#10b981"  # Green
    elif score >= 0.6:
        return "#f59e0b"  # Orange
    else:
        return "#ef4444"  # Red


def format_source_citation(source: str) -> str:
    """
    Format source path to citation
    
    Args:
        source: File path
        
    Returns:
        Formatted citation
    """
    from pathlib import Path
    
    # Extract filename
    filename = Path(source).name
    
    # Remove extension and format
    name = filename.rsplit('.', 1)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    
    return name.title()
