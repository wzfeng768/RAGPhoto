"""
Tools module for Agentic RAG v2
Contains all tools that the agent can use
"""

from .base_tool import BaseTool
from .web_search import WebSearchTool

__all__ = [
    'BaseTool',
    'WebSearchTool',
]

