"""Prompt templates for knowledge extraction and QA generation."""

from .extraction import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT
from .generation import GENERATION_SYSTEM_PROMPT, GENERATION_USER_PROMPT

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_USER_PROMPT",
    "GENERATION_SYSTEM_PROMPT",
    "GENERATION_USER_PROMPT",
]
