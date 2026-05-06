"""Stage 1: Knowledge Point Extraction from academic papers."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config
from .llm_client import LLMClient, LLMResponse
from .prompts.extraction import format_extraction_prompt


@dataclass
class KnowledgePoint:
    """A single knowledge point extracted from a paper."""

    category: str
    content: str
    evidence: str
    complexity: str
    keywords: list[str]

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "content": self.content,
            "evidence": self.evidence,
            "complexity": self.complexity,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgePoint":
        return cls(
            category=data.get("category", ""),
            content=data.get("content", ""),
            evidence=data.get("evidence", ""),
            complexity=data.get("complexity", "single-hop"),
            keywords=data.get("keywords", []),
        )


@dataclass
class ExtractionResult:
    """Result of knowledge extraction from a single paper."""

    paper_id: str
    paper_title: str
    source_file: str
    knowledge_points: list[KnowledgePoint]
    token_usage: dict

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "source_file": self.source_file,
            "knowledge_points": [kp.to_dict() for kp in self.knowledge_points],
            "token_usage": self.token_usage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionResult":
        return cls(
            paper_id=data.get("paper_id", ""),
            paper_title=data.get("paper_title", ""),
            source_file=data.get("source_file", ""),
            knowledge_points=[
                KnowledgePoint.from_dict(kp) for kp in data.get("knowledge_points", [])
            ],
            token_usage=data.get("token_usage", {}),
        )

    def save(self, output_dir: Path) -> Path:
        """Save extraction result to JSON file."""
        output_path = output_dir / f"{self.paper_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return output_path

    @classmethod
    def load(cls, path: Path) -> "ExtractionResult":
        """Load extraction result from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class KnowledgeExtractor:
    """Extract knowledge points from academic papers."""

    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
        self.valid_categories = set(config.categories)

    def _generate_paper_id(self, file_path: Path) -> str:
        """Generate a unique paper ID from file path."""
        # Use stem of the file name, sanitized
        name = file_path.stem
        # Remove special characters and limit length
        sanitized = re.sub(r"[^\w\s-]", "", name)
        sanitized = re.sub(r"[\s]+", "_", sanitized)
        return sanitized[:100]

    def _extract_title_from_content(self, content: str) -> str:
        """Extract paper title from markdown content."""
        lines = content.split("\n")
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return "Unknown Title"

    def _preprocess_content(self, content: str) -> str:
        """Preprocess markdown content for extraction."""
        # Remove excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Remove image references (keep alt text if informative)
        content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[Figure: \1]", content)

        # Limit content length to avoid token limits
        max_chars = 50000  # Approximately 12-15k tokens
        if len(content) > max_chars:
            # Try to keep introduction and results/conclusion sections
            content = content[:max_chars]

        return content.strip()

    def _validate_knowledge_points(
        self, knowledge_points: list[dict]
    ) -> list[KnowledgePoint]:
        """Validate and filter knowledge points."""
        validated = []
        for kp in knowledge_points:
            # Check required fields
            if not kp.get("content") or not kp.get("category"):
                continue

            # Validate category
            category = kp.get("category", "")
            if category not in self.valid_categories:
                # Try to find closest match
                for valid_cat in self.valid_categories:
                    if valid_cat.lower() in category.lower() or category.lower() in valid_cat.lower():
                        category = valid_cat
                        break
                else:
                    continue

            # Validate complexity
            complexity = kp.get("complexity", "single-hop")
            if complexity not in ["single-hop", "multi-hop"]:
                complexity = "single-hop"

            validated.append(
                KnowledgePoint(
                    category=category,
                    content=kp.get("content", ""),
                    evidence=kp.get("evidence", ""),
                    complexity=complexity,
                    keywords=kp.get("keywords", []),
                )
            )

        return validated

    def extract_from_file(self, file_path: Path) -> ExtractionResult:
        """Extract knowledge points from a single markdown file."""
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Generate paper ID
        paper_id = self._generate_paper_id(file_path)

        # Extract title
        paper_title = self._extract_title_from_content(content)

        # Preprocess content
        processed_content = self._preprocess_content(content)

        # Generate extraction prompt
        messages = format_extraction_prompt(processed_content)

        # Call LLM
        try:
            parsed_response, response = self.llm_client.chat_json(messages)
        except Exception as e:
            # Return empty result on error
            return ExtractionResult(
                paper_id=paper_id,
                paper_title=paper_title,
                source_file=str(file_path),
                knowledge_points=[],
                token_usage={"error": str(e)},
            )

        # Extract knowledge points from response
        raw_knowledge_points = parsed_response.get("knowledge_points", [])

        # Use title from response if available
        if parsed_response.get("paper_title"):
            paper_title = parsed_response["paper_title"]

        # Validate knowledge points
        knowledge_points = self._validate_knowledge_points(raw_knowledge_points)

        return ExtractionResult(
            paper_id=paper_id,
            paper_title=paper_title,
            source_file=str(file_path),
            knowledge_points=knowledge_points,
            token_usage=response.usage.to_dict(),
        )

    def extract_from_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> list[ExtractionResult]:
        """Extract knowledge points from all markdown files in a directory."""
        # Find all markdown files
        md_files = list(input_dir.rglob("*.md"))

        results = []
        for i, file_path in enumerate(md_files):
            # Check if already processed
            paper_id = self._generate_paper_id(file_path)
            output_path = output_dir / f"{paper_id}.json"

            if output_path.exists():
                # Load existing result
                result = ExtractionResult.load(output_path)
            else:
                # Extract and save
                result = self.extract_from_file(file_path)
                result.save(output_dir)

            results.append(result)

            # Progress callback
            if progress_callback:
                progress_callback(
                    current=i + 1,
                    total=len(md_files),
                    paper_title=result.paper_title,
                    knowledge_count=len(result.knowledge_points),
                )

        return results
