"""Stage 2: QA Pair Generation from knowledge points."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config
from .llm_client import LLMClient
from .prompts.generation import format_generation_prompt, format_cross_doc_prompt
from .stage1_extractor import ExtractionResult


@dataclass
class QAPair:
    """A single question-answer pair."""

    question: str
    answer: str
    category: str
    source_title: str
    difficulty: str = "medium"
    reasoning_type: str = "single-hop"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "source_title": self.source_title,
            "difficulty": self.difficulty,
            "reasoning_type": self.reasoning_type,
        }

    @classmethod
    def from_dict(cls, data: dict, source_title: str = "") -> "QAPair":
        return cls(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            category=data.get("category", ""),
            source_title=data.get("source_title", source_title),
            difficulty=data.get("difficulty", "medium"),
            reasoning_type=data.get("reasoning_type", "single-hop"),
        )


@dataclass
class GenerationResult:
    """Result of QA generation for a single paper."""

    paper_id: str
    paper_title: str
    qa_pairs: list[QAPair]
    token_usage: dict

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "token_usage": self.token_usage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationResult":
        paper_title = data.get("paper_title", "")
        return cls(
            paper_id=data.get("paper_id", ""),
            paper_title=paper_title,
            qa_pairs=[
                QAPair.from_dict(qa, source_title=paper_title)
                for qa in data.get("qa_pairs", [])
            ],
            token_usage=data.get("token_usage", {}),
        )

    def save(self, output_dir: Path) -> Path:
        """Save generation result to JSON file."""
        output_path = output_dir / f"{self.paper_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return output_path

    @classmethod
    def load(cls, path: Path) -> "GenerationResult":
        """Load generation result from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class QAGenerator:
    """Generate QA pairs from extracted knowledge points."""

    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
        self.valid_categories = set(config.categories)
        self.valid_difficulties = {"easy", "medium", "hard"}
        self.valid_reasoning_types = {"single-hop", "multi-hop", "cross-doc"}

    def _validate_qa_pairs(
        self, qa_pairs: list[dict], source_title: str
    ) -> list[QAPair]:
        """Validate and filter QA pairs."""
        validated = []
        for qa in qa_pairs:
            # Check required fields
            if not qa.get("question") or not qa.get("answer"):
                continue

            # Validate category
            category = qa.get("category", "")
            if category not in self.valid_categories:
                # Try to find closest match
                for valid_cat in self.valid_categories:
                    if (
                        valid_cat.lower() in category.lower()
                        or category.lower() in valid_cat.lower()
                    ):
                        category = valid_cat
                        break
                else:
                    # Use first category as fallback
                    category = self.config.categories[0]

            # Validate difficulty
            difficulty = qa.get("difficulty", "medium")
            if difficulty not in self.valid_difficulties:
                difficulty = "medium"

            # Validate reasoning type
            reasoning_type = qa.get("reasoning_type", "single-hop")
            if reasoning_type not in self.valid_reasoning_types:
                reasoning_type = "single-hop"

            validated.append(
                QAPair(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    category=category,
                    source_title=source_title,
                    difficulty=difficulty,
                    reasoning_type=reasoning_type,
                )
            )

        return validated

    def generate_from_extraction(
        self, extraction_result: ExtractionResult
    ) -> GenerationResult:
        """Generate QA pairs from an extraction result."""
        if not extraction_result.knowledge_points:
            return GenerationResult(
                paper_id=extraction_result.paper_id,
                paper_title=extraction_result.paper_title,
                qa_pairs=[],
                token_usage={"error": "No knowledge points available"},
            )

        # Prepare knowledge points for prompt
        knowledge_points = [kp.to_dict() for kp in extraction_result.knowledge_points]

        # Generate prompt
        messages = format_generation_prompt(
            paper_title=extraction_result.paper_title,
            knowledge_points=knowledge_points,
        )

        # Call LLM
        try:
            parsed_response, response = self.llm_client.chat_json(messages)
        except Exception as e:
            return GenerationResult(
                paper_id=extraction_result.paper_id,
                paper_title=extraction_result.paper_title,
                qa_pairs=[],
                token_usage={"error": str(e)},
            )

        # Extract and validate QA pairs
        raw_qa_pairs = parsed_response.get("qa_pairs", [])
        qa_pairs = self._validate_qa_pairs(
            raw_qa_pairs, extraction_result.paper_title
        )

        # Ensure we have the target number of QA pairs
        min_qa = self.config.qa_settings.min_qa_per_paper
        max_qa = self.config.qa_settings.max_qa_per_paper

        if len(qa_pairs) > max_qa:
            qa_pairs = qa_pairs[:max_qa]

        return GenerationResult(
            paper_id=extraction_result.paper_id,
            paper_title=extraction_result.paper_title,
            qa_pairs=qa_pairs,
            token_usage=response.usage.to_dict(),
        )

    def generate_cross_doc_qa(
        self,
        extraction_results: list[ExtractionResult],
        sample_size: Optional[int] = None,
    ) -> GenerationResult:
        """Generate cross-document QA pairs from multiple extraction results."""
        if sample_size is None:
            sample_size = self.config.qa_settings.cross_doc_sample_size

        # Sample papers if we have too many
        if len(extraction_results) > sample_size:
            extraction_results = random.sample(extraction_results, sample_size)

        # Group knowledge points by paper
        knowledge_by_paper = {}
        for result in extraction_results:
            if result.knowledge_points:
                knowledge_by_paper[result.paper_title] = [
                    kp.to_dict() for kp in result.knowledge_points[:50]  # Limit per paper
                ]

        if len(knowledge_by_paper) < 2:
            return GenerationResult(
                paper_id="cross_doc",
                paper_title="Cross-Document QA",
                qa_pairs=[],
                token_usage={"error": "Not enough papers for cross-document QA"},
            )

        # Generate prompt
        messages = format_cross_doc_prompt(knowledge_by_paper)

        # Call LLM
        try:
            parsed_response, response = self.llm_client.chat_json(messages)
        except Exception as e:
            return GenerationResult(
                paper_id="cross_doc",
                paper_title="Cross-Document QA",
                qa_pairs=[],
                token_usage={"error": str(e)},
            )

        # Extract and validate QA pairs
        raw_qa_pairs = parsed_response.get("qa_pairs", [])

        validated = []
        for qa in raw_qa_pairs:
            if not qa.get("question") or not qa.get("answer"):
                continue

            # For cross-doc, source_title includes multiple papers
            source_papers = qa.get("source_papers", [])
            source_title = " | ".join(source_papers) if source_papers else "Multiple Papers"

            validated.append(
                QAPair(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    category=qa.get("category", self.config.categories[0]),
                    source_title=source_title,
                    difficulty="hard",
                    reasoning_type="cross-doc",
                )
            )

        return GenerationResult(
            paper_id="cross_doc",
            paper_title="Cross-Document QA",
            qa_pairs=validated,
            token_usage=response.usage.to_dict(),
        )

    def generate_from_directory(
        self,
        knowledge_dir: Path,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> list[GenerationResult]:
        """Generate QA pairs from all extraction results in a directory."""
        # Find all extraction result files
        json_files = list(knowledge_dir.glob("*.json"))

        results = []
        for i, file_path in enumerate(json_files):
            # Load extraction result
            extraction_result = ExtractionResult.load(file_path)

            # Check if already processed
            output_path = output_dir / f"{extraction_result.paper_id}.json"

            if output_path.exists():
                # Load existing result
                result = GenerationResult.load(output_path)
            else:
                # Generate and save
                result = self.generate_from_extraction(extraction_result)
                result.save(output_dir)

            results.append(result)

            # Progress callback
            if progress_callback:
                progress_callback(
                    current=i + 1,
                    total=len(json_files),
                    paper_title=result.paper_title,
                    qa_count=len(result.qa_pairs),
                )

        return results
