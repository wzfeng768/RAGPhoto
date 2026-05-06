"""Checkpoint management for pipeline resume functionality."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Checkpoint:
    """Pipeline checkpoint for resume functionality."""

    stage: str = "extract"  # "extract", "generate", "cross_doc", "complete"
    processed_files: list[str] = field(default_factory=list)
    last_file: str = ""
    token_stats: dict = field(default_factory=dict)
    knowledge_count: int = 0
    qa_count: int = 0
    timestamp: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "processed_files": self.processed_files,
            "last_file": self.last_file,
            "token_stats": self.token_stats,
            "knowledge_count": self.knowledge_count,
            "qa_count": self.qa_count,
            "timestamp": self.timestamp,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(
            stage=data.get("stage", "extract"),
            processed_files=data.get("processed_files", []),
            last_file=data.get("last_file", ""),
            token_stats=data.get("token_stats", {}),
            knowledge_count=data.get("knowledge_count", 0),
            qa_count=data.get("qa_count", 0),
            timestamp=data.get("timestamp", ""),
            errors=data.get("errors", []),
        )


class CheckpointManager:
    """Manage pipeline checkpoints for resume functionality."""

    CHECKPOINT_FILE = ".checkpoint.json"

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        self._checkpoint: Optional[Checkpoint] = None

    def exists(self) -> bool:
        """Check if a checkpoint exists."""
        return self.checkpoint_path.exists()

    def load(self) -> Optional[Checkpoint]:
        """Load checkpoint from file."""
        if not self.exists():
            return None

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._checkpoint = Checkpoint.from_dict(data)
            return self._checkpoint
        except (json.JSONDecodeError, IOError):
            return None

    def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to file."""
        checkpoint.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._checkpoint = checkpoint

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

    def update(
        self,
        stage: Optional[str] = None,
        processed_file: Optional[str] = None,
        token_stats: Optional[dict] = None,
        knowledge_count: Optional[int] = None,
        qa_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update checkpoint with new information."""
        if self._checkpoint is None:
            self._checkpoint = Checkpoint()

        if stage is not None:
            self._checkpoint.stage = stage

        if processed_file is not None:
            if processed_file not in self._checkpoint.processed_files:
                self._checkpoint.processed_files.append(processed_file)
            self._checkpoint.last_file = processed_file

        if token_stats is not None:
            self._checkpoint.token_stats = token_stats

        if knowledge_count is not None:
            self._checkpoint.knowledge_count = knowledge_count

        if qa_count is not None:
            self._checkpoint.qa_count = qa_count

        if error is not None:
            self._checkpoint.errors.append(error)

        self.save(self._checkpoint)

    def get_unprocessed_files(self, all_files: list[str]) -> list[str]:
        """Get list of files that haven't been processed yet."""
        if self._checkpoint is None:
            return all_files

        processed_set = set(self._checkpoint.processed_files)
        return [f for f in all_files if f not in processed_set]

    def clear(self) -> None:
        """Clear the checkpoint."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self._checkpoint = None

    def get_current(self) -> Checkpoint:
        """Get current checkpoint or create new one."""
        if self._checkpoint is None:
            self._checkpoint = self.load() or Checkpoint()
        return self._checkpoint
