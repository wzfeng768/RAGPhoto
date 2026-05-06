"""Configuration management for QA Extractor."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM API configuration."""

    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="")
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: int = Field(default=120, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    retry_delay: int = Field(default=5, ge=0)


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    input_dir: str = Field(default="./MDs")
    output_dir: str = Field(default="./output")
    batch_size: int = Field(default=10, gt=0)
    checkpoint_interval: int = Field(default=5, gt=0)


class QASettings(BaseModel):
    """QA generation settings."""

    min_qa_per_paper: int = Field(default=8, gt=0)
    max_qa_per_paper: int = Field(default=12, gt=0)
    enable_cross_doc: bool = Field(default=True)
    cross_doc_sample_size: int = Field(default=50, gt=0)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    show_live_log: bool = Field(default=True)
    log_file: str = Field(default="./output/qa_extractor.log")
    save_token_stats: bool = Field(default=True)


# Default categories
DEFAULT_CATEGORIES = [
    "Materials Design & Synthesis",
    "Performance Metrics",
    "Structure-Property Relationships",
    "Device Architecture & Physics",
    "Processing & Fabrication",
    "Characterization Methods",
    "Stability & Degradation",
    "Computational & Machine Learning",
]


class Config(BaseModel):
    """Main configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    qa_settings: QASettings = Field(default_factory=QASettings)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    categories: list[str] = Field(default_factory=lambda: DEFAULT_CATEGORIES.copy())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Expand environment variables in api_key
        if "llm" in data and "api_key" in data["llm"]:
            api_key = data["llm"]["api_key"]
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                data["llm"]["api_key"] = os.environ.get(env_var, "")

        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        config = cls()

        # Override from environment
        if base_url := os.environ.get("QA_EXTRACTOR_BASE_URL"):
            config.llm.base_url = base_url
        if api_key := os.environ.get("QA_EXTRACTOR_API_KEY"):
            config.llm.api_key = api_key
        if model := os.environ.get("QA_EXTRACTOR_MODEL"):
            config.llm.model = model
        if input_dir := os.environ.get("QA_EXTRACTOR_INPUT_DIR"):
            config.pipeline.input_dir = input_dir
        if output_dir := os.environ.get("QA_EXTRACTOR_OUTPUT_DIR"):
            config.pipeline.output_dir = output_dir

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def ensure_directories(self) -> None:
        """Create necessary output directories."""
        output_dir = Path(self.pipeline.output_dir)
        (output_dir / "knowledge").mkdir(parents=True, exist_ok=True)
        (output_dir / "qa_pairs").mkdir(parents=True, exist_ok=True)
        (output_dir / "final" / "qa_by_category").mkdir(parents=True, exist_ok=True)
        (output_dir / "stats").mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment."""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)

    # Try default locations
    default_paths = ["config.yaml", "config.yml", ".qa_extractor.yaml"]
    for path in default_paths:
        if Path(path).exists():
            return Config.from_yaml(path)

    # Fall back to environment variables
    return Config.from_env()
