"""Main pipeline orchestration."""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from .checkpoint import CheckpointManager
from .config import Config
from .llm_client import LLMClient
from .monitor import ProgressMonitor
from .stage1_extractor import ExtractionResult, KnowledgeExtractor
from .stage2_generator import GenerationResult, QAGenerator


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging with Rich handler."""
    logger = logging.getLogger("qa_extractor")
    logger.setLevel(logging.INFO)

    # Rich console handler
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


class Pipeline:
    """Main pipeline for QA extraction."""

    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.logger = setup_logging(config.monitoring.log_file)

        # Ensure output directories exist
        config.ensure_directories()

        # Initialize components
        self.llm_client = LLMClient(config.llm)
        self.extractor = KnowledgeExtractor(config, self.llm_client)
        self.generator = QAGenerator(config, self.llm_client)
        self.monitor = ProgressMonitor(config)

        # Paths
        self.input_dir = Path(config.pipeline.input_dir)
        self.output_dir = Path(config.pipeline.output_dir)
        self.knowledge_dir = self.output_dir / "knowledge"
        self.qa_dir = self.output_dir / "qa_pairs"

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.output_dir)

    def _get_md_files(self) -> list[Path]:
        """Get all markdown files from input directory."""
        return sorted(self.input_dir.rglob("*.md"))

    def run_stage1(
        self,
        progress: Optional[Progress] = None,
        resume: bool = True,
    ) -> list[ExtractionResult]:
        """Run Stage 1: Knowledge extraction."""
        self.logger.info("Starting Stage 1: Knowledge Extraction")

        md_files = self._get_md_files()
        total_files = len(md_files)

        if total_files == 0:
            self.logger.warning(f"No markdown files found in {self.input_dir}")
            return []

        # Check for resume
        checkpoint = self.checkpoint_manager.load() if resume else None
        processed_set = set(checkpoint.processed_files) if checkpoint else set()

        results = []
        task_id = None

        if progress:
            task_id = progress.add_task(
                "[cyan]Stage 1: Extracting Knowledge",
                total=total_files,
            )

        for i, file_path in enumerate(md_files):
            file_key = str(file_path)

            # Check if result file exists - always verify the output file is present
            paper_id = self.extractor._generate_paper_id(file_path)
            result_path = self.knowledge_dir / f"{paper_id}.json"

            # Skip if already processed AND result file exists AND no errors
            if file_key in processed_set and result_path.exists():
                # Load existing result
                result = ExtractionResult.load(result_path)

                # Check if the result has an error - if so, re-process
                if "error" in result.token_usage or len(result.knowledge_points) == 0:
                    self.logger.info(f"Re-processing failed file: {file_path.name}")
                else:
                    results.append(result)
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
                    continue

            # Extract knowledge
            self.logger.info(f"Processing: {file_path.name}")
            result = self.extractor.extract_from_file(file_path)
            result.save(self.knowledge_dir)
            results.append(result)

            # Update checkpoint
            category_counts = Counter(kp.category for kp in result.knowledge_points)
            self.checkpoint_manager.update(
                stage="extract",
                processed_file=file_key,
                token_stats=self.llm_client.get_stats().to_dict(),
                knowledge_count=self.checkpoint_manager.get_current().knowledge_count
                + len(result.knowledge_points),
            )

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            self.logger.info(
                f"  Extracted {len(result.knowledge_points)} knowledge points"
            )

        self.logger.info(f"Stage 1 complete: {len(results)} papers processed")
        return results

    def run_stage2(
        self,
        extraction_results: Optional[list[ExtractionResult]] = None,
        progress: Optional[Progress] = None,
        resume: bool = True,
    ) -> list[GenerationResult]:
        """Run Stage 2: QA generation."""
        self.logger.info("Starting Stage 2: QA Generation")

        # Load extraction results if not provided
        if extraction_results is None:
            extraction_results = []
            for json_file in self.knowledge_dir.glob("*.json"):
                extraction_results.append(ExtractionResult.load(json_file))

        if not extraction_results:
            self.logger.warning("No extraction results found")
            return []

        # Check for resume
        checkpoint = self.checkpoint_manager.load() if resume else None
        processed_set = set()
        if checkpoint and checkpoint.stage in ["generate", "cross_doc", "complete"]:
            # Get already generated paper IDs
            for qa_file in self.qa_dir.glob("*.json"):
                if qa_file.stem != "cross_doc":
                    processed_set.add(qa_file.stem)

        results = []
        task_id = None

        if progress:
            task_id = progress.add_task(
                "[green]Stage 2: Generating QA Pairs",
                total=len(extraction_results),
            )

        for i, extraction_result in enumerate(extraction_results):
            # Skip if already processed
            if extraction_result.paper_id in processed_set:
                result_path = self.qa_dir / f"{extraction_result.paper_id}.json"
                if result_path.exists():
                    result = GenerationResult.load(result_path)
                    results.append(result)

                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
                continue

            # Generate QA pairs
            self.logger.info(f"Generating QA for: {extraction_result.paper_title[:50]}...")
            result = self.generator.generate_from_extraction(extraction_result)
            result.save(self.qa_dir)
            results.append(result)

            # Update checkpoint
            self.checkpoint_manager.update(
                stage="generate",
                token_stats=self.llm_client.get_stats().to_dict(),
                qa_count=self.checkpoint_manager.get_current().qa_count
                + len(result.qa_pairs),
            )

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            self.logger.info(f"  Generated {len(result.qa_pairs)} QA pairs")

        self.logger.info(f"Stage 2 complete: {len(results)} papers processed")
        return results

    def run_cross_doc(
        self,
        extraction_results: Optional[list[ExtractionResult]] = None,
        progress: Optional[Progress] = None,
    ) -> Optional[GenerationResult]:
        """Run cross-document QA generation."""
        if not self.config.qa_settings.enable_cross_doc:
            self.logger.info("Cross-document QA generation disabled")
            return None

        self.logger.info("Starting Cross-Document QA Generation")

        # Load extraction results if not provided
        if extraction_results is None:
            extraction_results = []
            for json_file in self.knowledge_dir.glob("*.json"):
                extraction_results.append(ExtractionResult.load(json_file))

        if len(extraction_results) < 2:
            self.logger.warning("Need at least 2 papers for cross-document QA")
            return None

        # Check if already done
        cross_doc_path = self.qa_dir / "cross_doc.json"
        if cross_doc_path.exists():
            self.logger.info("Cross-document QA already generated")
            return GenerationResult.load(cross_doc_path)

        task_id = None
        if progress:
            task_id = progress.add_task(
                "[magenta]Cross-Document QA",
                total=1,
            )

        result = self.generator.generate_cross_doc_qa(extraction_results)
        result.save(self.qa_dir)

        # Update checkpoint
        self.checkpoint_manager.update(
            stage="cross_doc",
            token_stats=self.llm_client.get_stats().to_dict(),
        )

        if progress and task_id is not None:
            progress.update(task_id, advance=1)

        self.logger.info(f"Generated {len(result.qa_pairs)} cross-document QA pairs")
        return result

    def run(self, resume: bool = True) -> dict:
        """Run the full pipeline with Rich progress display."""
        self.console.print()
        self.console.rule("[bold blue]PhotoRAG QA Extractor")
        self.console.print()

        # Display configuration
        self.console.print(f"[cyan]Model:[/cyan] {self.config.llm.model}")
        self.console.print(f"[cyan]Input:[/cyan] {self.input_dir}")
        self.console.print(f"[cyan]Output:[/cyan] {self.output_dir}")
        self.console.print()

        md_files = self._get_md_files()
        self.console.print(f"Found [bold]{len(md_files)}[/bold] markdown files")
        self.console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            # Stage 1: Knowledge Extraction
            extraction_results = self.run_stage1(progress=progress, resume=resume)

            # Stage 2: QA Generation
            generation_results = self.run_stage2(
                extraction_results=extraction_results,
                progress=progress,
                resume=resume,
            )

            # Cross-document QA
            cross_doc_result = self.run_cross_doc(
                extraction_results=extraction_results,
                progress=progress,
            )

        # Mark complete
        self.checkpoint_manager.update(stage="complete")

        # Calculate statistics
        total_knowledge = sum(len(r.knowledge_points) for r in extraction_results)
        total_qa = sum(len(r.qa_pairs) for r in generation_results)
        if cross_doc_result:
            total_qa += len(cross_doc_result.qa_pairs)

        token_stats = self.llm_client.get_stats()

        # Print summary
        self.console.print()
        self.console.rule("[bold green]Complete")
        self.console.print()
        self.console.print(f"[green]✓[/green] Papers processed: {len(extraction_results)}")
        self.console.print(f"[green]✓[/green] Knowledge points: {total_knowledge}")
        self.console.print(f"[green]✓[/green] QA pairs generated: {total_qa}")
        self.console.print(f"[green]✓[/green] Total tokens: {token_stats.total_usage.total_tokens:,}")
        self.console.print(f"[green]✓[/green] Estimated cost: ${token_stats.estimate_cost():.2f}")
        self.console.print()

        return {
            "papers_processed": len(extraction_results),
            "knowledge_points": total_knowledge,
            "qa_pairs": total_qa,
            "token_usage": token_stats.to_dict(),
        }

    def close(self) -> None:
        """Clean up resources."""
        self.llm_client.close()
