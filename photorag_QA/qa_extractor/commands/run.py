"""Enhanced run command with live dashboard."""

import time
from pathlib import Path
from collections import Counter
from typing import Optional

from rich.console import Console

from ..config import Config
from ..llm_client import LLMClient
from ..checkpoint import CheckpointManager
from ..stage1_extractor import KnowledgeExtractor, ExtractionResult
from ..stage2_generator import QAGenerator, GenerationResult
from ..ui.dashboard import Dashboard, print_results_summary
from ..ui.themes import get_theme


def run_command(
    config: Config,
    console: Console,
    resume: bool = True,
) -> dict:
    """Run the full pipeline with live dashboard."""
    theme = get_theme()
    start_time = time.time()

    # Setup paths
    input_dir = Path(config.pipeline.input_dir)
    output_dir = Path(config.pipeline.output_dir)
    knowledge_dir = output_dir / "knowledge"
    qa_dir = output_dir / "qa_pairs"

    # Ensure directories exist
    config.ensure_directories()

    # Get file list
    md_files = sorted(input_dir.rglob("*.md"))

    # Initialize components
    llm_client = LLMClient(config.llm)
    extractor = KnowledgeExtractor(config, llm_client)
    generator = QAGenerator(config, llm_client)
    checkpoint_manager = CheckpointManager(output_dir)

    # Load checkpoint
    checkpoint = checkpoint_manager.load() if resume else None
    processed_files = set(checkpoint.processed_files) if checkpoint else set()

    # Dashboard config
    dashboard_config = {
        "model": config.llm.model,
        "input": str(input_dir),
        "output": str(output_dir),
        "resume": resume,
        "file_count": len(md_files),
    }

    extraction_results = []
    generation_results = []
    cross_doc_result = None

    try:
        with Dashboard(console, theme, dashboard_config) as dashboard:
            # Set totals
            dashboard.set_stage_total("extract", len(md_files))
            dashboard.set_stage_total("generate", len(md_files))

            # === Stage 1: Knowledge Extraction ===
            dashboard.log("Starting Stage 1: Knowledge Extraction", "info")
            dashboard.update_progress("extract", 0, len(md_files), "in_progress")

            for i, file_path in enumerate(md_files):
                file_key = str(file_path)
                paper_id = extractor._generate_paper_id(file_path)
                result_path = knowledge_dir / f"{paper_id}.json"

                # Check if already processed successfully
                if file_key in processed_files and result_path.exists():
                    result = ExtractionResult.load(result_path)
                    if "error" not in result.token_usage and len(result.knowledge_points) > 0:
                        extraction_results.append(result)
                        dashboard.update_progress("extract", i + 1, len(md_files), "in_progress")
                        continue

                # Process file
                dashboard.update_task(
                    filename=file_path.name,
                    title="",
                    status="calling_api",
                    status_message="Calling LLM API...",
                )

                result = extractor.extract_from_file(file_path)
                result.save(knowledge_dir)
                extraction_results.append(result)

                # Update checkpoint
                checkpoint_manager.update(
                    stage="extract",
                    processed_file=file_key,
                    token_stats=llm_client.get_stats().to_dict(),
                    knowledge_count=sum(len(r.knowledge_points) for r in extraction_results),
                )

                # Update dashboard
                stats = llm_client.get_stats()
                dashboard.update_tokens(
                    prompt_tokens=stats.total_usage.prompt_tokens,
                    completion_tokens=stats.total_usage.completion_tokens,
                    total_tokens=stats.total_usage.total_tokens,
                    estimated_cost=stats.estimate_cost(),
                    request_count=stats.request_count,
                )

                if "error" in result.token_usage:
                    dashboard.log(f"Error extracting: {file_path.name}", "error")
                else:
                    dashboard.log(f"Extracted {len(result.knowledge_points)} points: {file_path.name[:30]}", "success")

                dashboard.update_progress("extract", i + 1, len(md_files), "in_progress")

            dashboard.update_progress("extract", len(md_files), len(md_files), "complete")
            dashboard.log(f"Stage 1 complete: {len(extraction_results)} papers", "success")

            # === Stage 2: QA Generation ===
            dashboard.log("Starting Stage 2: QA Generation", "info")
            dashboard.update_progress("generate", 0, len(extraction_results), "in_progress")

            # Get already generated papers
            generated_set = set()
            if checkpoint and checkpoint.stage in ["generate", "cross_doc", "complete"]:
                for qa_file in qa_dir.glob("*.json"):
                    if qa_file.stem != "cross_doc":
                        generated_set.add(qa_file.stem)

            for i, extraction_result in enumerate(extraction_results):
                # Skip if already generated
                result_path = qa_dir / f"{extraction_result.paper_id}.json"
                if extraction_result.paper_id in generated_set and result_path.exists():
                    try:
                        result = GenerationResult.load(result_path)
                        if "error" not in result.token_usage and len(result.qa_pairs) > 0:
                            generation_results.append(result)
                            dashboard.update_progress("generate", i + 1, len(extraction_results), "in_progress")
                            continue
                    except Exception:
                        pass

                # Skip if no knowledge points
                if not extraction_result.knowledge_points:
                    generation_results.append(GenerationResult(
                        paper_id=extraction_result.paper_id,
                        paper_title=extraction_result.paper_title,
                        qa_pairs=[],
                        token_usage={"skipped": "no knowledge points"},
                    ))
                    dashboard.update_progress("generate", i + 1, len(extraction_results), "in_progress")
                    continue

                # Generate QA pairs
                dashboard.update_task(
                    filename=extraction_result.paper_title[:50],
                    title="",
                    status="calling_api",
                    status_message="Generating QA pairs...",
                )

                result = generator.generate_from_extraction(extraction_result)
                result.save(qa_dir)
                generation_results.append(result)

                # Update checkpoint
                checkpoint_manager.update(
                    stage="generate",
                    token_stats=llm_client.get_stats().to_dict(),
                    qa_count=sum(len(r.qa_pairs) for r in generation_results),
                )

                # Update dashboard
                stats = llm_client.get_stats()
                dashboard.update_tokens(
                    prompt_tokens=stats.total_usage.prompt_tokens,
                    completion_tokens=stats.total_usage.completion_tokens,
                    total_tokens=stats.total_usage.total_tokens,
                    estimated_cost=stats.estimate_cost(),
                    request_count=stats.request_count,
                )

                if "error" in result.token_usage:
                    dashboard.log(f"Error generating QA: {extraction_result.paper_id[:30]}", "error")
                else:
                    dashboard.log(f"Generated {len(result.qa_pairs)} QA pairs", "success")

                dashboard.update_progress("generate", i + 1, len(extraction_results), "in_progress")

            dashboard.update_progress("generate", len(extraction_results), len(extraction_results), "complete")
            dashboard.log(f"Stage 2 complete: {sum(len(r.qa_pairs) for r in generation_results)} QA pairs", "success")

            # === Stage 3: Cross-Document QA ===
            if config.qa_settings.enable_cross_doc and len(extraction_results) >= 2:
                dashboard.log("Starting Cross-Document QA Generation", "info")
                dashboard.update_progress("cross_doc", 0, 1, "in_progress")

                cross_doc_path = qa_dir / "cross_doc.json"
                if cross_doc_path.exists():
                    cross_doc_result = GenerationResult.load(cross_doc_path)
                    dashboard.log("Cross-doc QA already exists, skipping", "info")
                else:
                    dashboard.update_task(
                        filename="Cross-Document",
                        title="Generating cross-document QA pairs",
                        status="calling_api",
                        status_message="Analyzing multiple papers...",
                    )

                    cross_doc_result = generator.generate_cross_doc_qa(extraction_results)
                    cross_doc_result.save(qa_dir)

                    dashboard.log(f"Generated {len(cross_doc_result.qa_pairs)} cross-doc QA pairs", "success")

                dashboard.update_progress("cross_doc", 1, 1, "complete")
            else:
                dashboard.update_progress("cross_doc", 0, 1, "complete")

            # Clear task panel
            dashboard.update_task(status="success", status_message="Complete!")

            # Small delay to show final state
            time.sleep(0.5)

    finally:
        llm_client.close()

    # Mark complete
    checkpoint_manager.update(stage="complete")

    # Calculate final stats
    duration = time.time() - start_time
    total_knowledge = sum(len(r.knowledge_points) for r in extraction_results)
    total_qa = sum(len(r.qa_pairs) for r in generation_results)
    cross_doc_qa = len(cross_doc_result.qa_pairs) if cross_doc_result else 0

    stats = llm_client.get_stats()

    # Category distribution
    category_counts = Counter()
    for result in generation_results:
        for qa in result.qa_pairs:
            category_counts[qa.category] += 1
    if cross_doc_result:
        for qa in cross_doc_result.qa_pairs:
            category_counts[qa.category] += 1

    # Print results summary
    print_results_summary(
        console=console,
        papers=len(extraction_results),
        knowledge=total_knowledge,
        qa_pairs=total_qa,
        cross_doc=cross_doc_qa,
        tokens=stats.total_usage.total_tokens,
        cost=stats.estimate_cost(),
        duration=duration,
        category_data=dict(category_counts) if category_counts else None,
        theme=theme,
    )

    return {
        "papers_processed": len(extraction_results),
        "knowledge_points": total_knowledge,
        "qa_pairs": total_qa + cross_doc_qa,
        "token_usage": stats.to_dict(),
        "duration": duration,
    }
