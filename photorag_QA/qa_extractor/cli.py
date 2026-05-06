"""Rich CLI interface for QA Extractor."""

import json
from pathlib import Path

import click
from rich.console import Console

from .config import Config, load_config
from .exporter import QAExporter
from .ui.banner import print_banner, print_error_banner
from .ui.themes import get_theme, Icons
from .commands.run import run_command
from .commands.stats import stats_command
from .commands.status import status_command
from .commands.validate import validate_command


console = Console()
theme = get_theme()


@click.group()
@click.version_option(version="0.2.0", prog_name="qa-extractor")
def cli():
    """PhotoRAG QA Extractor - Extract QA pairs from academic literature.

    \b
    Commands:
      run       Run the full QA extraction pipeline
      extract   Run Stage 1: Extract knowledge points
      generate  Run Stage 2: Generate QA pairs
      stats     Show detailed statistics
      status    Show pipeline status and checkpoint
      validate  Check output quality and errors
      export    Export QA pairs to file
      init      Generate sample configuration
      clear     Clear checkpoint to start fresh
    """
    pass


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    help="Input directory containing markdown files",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--no-resume",
    is_flag=True,
    help="Start fresh, ignoring any existing checkpoint",
)
def run(config, input, output, no_resume):
    """Run the full QA extraction pipeline with live dashboard."""
    try:
        # Load configuration
        cfg = load_config(config)

        # Override with CLI options
        if input:
            cfg.pipeline.input_dir = input
        if output:
            cfg.pipeline.output_dir = output

        # Validate API key
        if not cfg.llm.api_key:
            print_error_banner(
                console,
                "API key not set. Please set api_key in config.yaml file.",
            )
            raise click.Abort()

        # Run pipeline with new dashboard
        results = run_command(cfg, console, resume=not no_resume)

    except FileNotFoundError as e:
        print_error_banner(console, str(e))
        raise click.Abort()
    except KeyboardInterrupt:
        console.print(f"\n[{theme.warning}]{Icons.WARNING} Interrupted by user[/]")
        raise click.Abort()
    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input directory containing markdown files",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    required=True,
    help="Output directory for knowledge points",
)
def extract(config, input, output):
    """Run Stage 1: Extract knowledge points from papers."""
    try:
        from .pipeline import Pipeline
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

        cfg = load_config(config)
        cfg.pipeline.input_dir = input
        cfg.pipeline.output_dir = output

        if not cfg.llm.api_key:
            print_error_banner(console, "API key not set in config file.")
            raise click.Abort()

        print_banner(console, theme, show_config=True, config_info={
            "model": cfg.llm.model,
            "input": input,
            "output": output,
        })

        pipeline = Pipeline(cfg)
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                results = pipeline.run_stage1(progress=progress, resume=True)

            console.print(f"\n[{theme.success}]{Icons.SUCCESS}[/] Extracted knowledge from {len(results)} papers")

        finally:
            pipeline.close()

    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input directory containing knowledge point files",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    required=True,
    help="Output directory for QA pairs",
)
def generate(config, input, output):
    """Run Stage 2: Generate QA pairs from knowledge points."""
    try:
        from .pipeline import Pipeline
        from .stage1_extractor import ExtractionResult
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

        cfg = load_config(config)
        cfg.pipeline.output_dir = output

        if not cfg.llm.api_key:
            print_error_banner(console, "API key not set in config file.")
            raise click.Abort()

        # Load extraction results
        knowledge_dir = Path(input)
        extraction_results = []
        for json_file in knowledge_dir.glob("*.json"):
            extraction_results.append(ExtractionResult.load(json_file))

        if not extraction_results:
            print_error_banner(console, f"No knowledge files found in {input}")
            raise click.Abort()

        print_banner(console, theme)
        console.print(f"Found [{theme.primary}]{len(extraction_results)}[/] knowledge files\n")

        pipeline = Pipeline(cfg)
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                results = pipeline.run_stage2(
                    extraction_results=extraction_results,
                    progress=progress,
                    resume=True,
                )

            total_qa = sum(len(r.qa_pairs) for r in results)
            console.print(f"\n[{theme.success}]{Icons.SUCCESS}[/] Generated {total_qa} QA pairs from {len(results)} papers")

        finally:
            pipeline.close()

    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(exists=True),
    required=True,
    help="Output directory containing QA results",
)
@click.option(
    "--detailed", "-d",
    is_flag=True,
    help="Show detailed per-paper breakdown",
)
def stats(output, detailed):
    """Show detailed statistics about generated QA pairs."""
    try:
        stats_command(output, console, detailed=detailed)
    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(exists=True),
    required=True,
    help="Output directory to check status for",
)
def status(output):
    """Show current pipeline status and checkpoint information."""
    try:
        status_command(output, console)
    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(exists=True),
    required=True,
    help="Output directory to validate",
)
@click.option(
    "--fix",
    is_flag=True,
    help="Remove error files for re-processing",
)
def validate(output, fix):
    """Check output quality and report errors."""
    try:
        validate_command(output, console, fix=fix)
    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


@cli.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input directory containing QA results",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    required=True,
    help="Output file path (JSON)",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="Output format",
)
@click.option(
    "--by-category",
    is_flag=True,
    help="Also export separate files by category",
)
def export(input, output, format, by_category):
    """Export QA pairs to a single file."""
    try:
        cfg = Config()
        exporter = QAExporter(cfg)

        input_dir = Path(input)
        qa_dir = input_dir / "qa_pairs" if (input_dir / "qa_pairs").exists() else input_dir

        result = exporter.export_all(
            qa_dir=qa_dir,
            output_path=Path(output),
            format=format,
            split_by_category=by_category,
        )

        console.print(f"\n[{theme.success}]{Icons.SUCCESS}[/] Exported {result['total_qa_pairs']} QA pairs to {output}")

        if by_category:
            console.print(f"[{theme.success}]{Icons.SUCCESS}[/] Category files saved to {Path(output).parent / 'qa_by_category'}")

    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    required=True,
    help="Output directory to clear checkpoint from",
)
@click.confirmation_option(prompt="Are you sure you want to clear the checkpoint?")
def clear(output):
    """Clear the checkpoint to start fresh."""
    try:
        from .checkpoint import CheckpointManager

        checkpoint_manager = CheckpointManager(Path(output))
        checkpoint_manager.clear()

        console.print(f"[{theme.success}]{Icons.SUCCESS}[/] Checkpoint cleared")

    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="config.yaml",
    help="Output path for config file",
)
def init(output):
    """Generate a sample configuration file."""
    try:
        config = Config()

        # Set placeholder values
        config.llm.api_key = "sk-your-api-key-here"

        config.to_yaml(output)

        print_banner(console, theme)
        console.print(f"[{theme.success}]{Icons.SUCCESS}[/] Configuration file created: [{theme.primary}]{output}[/]")
        console.print()
        console.print("Next steps:")
        console.print(f"  1. Edit config and set your [{theme.primary}]api_key[/] and [{theme.primary}]base_url[/]")
        console.print("  2. Adjust other settings as needed")
        console.print(f"  3. Run: [{theme.primary}]python -m qa_extractor run -c {output}[/]")
        console.print()

    except Exception as e:
        print_error_banner(console, str(e))
        raise click.Abort()


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
