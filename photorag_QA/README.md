# PhotoRAG QA Extractor

A Python CLI tool for extracting high-quality question-answer pairs from optoelectronic polymer materials literature for RAG (Retrieval-Augmented Generation) evaluation.

## Features

- **Two-stage extraction pipeline**: Knowledge point extraction → QA pair generation
- **Multiple complexity levels**: Single-hop, multi-hop, and cross-document questions
- **8 knowledge categories**: Materials Design, Performance Metrics, Structure-Property Relationships, etc.
- **Beautiful TUI Dashboard**: Live progress panels, token tracking, activity logs with real-time updates
- **Pipeline Status & Validation**: Check progress, detect errors, auto-fix failed files
- **Checkpoint/Resume**: Automatically saves progress, supports interruption and continuation
- **Flexible LLM backend**: Supports any OpenAI-compatible API endpoint
- **Visual Statistics**: Bar charts for category/difficulty distributions
- **Export options**: JSON, JSONL, split by category

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Commands](#cli-commands)
- [TUI Dashboard](#tui-dashboard)
- [Pipeline Overview](#pipeline-overview)
- [Output Format](#output-format)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.10 or higher
- Access to an OpenAI-compatible LLM API

### Install Dependencies

```bash
cd QAExtractor
pip install -r requirements.txt
```

### Verify Installation

```bash
python -m qa_extractor --version
```

## Quick Start

### 1. Configure the Tool

Edit `config.yaml` to set your API key and endpoint:

```yaml
llm:
  base_url: "https://your-api-endpoint.com/v1"  # Your API endpoint
  api_key: "sk-your-api-key-here"               # Your API key
  model: "gpt-4o"                                # Model name

pipeline:
  input_dir: "../MDs"                            # Path to markdown files
  output_dir: "./output"                         # Output directory
```

### 2. Run the Pipeline

```bash
python -m qa_extractor run -c config.yaml
```

The tool will:
1. Scan all markdown files in the input directory
2. Extract knowledge points from each paper (Stage 1)
3. Generate QA pairs from knowledge points (Stage 2)
4. Optionally generate cross-document QA pairs
5. Save results to the output directory

## Configuration

### Configuration File Structure

```yaml
# LLM API Configuration
llm:
  base_url: "https://api.openai.com/v1"    # API endpoint URL
  api_key: "sk-your-api-key-here"          # Your API key
  model: "gpt-4o"                          # Model identifier
  temperature: 0.7                          # Generation temperature (0-2)
  max_tokens: 4096                          # Max tokens per request
  timeout: 120                              # Request timeout (seconds)
  retry_attempts: 3                         # Retry count on failure
  retry_delay: 5                            # Delay between retries (seconds)

# Pipeline Configuration
pipeline:
  input_dir: "../MDs"                       # Input markdown directory
  output_dir: "./output"                    # Output directory
  batch_size: 10                            # Papers per batch
  checkpoint_interval: 5                    # Save checkpoint every N papers

# QA Generation Settings
qa_settings:
  min_qa_per_paper: 8                       # Minimum QA pairs per paper
  max_qa_per_paper: 12                      # Maximum QA pairs per paper
  enable_cross_doc: true                    # Enable cross-document QA
  cross_doc_sample_size: 50                 # Papers sampled for cross-doc QA

# Knowledge Categories
categories:
  - "Materials Design & Synthesis"
  - "Performance Metrics"
  - "Structure-Property Relationships"
  - "Device Architecture & Physics"
  - "Processing & Fabrication"
  - "Characterization Methods"
  - "Stability & Degradation"
  - "Computational & Machine Learning"

# Monitoring Configuration
monitoring:
  show_live_log: true                       # Show live progress
  log_file: "./output/qa_extractor.log"     # Log file path
  save_token_stats: true                    # Save token usage statistics
```

### Environment Variables (Optional)

Environment variables can override config file settings:

| Variable | Description |
|----------|-------------|
| `QA_EXTRACTOR_API_KEY` | Override API key |
| `QA_EXTRACTOR_BASE_URL` | Override API base URL |
| `QA_EXTRACTOR_MODEL` | Override model name |
| `QA_EXTRACTOR_INPUT_DIR` | Override input directory |
| `QA_EXTRACTOR_OUTPUT_DIR` | Override output directory |

### Using Different LLM Providers

**OpenAI:**
```yaml
llm:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
```

**Azure OpenAI:**
```yaml
llm:
  base_url: "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
  model: "gpt-4o"
```

**Local Model (Ollama/vLLM):**
```yaml
llm:
  base_url: "http://localhost:11434/v1"  # Ollama
  model: "llama3.1"
```

**Other OpenAI-compatible APIs:**
```yaml
llm:
  base_url: "https://your-provider.com/v1"
  model: "your-model-name"
```

## CLI Commands

### Command Overview

| Command | Description |
|---------|-------------|
| `run` | Run the full QA extraction pipeline with live dashboard |
| `extract` | Run Stage 1: Extract knowledge points |
| `generate` | Run Stage 2: Generate QA pairs |
| `stats` | Show detailed statistics with visual charts |
| `status` | Show pipeline status and checkpoint info |
| `validate` | Check output quality and errors |
| `export` | Export QA pairs to file |
| `init` | Generate sample configuration |
| `clear` | Clear checkpoint to start fresh |

### `run` - Full Pipeline

Run the complete extraction pipeline:

```bash
python -m qa_extractor run [OPTIONS]

Options:
  -c, --config PATH    Path to configuration file
  -i, --input PATH     Input directory (overrides config)
  -o, --output PATH    Output directory (overrides config)
  --no-resume          Start fresh, ignore existing checkpoint
```

**Examples:**

```bash
# Run with config file
python -m qa_extractor run -c config.yaml

# Run with custom paths
python -m qa_extractor run -c config.yaml -i ./papers -o ./results

# Start fresh (ignore checkpoint)
python -m qa_extractor run -c config.yaml --no-resume
```

### `extract` - Stage 1 Only

Extract knowledge points from papers:

```bash
python -m qa_extractor extract [OPTIONS]

Options:
  -c, --config PATH    Path to configuration file
  -i, --input PATH     Input directory (required)
  -o, --output PATH    Output directory (required)
```

**Example:**

```bash
python -m qa_extractor extract -i ../MDs -o ./output
```

### `generate` - Stage 2 Only

Generate QA pairs from extracted knowledge points:

```bash
python -m qa_extractor generate [OPTIONS]

Options:
  -c, --config PATH    Path to configuration file
  -i, --input PATH     Knowledge points directory (required)
  -o, --output PATH    Output directory (required)
```

**Example:**

```bash
python -m qa_extractor generate -i ./output/knowledge -o ./output
```

### `stats` - View Statistics

Display statistics about generated QA pairs with visual bar charts:

```bash
python -m qa_extractor stats [OPTIONS]

Options:
  -o, --output PATH    Output directory containing results (required)
  -d, --detailed       Show detailed per-paper breakdown
```

**Examples:**

```bash
# Basic statistics with charts
python -m qa_extractor stats -o ./output

# Detailed per-paper breakdown
python -m qa_extractor stats -o ./output --detailed
```

**Sample Output:**

```
──────────────────── QA Extraction Statistics ────────────────────

┌─ Results Summary ────────────────────────────────────────────────┐
│   Papers Processed    5                                          │
│   Knowledge Points    47                                         │
│   QA Pairs Generated  38                                         │
│   Cross-Doc QAs       10                                         │
└──────────────────────────────────────────────────────────────────┘

┌─ Category Distribution ──────────────────────────────────────────┐
│   Materials Design       ████████████████░░░░  12  (25.0%)       │
│   Performance Metrics    ██████████████░░░░░░  10  (20.8%)       │
│   Structure-Property     ████████████░░░░░░░░   8  (16.7%)       │
│   Device Architecture    ████████░░░░░░░░░░░░   6  (12.5%)       │
│   ...                                                            │
└──────────────────────────────────────────────────────────────────┘
```

### `status` - Pipeline Status

Show current pipeline status and checkpoint information:

```bash
python -m qa_extractor status [OPTIONS]

Options:
  -o, --output PATH    Output directory to check status for (required)
```

**Example:**

```bash
python -m qa_extractor status -o ./output
```

**Sample Output:**

```
┌─ Pipeline Status ────────────────────────────────────────────────┐
│  Stage        Progress      Files     Results        Errors      │
│  ──────────   ───────────   ───────   ────────────   ─────────   │
│  Extract      ● Done        5/5       47 points      0           │
│  Generate     ◐ Partial     3/5       28 pairs       2           │
│  Cross-Doc    ○ Pending     -         -              -           │
└──────────────────────────────────────────────────────────────────┘

┌─ Checkpoint Info ────────────────────────────────────────────────┐
│  Last Stage:  generate                                           │
│  Last Update: 2026-01-08 08:44:40                                │
│  Tokens Used: 16,274                                             │
│  Est. Cost:   $0.42                                              │
└──────────────────────────────────────────────────────────────────┘
```

### `validate` - Check Output Quality

Validate output files and report errors:

```bash
python -m qa_extractor validate [OPTIONS]

Options:
  -o, --output PATH    Output directory to validate (required)
  --fix                Remove error files for re-processing
```

**Examples:**

```bash
# Check for errors
python -m qa_extractor validate -o ./output

# Check and fix errors (removes failed files so they can be re-processed)
python -m qa_extractor validate -o ./output --fix
```

**Sample Output:**

```
┌─ Validation Summary ─────────────────────────────────────────────┐
│  Knowledge Files    5 OK       QA Files        3 OK              │
│  Knowledge Errors   0          QA Errors       2                 │
│  Total Knowledge    47 points  Total QA        28 pairs          │
└──────────────────────────────────────────────────────────────────┘

                     Files with Errors
┌──────────┬────────────────────────────┬──────────────────────────┐
│ Type     │ File                       │ Error                    │
├──────────┼────────────────────────────┼──────────────────────────┤
│ QA       │ 1-s20-S0038092X...json     │ RetryError[HTTPStatus... │
│ QA       │ 1-s20-S2210271X...json     │ API returned empty co... │
└──────────┴────────────────────────────┴──────────────────────────┘

⚠ Found issues. Use --fix to remove error files for re-processing.
```

### `export` - Export Results

Export QA pairs to a single file:

```bash
python -m qa_extractor export [OPTIONS]

Options:
  -i, --input PATH     Input directory (required)
  -o, --output PATH    Output file path (required)
  -f, --format         Output format: json or jsonl (default: json)
  --by-category        Also export separate files by category
```

**Examples:**

```bash
# Export to single JSON file
python -m qa_extractor export -i ./output -o ./final_qa.json

# Export to JSONL format
python -m qa_extractor export -i ./output -o ./final_qa.jsonl -f jsonl

# Export with category split
python -m qa_extractor export -i ./output -o ./final_qa.json --by-category
```

### `init` - Generate Config Template

Generate a sample configuration file:

```bash
python -m qa_extractor init [OPTIONS]

Options:
  -o, --output PATH    Output path for config file (default: config.yaml)
```

**Example:**

```bash
python -m qa_extractor init -o my_config.yaml
```

### `clear` - Clear Checkpoint

Clear the checkpoint to start fresh:

```bash
python -m qa_extractor clear [OPTIONS]

Options:
  -o, --output PATH    Output directory to clear checkpoint from (required)
```

**Example:**

```bash
python -m qa_extractor clear -o ./output
```

## TUI Dashboard

When running the `run` command, a live TUI (Text User Interface) dashboard displays real-time progress:

```
╭──────────────────────────────────────────────────────────────────╮
│                                                                  │
│   ██████╗  █████╗     ███████╗██╗  ██╗████████╗██████╗           │
│  ██╔═══██╗██╔══██╗    ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗          │
│  ██║   ██║███████║    █████╗   ╚███╔╝    ██║   ██████╔╝          │
│  ██║▄▄ ██║██╔══██║    ██╔══╝   ██╔██╗    ██║   ██╔══██╗          │
│  ╚██████╔╝██║  ██║    ███████╗██╔╝ ██╗   ██║   ██║  ██║          │
│   ╚══▀▀═╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝          │
│              PhotoRAG Knowledge Extraction Pipeline              │
╰──────────────────────────────────────────────────────────────────╯

┌─ Configuration ─────────────────┐ ┌─ Token Usage ──────────────────┐
│ Model:    gemini-2.5-flash      │ │ ▸ Prompt:      12,453 tokens   │
│ Input:    ../MDs (5 files)      │ │ ▸ Completion:   3,821 tokens   │
│ Output:   ./output              │ │ ▸ Total:       16,274 tokens   │
│ Resume:   enabled               │ │ ▸ Est. Cost:   $0.42           │
└─────────────────────────────────┘ └────────────────────────────────┘

┌─ Progress ───────────────────────────────────────────────────────┐
│ ⏳ Stage 1: Extract Knowledge  ████████████░░░░░░░░░  3/5   60%  │
│ ○ Stage 2: Generate QA         ░░░░░░░░░░░░░░░░░░░░░  0/5    0%  │
│ ○ Stage 3: Cross-Doc QA        ░░░░░░░░░░░░░░░░░░░░░  0/1    0%  │
└──────────────────────────────────────────────────────────────────┘

┌─ Current Task ───────────────────────────────────────────────────┐
│ 📄 Processing: 1-s2.0-S0038092X2030832X-main.md                  │
│    Title: Effects of monohalogenated terminal units...           │
│    Status: ⏳ Calling LLM API...                                  │
└──────────────────────────────────────────────────────────────────┘

┌─ Activity Log ───────────────────────────────────────────────────┐
│ 08:41:22 │ ✓ Extracted 11 knowledge points from paper 1          │
│ 08:42:00 │ ✓ Extracted 12 knowledge points from paper 2          │
│ 08:42:45 │ ⏳ Processing paper 3...                               │
└──────────────────────────────────────────────────────────────────┘
```

### Dashboard Panels

| Panel | Description |
|-------|-------------|
| **Configuration** | Shows model, input/output paths, resume status |
| **Token Usage** | Real-time token counts and estimated cost |
| **Progress** | Multi-stage progress bars with percentages |
| **Current Task** | Currently processing file with status |
| **Activity Log** | Scrolling log of completed operations |

### Completion Summary

After the pipeline completes, a summary is displayed:

```
╭──────────────────────────────────────────────────────────────────╮
│                    ✓ Pipeline Complete                           │
╰──────────────────────────────────────────────────────────────────╯

┌─ Results Summary ────────────────────────────────────────────────┐
│   📚 Papers Processed    5                                       │
│   🧠 Knowledge Points    47                                      │
│   ❓ QA Pairs Generated  38                                      │
│   🔗 Cross-Doc QAs       10                                      │
└──────────────────────────────────────────────────────────────────┘

┌─ Token Usage & Cost ─────────────────────────────────────────────┐
│   Total Tokens        58,078      │   Duration     4m 23s        │
│   Est. Cost           $1.24       │   Avg/Paper    52.6s         │
└──────────────────────────────────────────────────────────────────┘
```

## Pipeline Overview

### Stage 1: Knowledge Extraction

The first stage reads each markdown file and extracts structured knowledge points.

**Input:** Markdown files (converted from academic papers)

**Output:** JSON files in `output/knowledge/`

**Process:**
1. Read and preprocess markdown content
2. Send to LLM with extraction prompt
3. Parse and validate knowledge points
4. Save results with token usage

**Knowledge Point Structure:**
```json
{
  "category": "Materials Design & Synthesis",
  "content": "BS3TSe-4F was designed using asymmetric selenium substitution...",
  "evidence": "Direct quote from paper...",
  "complexity": "single-hop",
  "keywords": ["selenium substitution", "asymmetric", "acceptor"]
}
```

### Stage 2: QA Generation

The second stage generates question-answer pairs from knowledge points.

**Input:** Knowledge point files from Stage 1

**Output:** JSON files in `output/qa_pairs/`

**Process:**
1. Load knowledge points for each paper
2. Send to LLM with generation prompt
3. Generate diverse QA pairs (different types and difficulties)
4. Validate and save results

**QA Pair Structure:**
```json
{
  "question": "What molecular design strategy was used in BS3TSe-4F?",
  "answer": "BS3TSe-4F was designed using asymmetric selenium substitution...",
  "category": "Materials Design & Synthesis",
  "source_title": "Achieving 19% Power Conversion Efficiency...",
  "difficulty": "medium",
  "reasoning_type": "single-hop"
}
```

### Cross-Document QA (Optional)

If enabled, generates QA pairs that require information from multiple papers.

**Output:** `output/qa_pairs/cross_doc.json`

## Output Format

### Directory Structure

```
output/
├── knowledge/                    # Stage 1 output
│   ├── paper_001.json
│   ├── paper_002.json
│   └── ...
├── qa_pairs/                     # Stage 2 output
│   ├── paper_001.json
│   ├── paper_002.json
│   ├── ...
│   └── cross_doc.json           # Cross-document QA pairs
├── final/                        # Exported results
│   ├── all_qa_pairs.json
│   └── qa_by_category/
│       ├── materials_design_synthesis.json
│       ├── performance_metrics.json
│       └── ...
├── stats/
│   └── summary_report.md        # Human-readable report
├── .checkpoint.json              # Resume checkpoint
└── qa_extractor.log             # Detailed log
```

### Final Export Format

**JSON Format (`all_qa_pairs.json`):**
```json
{
  "meta": {
    "generated_at": "2024-01-07T15:30:00",
    "model": "gpt-4o",
    "total_papers": 329,
    "total_qa_pairs": 3156,
    "categories": 8,
    "statistics": {
      "category_distribution": {...},
      "difficulty_distribution": {...},
      "reasoning_type_distribution": {...}
    }
  },
  "qa_pairs": [
    {
      "id": "qa_0001",
      "question": "What is the power conversion efficiency of...",
      "answer": "The PMHJ OSCs achieved a PCE of 18.48%...",
      "category": "Performance Metrics",
      "source_title": "Achieving 19% Power Conversion Efficiency...",
      "difficulty": "easy",
      "reasoning_type": "single-hop"
    },
    ...
  ]
}
```

**JSONL Format (`all_qa_pairs.jsonl`):**
```jsonl
{"id": "qa_0001", "question": "...", "answer": "...", "category": "...", ...}
{"id": "qa_0002", "question": "...", "answer": "...", "category": "...", ...}
```

## Advanced Usage

### Resuming Interrupted Runs

The tool automatically saves checkpoints. To resume:

```bash
# Simply run again - will continue from last checkpoint
python -m qa_extractor run -c config.yaml
```

To start fresh:
```bash
python -m qa_extractor run -c config.yaml --no-resume
# or
python -m qa_extractor clear -o ./output
python -m qa_extractor run -c config.yaml
```

### Processing Specific Files

To process only specific files, create a subdirectory with those files:

```bash
mkdir ./selected_papers
cp ../MDs/paper1.md ../MDs/paper2.md ./selected_papers/
python -m qa_extractor run -c config.yaml -i ./selected_papers
```

### Monitoring Token Usage

Token usage is tracked automatically and displayed in the CLI. To get detailed statistics:

```bash
# View in stats command
python -m qa_extractor stats -o ./output

# Check the log file
cat ./output/qa_extractor.log
```

### Customizing Prompts

To customize the extraction or generation prompts, edit the files:
- `qa_extractor/prompts/extraction.py` - Knowledge extraction prompts
- `qa_extractor/prompts/generation.py` - QA generation prompts

**Recent Improvements:**
- **Chain of Thought (CoT)**: Added an `analysis` field to the extraction JSON to encourage the model to think before extracting.
- **Self-Containment**: Enforced strict rules to replace pronouns (e.g., "It", "The device") with specific material names.
- **Source Tracing**: Added `source_knowledge_point_indices` to QA pairs to link questions back to their source facts.
- **Negative Constraints**: Added rules to prevent ambiguous questions like "What is the performance of the proposed material?".

### Running Stages Separately

For large datasets, you may want to run stages separately:

```bash
# Stage 1: Extract knowledge (can be interrupted and resumed)
python -m qa_extractor extract -i ../MDs -o ./output

# Verify extraction results
python -m qa_extractor stats -o ./output

# Stage 2: Generate QA pairs
python -m qa_extractor generate -i ./output/knowledge -o ./output

# Export final results
python -m qa_extractor export -i ./output -o ./final_qa.json --by-category
```

## Troubleshooting

### Common Issues

**1. API Key Not Set**
```
Error: API key not set. Set QA_EXTRACTOR_API_KEY environment variable...
```
Solution: Set the environment variable or add the key to config.yaml

**2. Connection Timeout**
```
Error: Request timeout
```
Solution: Increase `timeout` in config.yaml or check your network connection

**3. Rate Limiting**
```
Error: Rate limit exceeded
```
Solution: Increase `retry_delay` in config.yaml or reduce `batch_size`

**4. Invalid JSON Response**
```
Error: Failed to parse JSON response
```
Solution: This may happen with some models. Try a different model or lower temperature.

**5. Out of Memory**
```
Error: Memory allocation failed
```
Solution: Reduce `batch_size` or process fewer files at once

### Debug Mode

For detailed logging, check the log file:

```bash
tail -f ./output/qa_extractor.log
```

### Getting Help

```bash
# General help
python -m qa_extractor --help

# Command-specific help
python -m qa_extractor run --help
python -m qa_extractor export --help
```

## Knowledge Categories

The tool extracts knowledge and generates QA pairs in 8 categories:

| Category | Description | Example Topics |
|----------|-------------|----------------|
| Materials Design & Synthesis | Molecular design and synthesis | D-A structure, side chains, synthesis routes |
| Performance Metrics | Device performance parameters | PCE, VOC, JSC, FF, EQE |
| Structure-Property Relationships | Structure-performance correlations | Energy levels, morphology effects |
| Device Architecture & Physics | Device structure and mechanisms | BHJ, PMHJ, charge transport |
| Processing & Fabrication | Processing methods | Solvents, annealing, deposition |
| Characterization Methods | Measurement techniques | GIWAXS, AFM, spectroscopy |
| Stability & Degradation | Stability and lifetime | Thermal stability, degradation |
| Computational & Machine Learning | Theoretical methods | DFT, MD simulations, ML |

## License

This project is for research purposes.

## Acknowledgments

Built for the PhotoRAG project - a knowledge retrieval system for optoelectronic polymer materials research.
