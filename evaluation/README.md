# RAG System Evaluation

This evaluation suite compares three question-answering approaches across multiple datasets from the PhotoRAG QA collection.

## Evaluation Modes

1. **Agentic RAG + Knowledge Graph**: Full agentic RAG system with knowledge graph enhancement
2. **Agentic RAG (Vector Only)**: Agentic RAG with only vector retrieval (no KG)
3. **Direct LLM**: Direct LLM response without any retrieval

## Available Datasets

The system supports 9 different datasets:

| Dataset | Questions | Description |
|---------|-----------|-------------|
| `all` | 2,644 | All questions across all categories |
| `computational` | 67 | Computational & Machine Learning |
| `characterization` | 334 | Characterization Methods |
| `stability` | 108 | Stability & Degradation |
| `materials` | 359 | Materials Design & Synthesis |
| `device` | 399 | Device Architecture & Physics |
| `structure` | 621 | Structure-Property Relationships |
| `processing` | 291 | Processing & Fabrication |
| `performance` | 465 | Performance Metrics |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Parent System

Ensure the agentic_rag system is properly configured:
- Milvus vector database is running
- Neo4j graph database is running (for KG mode)
- OpenAI API key is configured in `../agentic_rag/.env`

## Running Evaluation

### Quick Start (default dataset: computational, 10 questions)

```bash
python evaluate_rag.py
```

### Dataset Selection

```bash
# Evaluate specific dataset
python evaluate_rag.py --dataset computational --num_questions 10
python evaluate_rag.py --dataset performance --num_questions 20
python evaluate_rag.py --dataset all --num_questions 100

# Evaluate all questions in a dataset
python evaluate_rag.py --dataset computational --num_questions all
```

### Mode Selection

```bash
# Only evaluate specific mode (faster)
python evaluate_rag.py --dataset computational --mode direct_llm --num_questions 10
python evaluate_rag.py --dataset computational --mode agentic_with_kg --num_questions 10
python evaluate_rag.py --dataset computational --mode agentic_no_kg --num_questions 10

# Evaluate all modes (default)
python evaluate_rag.py --dataset computational --mode all --num_questions 10
```

### Combined Examples

```bash
# Quick test: 5 questions, direct LLM only, computational dataset
python evaluate_rag.py --dataset computational --mode direct_llm --num_questions 5

# Full evaluation: all questions, all modes, computational dataset
python evaluate_rag.py --dataset computational --mode all --num_questions all

# Large scale: 100 questions from all datasets
python evaluate_rag.py --dataset all --mode all --num_questions 100
```

### Incremental Evaluation & Resume ⭐New

The evaluation system now supports **incremental saving** (enabled by default) and **resume** functionality:

```bash
# Standard evaluation (incremental save enabled by default)
python evaluate_rag.py --dataset computational --num_questions 50

# If interrupted, resume from where you left off
python evaluate_rag.py --dataset computational --num_questions 50 --resume

# Use batch mode (disable incremental save)
python evaluate_rag.py --dataset computational --num_questions 10 --no-incremental

# Skip RAGAS evaluation if unstable (use only fact-checking) ⚠️Recommended
python evaluate_rag.py --dataset computational --num_questions 10 --skip-ragas
```

**Benefits of Incremental Evaluation:**
- ✅ **No progress loss**: Each completed question is immediately saved
- ✅ **Resume support**: Continue from interruption point with `--resume`
- ✅ **Real-time monitoring**: View results while evaluation is running
- ✅ **Automatic NaN retry**: Retries up to 3 times for NaN RAGAS metrics
- ✅ **Detailed logs**: Comprehensive output for each question

**Example: Resume after interruption**
```bash
# Start evaluation of 100 questions
python evaluate_rag.py --dataset all --num_questions 100 --mode direct_llm

# Interrupted at question 45 (network issue, API limit, etc.)
# Results for 45 questions are already saved

# Resume from question 46
python evaluate_rag.py --dataset all --num_questions 100 --mode direct_llm --resume

# Output: 
# 📂 Found existing progress: 45 questions already completed
#    Resuming from question 46...
```

**Monitoring progress in real-time:**
```bash
# Check how many questions completed
cat results/computational/direct_llm/results.json | jq '.metadata.num_questions_completed'

# Check status
cat results/computational/direct_llm/results.json | jq '.metadata.status'
# Output: "in_progress" or "completed"
```

### Custom Model Configuration ⭐New

You can now specify different models for answer generation and evaluation:

**Use different models:**
```bash
# Use gpt-4o-mini for answering, gpt-4 for evaluation
python evaluate_rag.py \
  --dataset computational \
  --num_questions 10 \
  --answer-model gpt-4o-mini \
  --eval-model gpt-4
```

**Use different API providers:**
```bash
# Answer generation with OpenAI, evaluation with Alibaba Qwen
python evaluate_rag.py \
  --dataset computational \
  --num_questions 10 \
  --answer-model gpt-4o-mini \
  --answer-api-base https://api.openai.com/v1 \
  --answer-api-key sk-... \
  --eval-model qwen-plus \
  --eval-api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --eval-api-key sk-...
```

**Customize temperature:**
```bash
# Set different temperatures for generation and evaluation
python evaluate_rag.py \
  --dataset computational \
  --num_questions 10 \
  --answer-model gpt-4o-mini \
  --answer-temperature 0.7 \
  --eval-model gpt-4 \
  --eval-temperature 0.0
```

**Benefits:**
- ✅ **Flexibility**: Use different models for generation vs evaluation
- ✅ **Cost optimization**: Cheap model for generation, quality model for evaluation
- ✅ **Multi-provider**: Support OpenAI, Alibaba Qwen, DeepSeek, etc.
- ✅ **Backward compatible**: Works with default config if no parameters provided
- ✅ **Full tracking**: All model info recorded in result metadata

## Results

Results are organized by dataset in the `results/` directory:

```
results/
├── computational/
│   ├── agentic_with_kg/
│   │   └── results.json
│   ├── agentic_no_kg/
│   │   └── results.json
│   ├── direct_llm/
│   │   └── results.json
│   └── evaluation_summary.json
├── performance/
│   ├── agentic_with_kg/
│   │   └── results.json
│   └── ...
└── all/
    └── ...
```

### Analyzing Results

Use the analysis script to correctly interpret results:

```bash
# Analyze all available datasets
python analyze_results.py

# Analyze specific dataset
python analyze_results.py --dataset computational
python analyze_results.py --dataset all
```

This will:
- Load all available results for specified dataset(s)
- Display metrics for each mode correctly
- Show fair comparisons between modes
- Highlight which metrics are valid for each mode
- Show weighted scores (factual correctness + RAGAS)

## Evaluation Metrics

### RAGAS Metrics (6 metrics)

All 6 RAGAS metrics are evaluated for each mode:

1. **Faithfulness**: How faithful the answer is to the retrieved context (0 for direct LLM without context)
2. **Answer Relevancy**: How relevant the answer is to the question
3. **Context Precision**: Precision of retrieved contexts (0 for direct LLM without context)
4. **Context Recall**: Recall of retrieved contexts compared to ground truth (0 for direct LLM without context)
5. **Answer Correctness**: Correctness compared to ground truth answer
6. **Answer Similarity**: Semantic similarity to ground truth answer

### Factual Correctness Metrics (7 metrics)

Custom fact-checking metrics prioritizing factual accuracy:

1. **Factual Correctness**: Overall factual accuracy score (with one-vote veto for critical errors)
2. **Key Facts Coverage**: Coverage of key facts from ground truth
3. **Fact Accuracy**: Accuracy of facts after error penalties
4. **Fact Precision**: Proportion of correct facts in generated answer
5. **Fact Recall**: Recall of facts from ground truth
6. **Hallucination Rate**: Rate of unsupported factual claims
7. **Context Supported Rate**: Rate of facts supported by retrieved contexts

**⭐ Intelligent Numerical Evaluation:**

The system now recognizes semantically equivalent numerical expressions, avoiding overly strict literal matching:

| Ground Truth | Accepted as Correct |
|-------------|---------------------|
| 98%-99% | "approximately 100%", "nearly 100%", "close to 100%", "接近100%" |
| 50% | "around 50%", "about half", "约50%" |
| 10-15% | "roughly 10-15%", "approximately 10-15%", "大约10-15%" |

**Evaluation Principles:**
- Focus on semantic meaning rather than exact wording
- 1-2% deviations with approximate qualifiers ("about", "approximately", "接近") are considered correct
- Overlapping or adjacent numerical ranges are treated as compatible
- Supports both English and Chinese approximate expressions

### Weighted Score (Recommended)

The **weighted score** combines both evaluations:
- Factual Correctness: 70% (prioritized)
- RAGAS Average: 30% (supporting)

This ensures factually correct answers score higher even if phrasing is simple.

## Output Format

Each result file contains:
- **Metadata**: model, mode, dataset, timestamp, kg_enabled flag
- **Individual Q&A Results**: Each question includes:
  - question_id, question, ground_truth (answer)
  - generated_answer, contexts, sources
  - response_time, iterations, quality_score
  - category, source_title, difficulty, reasoning_type
  - error (if any)
- **Aggregated RAGAS Metrics**: All 6 metrics plus average_score
- **Fact Correctness Metrics**: All 7 fact-checking metrics
- **Weighted Score**: Combined score (70% fact + 30% RAGAS)
- **Performance Statistics**: avg_response_time, avg_iterations, total_time, error_count

## Troubleshooting

### RAGAS Evaluation Timeout or Errors ⚠️ Common Issue

**Symptoms:**
- `TimeoutError` during RAGAS evaluation
- `IndexError: list index out of range` 
- `RuntimeError: Event loop is closed`
- NaN values in metrics
- Program hangs during "Computing RAGAS metrics..."

**Solution:**
Use the `--skip-ragas` flag to skip RAGAS evaluation and use only fact-checking:

```bash
python evaluate_rag.py --dataset computational --num_questions 10 --skip-ragas
```

This is recommended for production evaluations as:
- ✅ Faster (no timeout delays)
- ✅ More stable
- ✅ Fact-checking metrics are often more reliable for accuracy assessment

### RAGAS Not Installed

```bash
pip install ragas datasets langchain-openai
```

### Neo4j Connection Error

If Neo4j is not available, the system will automatically disable KG for the no-KG mode. Ensure Neo4j is running for the with-KG evaluation.

### Milvus Connection Error

Verify Milvus is running:

```bash
# Check if Milvus is accessible
python -c "from pymilvus import connections; connections.connect(host='localhost', port='19531')"
```

## Important Notes

### Dataset Selection Strategy

- **Do NOT evaluate `all` and individual categories together** - the `all` dataset contains all category questions, so comparing them would be redundant
- Use `all` for comprehensive cross-category analysis
- Use individual categories (e.g., `computational`) for focused domain testing
- Results are separated by dataset to avoid confusion

### Performance Considerations

- Each question takes ~10-30 seconds (Agentic modes) or ~5-10 seconds (Direct LLM)
- Large datasets like `performance` (465 questions) or `all` (2,644 questions) will take hours
- Start with small tests (`--num_questions 5`) before running full evaluations

### Other Notes

- By default, the evaluation uses the LLM model configured in agentic_rag, but you can customize models via command-line parameters
- Set `verbose=True` in config.py to see detailed logs during evaluation
- Results include response times for performance comparison
- Factual correctness metrics use English prompts for better LLM understanding
- All model configurations (answer generation and evaluation) are fully tracked in result metadata

