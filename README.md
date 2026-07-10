<p align="center">
  <img src="logo.png" alt="RAGPhoto logo" width="200">
</p>

<h1 align="center">RAGPhoto</h1>

RAGPhoto is an agentic retrieval-augmented generation method for source-attributed question answering over the organic photovoltaic (OPV) literature. It combines a section-aware Milvus passage index, an OPV-typed Neo4j knowledge graph, and a retrieve-check-rewrite loop of up to three rounds.

This README follows the statistics and evaluation scope reported in the manuscript. Some development files in the repository and the first Figshare archive predate the final manuscript analysis; those differences are identified below.

## Manuscript scope

| Component | Manuscript value |
|---|---:|
| Source publications | 1,575 |
| Section-aware passages | 64,343 |
| Knowledge-graph entities | 72,775 |
| Knowledge-graph relations | 409,092 |
| Expert-reviewed benchmark questions | 1,511 |
| Answer-generation backends | 5 |
| Automated evaluation model | GLM-4.7 |

The five answer-generation backends are:

- Llama-3.3-70B
- DeepSeek-V3.2-Thinking
- Qwen3-235B
- GPT-4o-mini
- MiniMax-M2.1

GLM-4.7 is the common automated judge for the manuscript results. It is not one of the five answer-generation backends in the reported comparison.

Three configurations are reported:

| Configuration | Vector retrieval | Knowledge graph | Reflection and validation |
|---|:---:|:---:|:---:|
| `RAGPhoto_LLM` | No | No | No |
| `RAGPhoto_no_kg` | Yes | No | Yes |
| `RAGPhoto_kg` | Yes | Yes | Yes |

Answer correctness in the manuscript is:

| Backend | `RAGPhoto_LLM` | `RAGPhoto_no_kg` | `RAGPhoto_kg` |
|---|---:|---:|---:|
| Llama-3.3-70B | 0.329 | 0.916 | 0.918 |
| DeepSeek-V3.2-Thinking | 0.456 | 0.923 | 0.934 |
| Qwen3-235B | 0.451 | 0.944 | 0.951 |
| GPT-4o-mini | 0.353 | 0.899 | 0.920 |
| MiniMax-M2.1 | 0.350 | 0.908 | 0.917 |

Most of the difference from direct generation is already present in the iterative vector-retrieval configuration. Adding the knowledge graph gives smaller average increments, from 0.002 to 0.021 depending on the backend, with larger differences on selected relation-dense cases.

The six reported metrics are Answer Correctness, Answer Relevancy, Faithfulness, Context Recall, Context Precision, and Answer Similarity. A small number of missing outputs were not imputed, so configuration-level means are descriptive summaries rather than strictly paired per-question estimates.

## Public QA files

The manuscript benchmark contains 1,511 expert-reviewed question-answer pairs. The file [`photorag_QA/QA/final/all_qa_pairs.json`](photorag_QA/QA/final/all_qa_pairs.json) currently contains a larger 2,644-item candidate export produced before the final expert screening. It is useful for developing the extraction and evaluation code, but it should not be treated as the exact benchmark used for the manuscript tables.

The Figshare v1 evaluation archive also contains development runs beyond the five-backend manuscript comparison, including runs in which GLM-4.7 generated answers. For manuscript-aligned model roles and numerical results, use the definitions and tables in this README and the paper.

## Repository layout

```text
RAGPhoto/
|-- agentic_rag/          Agentic retrieval runtime and Streamlit interface
|-- data_pipeline/        Passage indexing and knowledge-graph construction
|-- evaluation/           Evaluation, filtering, analysis, and plotting scripts
|-- photorag_QA/          QA extraction tools and candidate QA exports
|-- examples/             Example queries
`-- README.md
```

## Requirements

- Python 3.10 or later
- Docker with Docker Compose v2
- An OpenAI-compatible endpoint for answer generation
- An embedding endpoint for `text-embedding-3-small`
- A Qwen3-reranker-8b-compatible endpoint for the manuscript retrieval setup

The full paper evaluation used several hosted and local model endpoints. Running the included interface requires only one configured generation backend; reproducing every manuscript cell requires access to the corresponding five model checkpoints or APIs and GLM-4.7 for automated evaluation.

## Quick start

Clone the repository and create an environment:

```bash
git clone https://github.com/wzfeng768/RAGPhoto.git
cd RAGPhoto

python -m venv .venv
source .venv/bin/activate
pip install -r data_pipeline/config/requirements.txt
pip install -r agentic_rag/requirements.txt
```

Download `milvus_data.zip` from Figshare, then extract it and start the database services:

```bash
unzip /path/to/milvus_data.zip -d data_pipeline/config/
cd data_pipeline/config
docker compose up -d
docker ps --filter "name=test-"
```

The Compose file exposes these local ports:

| Service | Address |
|---|---|
| Milvus | `localhost:19531` |
| Neo4j Bolt | `bolt://localhost:7688` |
| Neo4j Browser | `http://localhost:7475` |

The bundled Compose credentials are intended for local testing. Change them before exposing either service outside the local machine.

Copy the agent configuration template:

```bash
cd ../../agentic_rag
cp config/env_example.txt .env
```

Set the API endpoints and keys in `.env`. For the paper-aligned retrieval settings, override the older example defaults with:

```dotenv
LLM_TEMPERATURE=0
MILVUS_PORT=19531
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=testpassword
MAX_ITERATIONS=3
QUALITY_THRESHOLD=0.70
ENABLE_EARLY_STOPPING=true
EARLY_STOP_THRESHOLD=0.90
VECTOR_TOP_K=20
ENABLE_RERANKING=true
RERANKER_MODEL=qwen3-reranker-8b
RERANKER_TOP_N=8
```

Temperature was 0 for both answer generation and automated evaluation in the manuscript runs.

Launch the Streamlit interface:

```bash
./run_optics_ui.sh
```

The default interface is available at `http://localhost:8503`. A single OPV query can also be run from the command line:

```bash
python run_agentic.py "How does molecular orientation affect charge transport in an OPV active layer?"
```

## Evaluation code

Install the evaluation dependencies from the repository root:

```bash
pip install -r evaluation/requirements.txt
```

The evaluation scripts support the three configuration names used in the paper:

```text
direct_llm          -> RAGPhoto_LLM
agentic_no_kg       -> RAGPhoto_no_kg
agentic_with_kg     -> RAGPhoto_kg
```

The exact manuscript benchmark is the 1,511-item expert-reviewed set, not the 2,644-item candidate export currently stored under `photorag_QA/QA/final/`. Avoid using the older aggregate counts or weighted-score examples in the development documentation when comparing against the manuscript tables.

The evaluation CLI accepts separate answer and judge models. Both temperatures should be set to 0 for manuscript-aligned runs:

```bash
cd evaluation
python evaluate_rag.py \
  --dataset computational \
  --num_questions 5 \
  --mode all \
  --answer-model <generation-model> \
  --answer-temperature 0 \
  --eval-model <glm-4.7-model-id> \
  --eval-temperature 0 \
  --results-subdir <run-name>
```

This command is a small operational check, not a reproduction of the full paper benchmark.

## Data availability

The current Figshare record is [10.6084/m9.figshare.32194434](https://doi.org/10.6084/m9.figshare.32194434) and is licensed under CC BY 4.0. Version 1 exposes these archives:

| Archive | Compressed size | Contents |
|---|---:|---|
| `milvus_data.zip` | 1,127,700,682 bytes | Persistent database files used by the Docker services |
| `evaluation_results.zip` | 83,950,206 bytes | Development and manuscript-related evaluation outputs |

Extract them into the paths expected by the repository:

```bash
unzip milvus_data.zip -d data_pipeline/config/
unzip evaluation_results.zip -d evaluation/
```

The source publications are not redistributed in this repository. They must be obtained through publisher APIs or institutional subscriptions under the applicable terms.

## Rebuilding the knowledge substrates

The databases can be rebuilt from locally obtained source documents:

```bash
cd data_pipeline

# Serial build
python run_pipeline.py --vector --kg

# Parallel build; choose a worker count appropriate for local hardware and API limits
python run_pipeline_parallel.py --vector --kg --num-workers 8 --batch-size 10
```

Rebuilding requires access to the source documents and configured embedding and extraction models. Exact counts can differ if the source set, model checkpoint, prompt, or preprocessing version changes.

## Scope and limitations

- The reported LLM-based scores use GLM-4.7 as a single automated judge. No independent expert-rating study was conducted for the manuscript.
- Knowledge-graph extraction was not benchmarked against a separately annotated gold graph.
- The cross-paper PCE example is conditioned on retrieved passages and is not a field-wide meta-analysis.
- The current system reasons over text. It does not directly analyze raw spectra, microscopy images, or device diagrams.

## Citation

The Figshare dataset can be cited as:

```text
Feng, W. (2026). RAGPhoto. figshare. Dataset.
https://doi.org/10.6084/m9.figshare.32194434
```

Please cite the accompanying RAGPhoto paper for the method and reported benchmark results once its bibliographic record is available.

## License

The Figshare dataset is distributed under CC BY 4.0. This GitHub repository does not currently declare a separate code license; contact the authors about code reuse until a license file is added. Source publications remain subject to their publishers' terms.
