<p align="center">
  <img src="logo.png" alt="RAGPhoto Logo" width="200">
</p>

<h1 align="center">RAGPhoto</h1>

<p align="center">
  Agentic Retrieval-Augmented Generation for Traceable Organic Photovoltaic Literature Question Answering
</p>

RAGPhoto is an agentic retrieval-augmented generation framework for traceable question answering over the organic photovoltaic (OPV) literature. It couples a **Milvus vector store** (64,343 section-aware chunks from 1,738 publications) with a **Neo4j domain knowledge graph** (76,520 entities, 363,948 relations) through **reflective multi-turn orchestration**, and serves answers with full source attribution via a Streamlit web interface.

Evaluated on a 1,511-question benchmark across six LLM backends, RAGPhoto raises answer correctness from 0.33--0.46 (bare LLM) to 0.92--0.95, with weaker models gaining the most (model equalizer effect).

---

## Project Structure

```
Data_Agentic_RAG/
├── data_pipeline/                # Data processing pipeline
│   ├── core/                     # Document loader, text splitter, vector/KG managers
│   ├── config/
│   │   ├── docker-compose.yml    # Milvus + Neo4j service definitions
│   │   ├── requirements.txt
│   │   └── milvus_data/          # Persisted database files (~2.5 GB)
│   ├── run_pipeline.py           # Serial pipeline
│   └── run_pipeline_parallel.py  # Parallel pipeline
│
├── agentic_rag/                  # Agentic RAG runtime
│   ├── core/                     # Agent orchestrator, reflection engine, RAG components
│   ├── tools/                    # Extensible tool system (web search, etc.)
│   ├── ui/
│   │   └── app_optics.py         # Streamlit frontend
│   ├── config/
│   │   └── env_example.txt       # Environment variable template
│   ├── run_agentic.py            # CLI entry point
│   ├── run_optics_ui.sh          # Streamlit launcher script
│   └── requirements.txt
│
├── evaluation/                   # Evaluation scripts and benchmark tools
└── verify_databases.py           # Database health check utility
```

---

## Knowledge Substrate

| Component | Metric | Value |
|-----------|--------|-------|
| Source corpus | Publications | 1,738 |
| | Document units | 1,575 |
| | Publication years | 2000--2024 |
| Vector database (Milvus) | Total chunks | 64,343 |
| | Embedding dimensions | 1,536 |
| | Total characters | ~55.8M |
| | Mean chunk size | 927 chars |
| Knowledge graph (Neo4j) | Entity nodes | 76,520 |
| | Relation edges | 363,948 |
| | Entity types | 5 (Material, Device, Property, Method, Characterization) |
| | Relation types | 8+ (ENHANCES, AFFECTS, COMPOSED_OF, etc.) |

---

## Prerequisites

- **Python** 3.10+
- **Docker** and **Docker Compose** v2
- An **OpenAI-compatible API key** (for embeddings and LLM generation)

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd Data_Agentic_RAG
```

### 2. Install Python dependencies

```bash
# Data pipeline dependencies
pip install -r data_pipeline/config/requirements.txt

# Agentic RAG dependencies
pip install -r agentic_rag/requirements.txt
```

### 3. Start database services

The project uses Docker Compose to run Milvus (vector DB), Neo4j (knowledge graph), etcd, and MinIO. Pre-built database files are included under `data_pipeline/config/milvus_data/`.

```bash
cd data_pipeline/config
docker compose up -d
```

Verify the containers are running:

```bash
docker ps --filter "name=test-"
```

You should see four containers: `test-milvus-standalone`, `test-milvus-etcd`, `test-milvus-minio`, and `test-neo4j`.

| Service | Host Port | Description |
|---------|-----------|-------------|
| Milvus gRPC | `localhost:19531` | Vector database |
| Neo4j Bolt | `localhost:7688` | Knowledge graph |
| Neo4j Browser | `http://localhost:7475` | Graph visualization (user: `neo4j`, password: `testpassword`) |

Return to the project root:

```bash
cd ../..
```

### 4. Configure environment variables

```bash
cd agentic_rag
cp config/env_example.txt .env
```

Edit `.env` and set your API keys:

```bash
# Required
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint

# Optional -- for reranking (improves retrieval quality)
RERANKER_API_KEY=your_reranker_key
RERANKER_BASE_URL=https://your-reranker-endpoint/v1
```

All other settings (model names, database ports, retrieval parameters) have working defaults. See `agentic_rag/config/env_example.txt` for the full list.

### 5. (Optional) Verify database health

From the project root:

```bash
python verify_databases.py
```

This checks connectivity to both Milvus and Neo4j and reports entity/vector counts.

### 6. Launch the Streamlit frontend

```bash
cd agentic_rag
./run_optics_ui.sh
```

The script checks that Docker services are running, installs any missing Python packages, and starts Streamlit. Open your browser at:

```
http://localhost:8503
```

You can also launch Streamlit directly:

```bash
cd agentic_rag
streamlit run ui/app_optics.py --server.port 8503
```

---

## CLI Usage (Optional)

```bash
cd agentic_rag

# Single query
python run_agentic.py "What is the efficiency of perovskite solar cells?"

# Interactive mode
python run_agentic.py --interactive

# Batch evaluation
python run_agentic.py --evaluate --sample-size 5
```

---

## Rebuilding the Databases (Optional)

The repository ships with pre-built databases. To rebuild from source documents:

```bash
cd data_pipeline

# Parallel build (recommended)
python run_pipeline_parallel.py --kg --num-workers 256 --batch-size 10

# Serial build
python run_pipeline.py --vector --kg
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vector database | Milvus 2.3 (IVF_FLAT index, cosine similarity) |
| Knowledge graph | Neo4j 5.15 |
| Embedding model | OpenAI text-embedding-3-small (1,536 dimensions) |
| LLM | Configurable (tested: Qwen3-235B, DeepSeek-V3.2, Llama-3.3-70B, GLM-4.7, GPT-4o-mini, MiniMax-M2.1) |
| Reranker | Qwen3-reranker-8b (configurable) |
| Frontend | Streamlit |
| Visualization | Plotly, NetworkX |
| Orchestration | Docker Compose |
