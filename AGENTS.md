# Repository Guidelines

## Project Structure & Module Organization
`RAGPhoto_5/`, `Agentic_RAG_v2/`, `Data_Agentic_RAG/`, and `literature_agent/` are the main Python apps. Shared scripts live at the repo root (`cross_project_evaluation.py`, `quick_test.py`, `start_neo4j.sh`). Large corpora and generated outputs live in `Data/`, `MDs/`, `PDFs/`, `top_quality_mds/`, `AI_PDFs/`, and `evaluation_results/`; do not hand-edit derived artifacts. `Grading/frontend/` is the checked-in React/Vite app.

## Build, Test, and Development Commands
`bash start_neo4j.sh` starts the shared Neo4j service on ports `7474/7687`. `python3 quick_test.py` runs a 1-question end-to-end smoke test. `python3 cross_project_evaluation.py --sample-size 10 --verbose` exercises the cross-project evaluation flow. In `RAGPhoto_5/`, use `python run_pipeline_parallel.py --vector --kg --force-rebuild` to rebuild databases; in `Agentic_RAG_v2/`, use `./run_optics_ui.sh` or `streamlit run ui/app_optics.py`. For `Grading/frontend/`, use `npm install`, `npm run dev`, `npm run build`, and `npm run lint`.

## Coding Style & Naming Conventions
Use standard Python style: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and explicit constants for paths and ports. Match the surrounding file's language and logging style. No repo-wide formatter is checked in, so keep edits local and conservative. Frontend TypeScript/React code is governed by ESLint (`Grading/frontend/eslint.config.js`); keep components in `PascalCase` and helpers in `camelCase`.

## Testing Guidelines
Prefer `pytest` for Python test suites. Test files use `test_*.py`, with integration-style checks under `Agentic_RAG_v2/tests/` and `Data_Agentic_RAG/evaluation/tests/`. Use the executable smoke scripts in `Agentic_RAG_v2/tests/scripts/` when you need to confirm imports, retrieval, or UI launch. Many tests require Milvus, Neo4j, and API keys, so note skipped or environment-dependent cases in your PR.

## Commit & Pull Request Guidelines
Follow the existing concise imperative style: `Add ...`, `Update ...`, `Initial commit ...`. Keep commits scoped to one subproject when practical. PRs should state what changed, which commands were run, any required services/config, and screenshots for UI work. Mention data or cache regeneration explicitly when outputs under `MDs/`, `Data/`, or evaluation folders change.

## Security & Configuration Tips
Keep secrets in `.env` files or shell exports, never in source. Avoid committing database dumps, caches, or large generated files unless they are intentional inputs to the repo.
