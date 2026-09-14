# RAG Evaluation and Deployment API

[See the workflow flowchart and code walkthrough](WORKFLOW.md)

Deployable FastAPI project for source-grounded retrieval, citation-aware answers, and reproducible RAG evaluation.

This first version uses a pure-Python TF-IDF vector retriever so the API can run locally and deploy without model downloads, GPU access, or paid API keys. The retrieval interface is intentionally separated from the API layer so dense embedding backends can be added later.

## What It Shows

- FastAPI inference-style API with `/search`, `/answer`, `/metrics`, and `/health`
- Source-grounded answers with document IDs, snippets, and retrieval scores
- Reproducible evaluation set with Retrieval@1, Retrieval@3, citation coverage, and latency
- Dockerfile for container deployment
- Vercel and Render config files for cloud deployment paths
- Unit tests for API health, retrieval ranking, and cited answer responses

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts\seed_corpus.py
python scripts\run_evaluation.py
uvicorn rag_api.app:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Metrics: `http://127.0.0.1:8000/metrics`

## Test

```powershell
python -m pytest -q
```

## Docker

```powershell
docker build -t rag-evaluation-deployment-api .
docker run -p 8000:8000 rag-evaluation-deployment-api
```

## Current Verified Metrics

Run `python scripts\run_evaluation.py` to regenerate `outputs/evaluation_metrics.json`.

Latest local run:

| Metric | Value |
| --- | ---: |
| Documents indexed | 32 |
| Evaluation questions | 20 |
| Retrieval@1 | 0.95 |
| Retrieval@3 | 1.00 |
| Citation coverage | 1.00 |
| Average latency | 0.13 ms |
| p95 latency | 0.16 ms |

The included metrics are measured on a seeded local ML/MLOps knowledge base and should not be presented as production traffic or user data.
