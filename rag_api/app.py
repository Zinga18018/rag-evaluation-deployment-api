from __future__ import annotations

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from rag_api.corpus import load_documents
from rag_api.evaluation import build_default_report
from rag_api.retriever import TfidfRagIndex


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


documents = load_documents()
index = TfidfRagIndex(documents)

app = FastAPI(
    title="RAG Evaluation and Deployment API",
    version="1.1.0",
    description="TF-IDF retrieval and traceable extractive answers with explicit abstention and development evaluation.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "rag-evaluation-deployment-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, int | str]:
    return {"status": "ok", "documents_indexed": len(documents)}


@app.get("/search")
def search(q: str = Query(..., min_length=3), top_k: int = Query(default=3, ge=1, le=5)) -> dict:
    results = index.search(q, top_k=top_k)
    return {"query": q, "results": [result.__dict__ for result in results]}


@app.post("/answer")
def answer(request: QueryRequest) -> dict:
    result = index.answer(request.question, top_k=request.top_k)
    return {
        "question": request.question,
        "answer": result.answer,
        "latency_ms": result.latency_ms,
        "sources": [source.__dict__ for source in result.sources],
        "status": result.status,
        "abstention_reason": result.abstention_reason,
        "retrieved_sources": [source.__dict__ for source in result.retrieved_sources],
    }


@app.get("/metrics")
def metrics() -> dict:
    return build_default_report()

