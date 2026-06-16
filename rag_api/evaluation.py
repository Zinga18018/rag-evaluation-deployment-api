from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from rag_api.corpus import default_corpus_dir, load_documents
from rag_api.retriever import TfidfRagIndex


def default_eval_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "eval_set.json"


def load_eval_set(eval_path: Path | None = None) -> list[dict[str, str]]:
    path = eval_path or default_eval_path()
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(index: TfidfRagIndex, eval_set: list[dict[str, str]], top_k: int = 3) -> dict[str, Any]:
    latencies: list[float] = []
    top1_hits = 0
    topk_hits = 0
    citation_hits = 0

    for row in eval_set:
        start = time.perf_counter()
        answer = index.answer(row["question"], top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        ranked_ids = [source.doc_id for source in answer.sources]
        expected = row["expected_doc_id"]
        top1_hits += int(bool(ranked_ids) and ranked_ids[0] == expected)
        topk_hits += int(expected in ranked_ids[:top_k])
        citation_hits += int(bool(answer.sources))

    total = len(eval_set)
    return {
        "documents_indexed": len(index.documents),
        "eval_questions": total,
        "retrieval_at_1": round(top1_hits / total, 4),
        f"retrieval_at_{top_k}": round(topk_hits / total, 4),
        "citation_coverage": round(citation_hits / total, 4),
        "avg_latency_ms": round(statistics.mean(latencies), 3),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
    }


def build_default_report() -> dict[str, Any]:
    index = TfidfRagIndex(load_documents(default_corpus_dir()))
    return evaluate(index, load_eval_set())


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = min(int(round((len(ordered) - 1) * percentile)), len(ordered) - 1)
    return ordered[rank]

