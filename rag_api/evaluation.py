from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import statistics
import time
from typing import Any

from rag_api.corpus import default_corpus_dir, load_documents
from rag_api.retriever import TfidfRagIndex


ROOT = Path(__file__).resolve().parents[1]


def default_eval_path() -> Path:
    return ROOT / "data" / "eval_set.json"


def load_eval_set(eval_path: Path | None = None) -> list[dict[str, Any]]:
    paths = [eval_path] if eval_path else [default_eval_path(), ROOT / "data" / "eval_challenge.json"]
    rows = []
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8-sig")):
            record = dict(row)
            record.setdefault("answerable", bool(record.get("expected_doc_id")))
            record.setdefault("relevant_doc_ids", [record["expected_doc_id"]] if record.get("expected_doc_id") else [])
            record.setdefault("category", "original_answerable_fixture")
            record.setdefault("label_status", "pending_human_review")
            record.setdefault("provenance", "Inherited authored development fixture; reference label not independently reviewed.")
            rows.append(record)
    return rows


def ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evaluate(index: TfidfRagIndex, eval_set: list[dict[str, Any]], top_k: int = 3) -> dict[str, Any]:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    ids = [row.get("question_id") for row in eval_set]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("evaluation case IDs must be nonempty and unique")
    documents = {doc.doc_id: doc for doc in index.documents}
    latencies = []
    records = []
    counts: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = {}
    for row in eval_set:
        expected = row.get("relevant_doc_ids", [row["expected_doc_id"]] if row.get("expected_doc_id") else [])
        answerable = row.get("answerable", bool(expected))
        if not isinstance(answerable, bool) or not isinstance(expected, list) or any(not isinstance(doc_id, str) for doc_id in expected):
            raise ValueError("answerable must be boolean and relevant_doc_ids a string list")
        if len(set(expected)) != len(expected) or answerable != bool(expected) or set(expected) - set(documents):
            raise ValueError("answerable cases require unique existing relevant sources; negatives require none")
        start = time.perf_counter()
        answer = index.answer(row["question"], top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        retrieved_ids = [source.doc_id for source in answer.retrieved_sources]
        cited_ids = [source.doc_id for source in answer.sources]
        answered = answer.status == "answered"
        relevant = set(expected)
        top1 = bool(retrieved_ids) and retrieved_ids[0] in relevant
        topk = bool(relevant.intersection(retrieved_ids[:top_k]))
        cited_relevant = len([doc_id for doc_id in cited_ids if doc_id in relevant])
        cited_valid = len([doc_id for doc_id in cited_ids if doc_id in documents])
        quotes_valid = sum(source.doc_id in documents and bool(source.snippet.strip())
                           and _normalized(source.snippet) in _normalized(documents[source.doc_id].text)
                           and f"{source.snippet} [{source.doc_id}]" in answer.answer
                           for source in answer.sources)
        failures = []
        if answerable and not topk:
            failures.append("relevant_source_not_retrieved")
        if answerable and not answered:
            failures.append("abstained_on_answerable_case")
        if not answerable and answered:
            failures.append("answered_unanswerable_case")
        if len(cited_ids) > cited_relevant:
            failures.append("cited_source_not_in_provisional_relevance_labels")
        if cited_valid < len(cited_ids):
            failures.append("unknown_citation_id")
        if quotes_valid < len(cited_ids):
            failures.append("quote_or_citation_not_traceable")
        if answered and not cited_ids:
            failures.append("answered_without_cited_sources")
        if not answered and cited_ids:
            failures.append("abstained_with_cited_sources")
        counts["answerable"] += int(answerable)
        counts["unanswerable"] += int(not answerable)
        counts["answered"] += int(answered)
        counts["abstained"] += int(not answered)
        counts["tp_answerable"] += int(answerable and answered)
        counts["fn_answerable"] += int(answerable and not answered)
        counts["fp_unanswerable"] += int(not answerable and answered)
        counts["tn_unanswerable"] += int(not answerable and not answered)
        counts["retrieval_top1"] += int(top1)
        counts["retrieval_topk"] += int(topk)
        counts["retrieved_relevant"] += sum(doc_id in relevant for doc_id in retrieved_ids)
        counts["retrieved_total_on_answerable"] += len(retrieved_ids) if answerable else 0
        counts["retrieval_present"] += int(bool(retrieved_ids))
        counts["cited_total"] += len(cited_ids)
        counts["cited_relevant"] += cited_relevant
        counts["cited_valid"] += cited_valid
        counts["quotes_traceable"] += quotes_valid
        counts["answerable_with_relevant_citation"] += int(answerable and cited_relevant > 0)
        counts["answered_with_citations"] += int(answered and bool(cited_ids))
        counts["negative_with_retrieval"] += int(not answerable and bool(retrieved_ids))
        counts["negative_with_retrieval_abstained"] += int(not answerable and bool(retrieved_ids) and not answered)
        counts["failed_cases"] += int(bool(failures))
        category = row.get("category", "unspecified_development")
        group = categories.setdefault(category, Counter())
        group["cases"] += 1
        group["answerable"] += int(answerable)
        group["answered"] += int(answered)
        group["abstained"] += int(not answered)
        group["answerability_correct"] += int(answerable == answered)
        group["failed_cases"] += int(bool(failures))
        records.append({
            "question_id": row["question_id"], "question": row["question"], "category": category,
            "expected_answerable": answerable, "relevant_doc_ids": expected,
            "label_status": row.get("label_status", "pending_human_review"),
            "provenance": row.get("provenance", "Authored development case; not independently reviewed."),
            "expected_rationale": row.get("expected_rationale"),
            "status": answer.status, "abstention_reason": answer.abstention_reason,
            "answer": answer.answer,
            "retrieved_sources": [asdict(source) for source in answer.retrieved_sources],
            "cited_sources": [asdict(source) for source in answer.sources],
            "retrieval_top1_hit": top1 if answerable else None,
            "retrieval_topk_hit": topk if answerable else None,
            "failure_reasons": failures,
            "semantic_support_verified": False,
        })
    n = len(eval_set)
    metrics = {
        "retrieval_hit_at_1": ratio(counts["retrieval_top1"], counts["answerable"]),
        f"retrieval_hit_at_{top_k}": ratio(counts["retrieval_topk"], counts["answerable"]),
        f"retrieval_precision_at_{top_k}_on_answerable": ratio(counts["retrieved_relevant"], top_k * counts["answerable"]),
        "returned_candidate_relevance_precision_on_answerable": ratio(counts["retrieved_relevant"], counts["retrieved_total_on_answerable"]),
        "retrieval_source_presence": ratio(counts["retrieval_present"], n),
        "answered_source_presence": ratio(counts["answered_with_citations"], counts["answered"]),
        "citation_id_validity": ratio(counts["cited_valid"], counts["cited_total"]),
        "extractive_quote_traceability": ratio(counts["quotes_traceable"], counts["cited_total"]),
        "citation_relevance_precision": ratio(counts["cited_relevant"], counts["cited_total"]),
        "answerable_cases_with_relevant_citation": ratio(counts["answerable_with_relevant_citation"], counts["answerable"]),
        "answerability_accuracy": ratio(counts["tp_answerable"] + counts["tn_unanswerable"], n),
        "answerable_response_rate": ratio(counts["tp_answerable"], counts["answerable"]),
        "unanswerable_abstention_rate": ratio(counts["tn_unanswerable"], counts["unanswerable"]),
        "false_answer_rate_on_unanswerable": ratio(counts["fp_unanswerable"], counts["unanswerable"]),
        "unanswerable_with_retrieval_abstention_rate": ratio(counts["negative_with_retrieval_abstained"], counts["negative_with_retrieval"]),
    }
    # Timing is kept outside deterministic per-case records and hashes.
    return {
        "schema_version": 2,
        "documents_indexed": len(index.documents), "eval_questions": n,
        "retrieval_at_1": metrics["retrieval_hit_at_1"]["value"],
        f"retrieval_at_{top_k}": metrics[f"retrieval_hit_at_{top_k}"]["value"],
        "metrics": metrics,
        "answerability_confusion": {"answered_answerable": counts["tp_answerable"], "abstained_answerable": counts["fn_answerable"],
                                    "answered_unanswerable": counts["fp_unanswerable"], "abstained_unanswerable": counts["tn_unanswerable"]},
        "category_counts": {name: dict(group) for name, group in categories.items()},
        "failed_cases": counts["failed_cases"], "case_results": records,
        "latency": {"measurements": len(latencies), "avg_ms": statistics.mean(latencies) if latencies else None,
                    "p95_ms": _percentile(latencies, 0.95) if latencies else None, "scope": "local in-process extractive answer; no network or model generation"},
        "protocol": {
            "top_k": top_k, "evidence_policy": asdict(index.evidence_policy),
            "label_status_counts": dict(Counter(row.get("label_status", "pending_human_review") for row in eval_set)),
            "split": "development; no training or threshold tuning during this evaluation",
            "data_source": "authored synthetic ML/MLOps corpus and authored development cases",
            "semantic_support": "not measured: valid source IDs and exact extractive provenance do not verify that a passage answers the question",
            "citation_relevance": "compares cited IDs to provisional relevant_doc_ids; requires human review and does not measure semantic entailment",
            "compatibility": "schema 2 replaces misleading citation_coverage with explicitly defined source-presence, relevance, and traceability metrics",
        },
        "reproducibility": {
            "python": platform.python_version(),
            "corpus_sha256": _canonical_hash([{"id": doc.doc_id, "title": doc.title, "text": doc.text} for doc in index.documents]),
            "evaluation_cases_sha256": _canonical_hash(eval_set),
            "deterministic_case_results_sha256": _canonical_hash(records),
            "source_sha256_normalized_lf": {name: sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
                                           for name in ("rag_api/retriever.py", "rag_api/evaluation.py", "rag_api/corpus.py",
                                                        "scripts/build_challenge.py", "scripts/run_evaluation.py")},
        },
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
