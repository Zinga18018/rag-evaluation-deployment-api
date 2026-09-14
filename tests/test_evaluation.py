import json

import pytest

from rag_api.corpus import Document, load_documents
from rag_api.evaluation import evaluate, load_eval_set
from rag_api.retriever import AnswerResult, EvidencePolicy, SearchResult, TfidfRagIndex


def test_zero_overlap_does_not_return_unrelated_sources_or_fallback():
    index = TfidfRagIndex(load_documents())
    assert index.search("zyxwv qwertyuiop") == []
    result = index.answer("zyxwv qwertyuiop")
    assert result.status == "abstained"
    assert result.sources == result.retrieved_sources == []
    assert result.abstention_reason == "no_matching_terms"


def test_weak_evidence_is_distinct_from_no_retrieval():
    index = TfidfRagIndex(load_documents())
    query = "LoRA " + " ".join(f"unknownword{n}" for n in range(30))
    result = index.answer(query)
    assert result.retrieved_sources
    assert result.status == "abstained"
    assert result.abstention_reason == "weak_lexical_evidence"
    assert result.sources == []


def test_cited_sources_are_exactly_the_snippets_used_in_answer():
    index = TfidfRagIndex(load_documents())
    result = index.answer("How does LoRA fine-tune a large language model efficiently?", top_k=5)
    assert result.status == "answered"
    assert 1 <= len(result.sources) <= 2
    assert result.answer == " ".join(f"{s.snippet} [{s.doc_id}]" for s in result.sources)
    assert {s.doc_id for s in result.sources} <= {s.doc_id for s in result.retrieved_sources}


def test_expanded_cases_have_explicit_provenance_and_negative_labels():
    rows = load_eval_set()
    assert len(rows) == 52
    assert len({row["question_id"] for row in rows}) == 52
    assert sum(row["answerable"] for row in rows) == 32
    assert sum(not row["answerable"] for row in rows) == 20
    assert all(row["label_status"] == "pending_human_review" for row in rows)
    assert sum(row["category"] == "overlapping_vocabulary_unanswerable" for row in rows) == 12
    assert all(not row["relevant_doc_ids"] for row in rows if not row["answerable"])


def test_irrelevant_source_presence_cannot_count_as_answer_quality(monkeypatch):
    source = Document("unrelated", "Caching", "Caching reduces repeated retrieval latency.", "fixture")
    index = TfidfRagIndex([source])
    hit = SearchResult(source.doc_id, source.title, 0.9, source.text)
    monkeypatch.setattr(index, "answer", lambda *args, **kwargs: AnswerResult(
        answer=f"{hit.snippet} [{hit.doc_id}]", sources=[hit], latency_ms=0,
        status="answered", retrieved_sources=[hit]))
    report = evaluate(index, [{"question_id": "irrelevant-source-negative", "question": "What was yesterday's measured latency?",
                              "answerable": False, "relevant_doc_ids": []}])
    metrics = report["metrics"]
    assert metrics["answered_source_presence"]["value"] == 1
    assert metrics["citation_id_validity"]["value"] == 1
    assert metrics["extractive_quote_traceability"]["value"] == 1
    assert metrics["citation_relevance_precision"]["value"] == 0
    assert metrics["answerability_accuracy"]["value"] == 0
    assert metrics["false_answer_rate_on_unanswerable"] == {"numerator": 1, "denominator": 1, "value": 1.0}
    assert report["case_results"][0]["semantic_support_verified"] is False
    assert "answered_unanswerable_case" in report["case_results"][0]["failure_reasons"]
    assert "citation_coverage" not in report


def test_retrieval_quality_uses_candidates_instead_of_only_citations():
    index = TfidfRagIndex(load_documents(), EvidencePolicy(min_query_term_coverage=1.0))
    row = {"question_id": "separate-retrieval", "question": "LoRA plus unknownword",
           "answerable": True, "relevant_doc_ids": ["lora_adaptation"]}
    report = evaluate(index, [row])
    assert report["metrics"]["retrieval_hit_at_3"]["value"] == 1
    assert report["case_results"][0]["status"] == "abstained"
    assert report["metrics"]["answerable_response_rate"]["value"] == 0


def test_all_negative_and_empty_sets_have_explicit_zero_denominators():
    index = TfidfRagIndex(load_documents())
    negative = {"question_id": "negative", "question": "zyxwv qwertyuiop", "answerable": False, "relevant_doc_ids": []}
    report = evaluate(index, [negative])
    assert report["metrics"]["retrieval_hit_at_1"] == {"numerator": 0, "denominator": 0, "value": None}
    assert report["metrics"]["unanswerable_abstention_rate"]["value"] == 1
    empty = evaluate(index, [])
    assert empty["eval_questions"] == 0
    assert all(metric["value"] is None for metric in empty["metrics"].values())
    assert empty["latency"]["avg_ms"] is None
    json.dumps(empty, allow_nan=False)


def test_case_hash_excludes_timing_and_is_reproducible():
    index = TfidfRagIndex(load_documents())
    cases = load_eval_set()
    first, second = evaluate(index, cases), evaluate(index, cases)
    assert first["case_results"] == second["case_results"]
    assert first["reproducibility"] == second["reproducibility"]
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["retrieval_hit_at_1"]["denominator"] == 32
    assert first["metrics"]["retrieval_precision_at_3_on_answerable"]["denominator"] == 96
    returned = sum(len(case["retrieved_sources"]) for case in first["case_results"] if case["expected_answerable"])
    assert first["metrics"]["returned_candidate_relevance_precision_on_answerable"]["denominator"] == returned
    assert first["metrics"]["unanswerable_abstention_rate"]["denominator"] == 20


def test_overlapping_vocabulary_failures_are_retained():
    index = TfidfRagIndex(load_documents())
    report = evaluate(index, load_eval_set())
    negatives = [case for case in report["case_results"] if not case["expected_answerable"] and case["retrieved_sources"]]
    assert negatives
    for case in negatives:
        assert ("answered_unanswerable_case" in case["failure_reasons"]) == (case["status"] == "answered")
    assert report["metrics"]["unanswerable_with_retrieval_abstention_rate"]["denominator"] == len(negatives)


@pytest.mark.parametrize("case", [
    {"question_id": "unknown", "question": "example", "answerable": True, "relevant_doc_ids": ["missing"]},
    {"question_id": "invalid", "question": "example", "answerable": False, "relevant_doc_ids": ["lora_adaptation"]},
])
def test_invalid_reference_labels_raise(case):
    with pytest.raises(ValueError):
        evaluate(TfidfRagIndex(load_documents()), [case])


def test_duplicate_case_ids_raise():
    row = {"question_id": "same", "question": "no answer", "answerable": False, "relevant_doc_ids": []}
    with pytest.raises(ValueError):
        evaluate(TfidfRagIndex([]), [row, row])


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_raises(top_k):
    with pytest.raises(ValueError):
        TfidfRagIndex([]).search("example", top_k)


def test_empty_index_abstains():
    result = TfidfRagIndex([]).answer("What is missing?")
    assert result.status == "abstained"
    assert result.sources == []


def test_duplicate_document_ids_raise():
    document = Document("duplicate", "Title", "Text", "fixture")
    with pytest.raises(ValueError):
        TfidfRagIndex([document, document])


@pytest.mark.parametrize("policy", [{"min_score": float("nan")}, {"min_query_term_coverage": 1.5}, {"max_cited_sources": 0}])
def test_invalid_policy_raises(policy):
    with pytest.raises(ValueError):
        EvidencePolicy(**policy)
