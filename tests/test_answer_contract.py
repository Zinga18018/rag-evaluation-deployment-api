from fastapi.testclient import TestClient

from rag_api.app import app


client = TestClient(app)


def test_answer_preserves_legacy_fields_and_adds_explicit_status():
    response = client.post("/answer", json={"question": "How does LoRA fine-tune a large language model efficiently?", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert {"question", "answer", "latency_ms", "sources"} <= set(payload)
    assert payload["status"] == "answered"
    assert payload["abstention_reason"] is None
    assert payload["retrieved_sources"]
    assert all(f"[{s['doc_id']}]" in payload["answer"] for s in payload["sources"])


def test_unrelated_request_returns_explicit_abstention():
    response = client.post("/answer", json={"question": "zyxwv qwertyuiop"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "abstained"
    assert payload["sources"] == payload["retrieved_sources"] == []


def test_search_does_not_pad_with_zero_score_results():
    response = client.get("/search", params={"q": "zyxwv qwertyuiop", "top_k": 5})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_invalid_api_top_k_keeps_validation():
    assert client.post("/answer", json={"question": "LoRA details", "top_k": 0}).status_code == 422
    assert client.get("/search", params={"q": "LoRA", "top_k": 6}).status_code == 422


def test_metrics_route_exposes_denominators_and_provenance():
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 2
    assert payload["metrics"]["retrieval_hit_at_1"]["denominator"] == 32
    assert payload["metrics"]["false_answer_rate_on_unanswerable"]["denominator"] == 20
    assert payload["protocol"]["label_status_counts"] == {"pending_human_review": 52}
