from fastapi.testclient import TestClient

from rag_api.app import app, index


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["documents_indexed"] >= 30


def test_search_finds_expected_doc() -> None:
    results = index.search("What does Retrieval@K measure in RAG evaluation?", top_k=3)
    assert results[0].doc_id == "retrieval_at_k"


def test_answer_returns_citations() -> None:
    response = client.post(
        "/answer",
        json={"question": "How does LoRA fine-tune a large language model efficiently?", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"]
    assert payload["sources"][0]["doc_id"] == "lora_adaptation"
    assert payload["latency_ms"] >= 0

