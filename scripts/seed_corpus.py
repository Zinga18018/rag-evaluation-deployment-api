from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
EVAL_PATH = ROOT / "data" / "eval_set.json"

TOPICS = [
    ("mlflow_tracking", "MLflow Experiment Tracking", "MLflow records parameters, metrics, artifacts, and model versions so machine learning experiments can be reproduced and compared."),
    ("docker_fastapi", "Dockerized FastAPI Deployment", "Docker packages a FastAPI model service with pinned dependencies, a health check, and a stable uvicorn startup command for cloud deployment."),
    ("rag_citations", "RAG Citation Policy", "Retrieval augmented generation should return source citations, document identifiers, and quoted evidence so users can audit generated answers."),
    ("vector_search", "Vector Search Retrieval", "Vector search ranks documents by embedding or sparse-vector similarity and returns top-k context passages for answer generation."),
    ("latency_monitoring", "API Latency Monitoring", "Latency monitoring tracks average, p95, and p99 response time so deployed model APIs can be debugged before users notice slowdowns."),
    ("data_drift", "Data Drift Detection", "Data drift checks compare live feature distributions with training data distributions to detect when a model may become unreliable."),
    ("model_registry", "Model Registry Governance", "A model registry stores approved model versions, signatures, metrics, owners, and deployment stages for controlled releases."),
    ("batch_inference", "Batch Inference Pipeline", "Batch inference scores scheduled datasets, writes predictions to storage, and validates row counts before downstream use."),
    ("online_inference", "Online Inference API", "Online inference serves predictions through low-latency endpoints with request validation, error handling, and structured logs."),
    ("feature_validation", "Feature Validation Rules", "Feature validation enforces schema, ranges, null thresholds, and categorical value constraints before model inference."),
    ("ab_testing", "A/B Test Readout", "A/B test readouts compare control and treatment conversion rates, confidence intervals, p-values, and rollout recommendations."),
    ("cohort_retention", "Cohort Retention Metrics", "Cohort retention groups users by start period and measures whether they return after 7, 14, or 30 days."),
    ("slurm_training", "SLURM Training Jobs", "SLURM schedules research-compute jobs, allocates CPU or GPU resources, and captures logs for long-running model training."),
    ("lora_adaptation", "LoRA Adapter Fine-Tuning", "LoRA fine-tuning trains low-rank adapter weights for large language models while keeping most base model parameters frozen."),
    ("macro_f1", "Macro-F1 Evaluation", "Macro-F1 averages class-level F1 scores and is useful when labels are imbalanced across stance, sentiment, or classification tasks."),
    ("mcc_metric", "Matthews Correlation Coefficient", "MCC measures classification quality using true and false positives and negatives, staying informative under label imbalance."),
    ("high_confidence_error", "High-Confidence Error Auditing", "High-confidence error auditing reviews cases where a model is confident but wrong to identify systematic failure modes."),
    ("prompt_injection", "Prompt Injection Defense", "Prompt injection defense separates system instructions from retrieved content and filters untrusted instructions before generation."),
    ("pii_redaction", "PII Redaction", "PII redaction removes names, emails, phone numbers, addresses, and identifiers before text is logged or sent to a model."),
    ("evaluation_dataset", "Evaluation Dataset Design", "Evaluation datasets should include expected answers, expected source documents, negative cases, and stable IDs for repeatable scoring."),
    ("github_actions", "GitHub Actions CI", "GitHub Actions can run tests, linting, and smoke checks on every pull request before deployment."),
    ("unit_tests", "Unit Test Coverage", "Unit tests validate retrieval ranking, API response shape, schema validation, and edge cases before code reaches production."),
    ("structured_logging", "Structured Logging", "Structured logs capture request IDs, latency, endpoint names, and error types for debugging deployed ML services."),
    ("cache_layer", "Caching Layer", "Caching repeated retrieval results reduces latency and cost for common questions in high-traffic AI applications."),
    ("document_chunking", "Document Chunking", "Document chunking splits long files into retrievable passages while preserving enough context for source-grounded answers."),
    ("retrieval_at_k", "Retrieval@K Metric", "Retrieval@K measures whether the expected source document appears within the top K retrieved documents for a question. Retrieval at K is the main metric for checking if a RAG retriever found the right evidence."),
    ("faithfulness_eval", "Answer Faithfulness Evaluation", "Faithfulness evaluation checks whether generated answers are supported by retrieved evidence rather than unsupported model guesses."),
    ("cloud_runbook", "Cloud Deployment Runbook", "A deployment runbook documents build commands, environment variables, rollback steps, and health checks for cloud services."),
    ("data_lineage", "Data Lineage", "Data lineage records where datasets came from, how they changed, and which model or report used each version."),
    ("schema_migration", "Schema Migration", "Schema migrations apply controlled database changes and prevent breaking downstream analytics or inference jobs."),
    ("rate_limiting", "API Rate Limiting", "Rate limiting protects public APIs from abusive traffic and keeps inference costs predictable."),
    ("incident_response", "ML Incident Response", "ML incident response defines triage steps for bad predictions, latency spikes, data drift, and failed deployments."),
]

EVAL_QUESTIONS = [
    ("How should experiments be tracked for reproducible machine learning?", "mlflow_tracking"),
    ("What should a Dockerized FastAPI deployment include?", "docker_fastapi"),
    ("Why should a RAG answer include citations?", "rag_citations"),
    ("How does vector search choose context passages?", "vector_search"),
    ("Which metric helps detect slow model API responses?", "latency_monitoring"),
    ("How can teams detect unreliable live feature distributions?", "data_drift"),
    ("Where should approved model versions and deployment stages be stored?", "model_registry"),
    ("What does batch inference validate before downstream use?", "batch_inference"),
    ("What is online inference for low latency predictions?", "online_inference"),
    ("Which rules catch bad model input features?", "feature_validation"),
    ("What does an A/B test readout compare?", "ab_testing"),
    ("How is cohort retention measured after 7 or 30 days?", "cohort_retention"),
    ("What schedules long-running GPU training jobs?", "slurm_training"),
    ("How does LoRA fine-tune a large language model efficiently?", "lora_adaptation"),
    ("Why is macro-F1 useful for imbalanced classification labels?", "macro_f1"),
    ("Which metric stays informative under label imbalance?", "mcc_metric"),
    ("What audits confident but wrong model predictions?", "high_confidence_error"),
    ("How do you defend a RAG system from prompt injection?", "prompt_injection"),
    ("What should be removed before text is logged or sent to a model?", "pii_redaction"),
    ("What does Retrieval@K measure in RAG evaluation?", "retrieval_at_k"),
]


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id, title, body in TOPICS:
        text = (
            f"# {title}\n\n"
            f"{body}\n\n"
            f"Operational note: {title} supports reliable ML engineering workflows by making model behavior easier to test, deploy, monitor, and explain.\n"
        )
        (CORPUS_DIR / f"{doc_id}.md").write_text(text, encoding="utf-8")

    eval_rows = [
        {"question_id": f"q{i + 1:02d}", "question": question, "expected_doc_id": expected}
        for i, (question, expected) in enumerate(EVAL_QUESTIONS)
    ]
    EVAL_PATH.write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")
    print(f"Seeded {len(TOPICS)} documents and {len(eval_rows)} evaluation questions.")


if __name__ == "__main__":
    main()
