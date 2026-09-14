"""Author reproducible development fixtures, explicitly pending human review."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSWERABLE = [
    ("What should evaluation datasets include for repeatable scoring?", "evaluation_dataset"),
    ("Which checks can GitHub Actions run on every pull request?", "github_actions"),
    ("What should unit tests validate before code reaches production?", "unit_tests"),
    ("Which debugging fields are captured in structured logs?", "structured_logging"),
    ("Why cache repeated retrieval results for common questions?", "cache_layer"),
    ("How does document chunking make long files retrievable?", "document_chunking"),
    ("What does faithfulness evaluation check about generated answers?", "faithfulness_eval"),
    ("What belongs in a cloud deployment runbook?", "cloud_runbook"),
    ("How does data lineage track datasets and versions?", "data_lineage"),
    ("Why apply controlled schema migrations to a database?", "schema_migration"),
    ("How does rate limiting protect public APIs?", "rate_limiting"),
    ("Which failures need ML incident response triage?", "incident_response"),
]
OVERLAP_NEGATIVES = [
    ("What exact p95 latency did this model API achieve yesterday?", "The latency note defines measurements but provides no dated observations."),
    ("What dollar cost did SLURM GPU training incur last month?", "The SLURM note explains scheduling but contains no expenditure records."),
    ("What learning rate should I use for this LoRA fine-tuning job?", "The LoRA note describes adapters without a task-specific learning rate."),
    ("What was the measured macro-F1 score on our latest classification test?", "The metric definition contains no experiment results."),
    ("Which rollback command restores our Dockerized FastAPI service?", "Deployment notes list concepts without this service's rollback command."),
    ("What is the exact redaction accuracy of the PII redaction system?", "The PII note lists removed information but gives no accuracy evaluation."),
    ("Which database vendor stores our approved model versions?", "The registry note does not name a database vendor."),
    ("What measured conversion rate did treatment achieve in the A/B test?", "The test-readout note describes metrics but supplies no trial outcomes."),
    ("What dataset row count was written by yesterday's batch inference?", "The batch note specifies validation, not a recorded run's row count."),
    ("Which names and email addresses appeared in the original unredacted text?", "The redaction note has no underlying records or identifying values."),
    ("What is the optimal document chunking size in tokens for my private corpus?", "The chunking note has no corpus-specific optimum or token count."),
    ("What was our observed 30-day cohort retention percentage?", "The cohort note defines retention periods without observed percentages."),
]
OUT_OF_DOMAIN = [
    "Who painted the Mona Lisa?",
    "How many moons orbit Jupiter?",
    "What ingredients make a traditional biryani?",
    "Who won Wimbledon in 2025?",
    "What is the capital city of Mongolia?",
    "How tall is Mount Everest?",
    "When was the Taj Mahal completed?",
    "What is the chemical formula of caffeine?",
]


def main():
    rows = []
    for number, (question, expected) in enumerate(ANSWERABLE, 1):
        rows.append({"question_id": f"dev-a{number:02d}", "question": question,
                     "answerable": True, "relevant_doc_ids": [expected],
                     "category": "additional_answerable", "expected_rationale": "The named corpus document states the requested definition or mechanism."})
    for number, (question, rationale) in enumerate(OVERLAP_NEGATIVES, 1):
        rows.append({"question_id": f"dev-overlap{number:02d}", "question": question,
                     "answerable": False, "relevant_doc_ids": [],
                     "category": "overlapping_vocabulary_unanswerable", "expected_rationale": rationale})
    for number, question in enumerate(OUT_OF_DOMAIN, 1):
        rows.append({"question_id": f"dev-ood{number:02d}", "question": question,
                     "answerable": False, "relevant_doc_ids": [],
                     "category": "out_of_domain_unanswerable", "expected_rationale": "The indexed ML/MLOps corpus contains no answer to this question."})
    for row in rows:
        row.update({"label_status": "pending_human_review", "dataset_version": "development-challenge-v1",
                    "provenance": "Authored development fixture based on the included synthetic corpus; not human-reviewed, not locked evaluation, and not a model-generated reference answer."})
    path = ROOT / "data" / "eval_challenge.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} pending_human_review challenge cases to {path}")


if __name__ == "__main__":
    main()
