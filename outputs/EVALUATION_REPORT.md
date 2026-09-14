# Extractive retrieval development evaluation

**32 documents; 52 cases.** All reference labels are pending human review. No model API was called.

| Metric | Count | Value |
|---|---:|---:|
| retrieval_hit_at_1 | 31/32 | 0.9688 |
| retrieval_hit_at_3 | 32/32 | 1.0000 |
| retrieval_precision_at_3_on_answerable | 32/96 | 0.3333 |
| returned_candidate_relevance_precision_on_answerable | 32/90 | 0.3556 |
| retrieval_source_presence | 44/52 | 0.8462 |
| answered_source_presence | 39/39 | 1.0000 |
| citation_id_validity | 46/46 | 1.0000 |
| extractive_quote_traceability | 46/46 | 1.0000 |
| citation_relevance_precision | 31/46 | 0.6739 |
| answerable_cases_with_relevant_citation | 31/32 | 0.9688 |
| answerability_accuracy | 43/52 | 0.8269 |
| answerable_response_rate | 31/32 | 0.9688 |
| unanswerable_abstention_rate | 12/20 | 0.6000 |
| false_answer_rate_on_unanswerable | 8/20 | 0.4000 |
| unanswerable_with_retrieval_abstention_rate | 4/12 | 0.3333 |

## Failure analysis

**16 cases have one or more diagnostic failures.** These are retained in [evaluation_failures.json](evaluation_failures.json), including exact queries, retrieved candidates, cited snippets, statuses, and provisional expected sources.

| Case | Category | Observed failure |
|---|---|---|
| q02 | original_answerable_fixture | cited_source_not_in_provisional_relevance_labels |
| q03 | original_answerable_fixture | cited_source_not_in_provisional_relevance_labels |
| q04 | original_answerable_fixture | cited_source_not_in_provisional_relevance_labels |
| q05 | original_answerable_fixture | abstained_on_answerable_case |
| q09 | original_answerable_fixture | cited_source_not_in_provisional_relevance_labels |
| q20 | original_answerable_fixture | cited_source_not_in_provisional_relevance_labels |
| dev-a04 | additional_answerable | cited_source_not_in_provisional_relevance_labels |
| dev-a08 | additional_answerable | cited_source_not_in_provisional_relevance_labels |
| dev-overlap01 | overlapping_vocabulary_unanswerable | answered_unanswerable_case, cited_source_not_in_provisional_relevance_labels |
| dev-overlap02 | overlapping_vocabulary_unanswerable | answered_unanswerable_case, cited_source_not_in_provisional_relevance_labels |
| dev-overlap05 | overlapping_vocabulary_unanswerable | answered_unanswerable_case, cited_source_not_in_provisional_relevance_labels |
| dev-overlap06 | overlapping_vocabulary_unanswerable | answered_unanswerable_case, cited_source_not_in_provisional_relevance_labels |
| dev-overlap07 | overlapping_vocabulary_unanswerable | answered_unanswerable_case, cited_source_not_in_provisional_relevance_labels |
| dev-overlap08 | overlapping_vocabulary_unanswerable | answered_unanswerable_case, cited_source_not_in_provisional_relevance_labels |
| dev-overlap09 | overlapping_vocabulary_unanswerable | answered_unanswerable_case, cited_source_not_in_provisional_relevance_labels |
| dev-overlap10 | overlapping_vocabulary_unanswerable | answered_unanswerable_case, cited_source_not_in_provisional_relevance_labels |

## Interpretation and limits

Retrieval hit rates use answerable cases only. Citation relevance compares used source IDs against provisional relevance labels. Source presence, valid IDs, and traceable extractive quotes do not establish that an answer addresses the question or that its claims are semantically supported.

The lexical abstention policy rejects zero-overlap and weak passages, but overlapping-vocabulary questions can still receive irrelevant answers. For example, a definition of API latency cannot answer a question about yesterday's measured latency. The per-case records preserve those failures.

These authored development cases are not an independent locked benchmark. No threshold search, model training, paid model request, or deployment was performed by this evaluation. Timing covers a local in-process extractor, not hosted RAG or LLM latency.

## Reproducibility

Run `python scripts/run_evaluation.py`. The [full report](evaluation_metrics.json) records corpus, normalized source, case-set, and deterministic result hashes; timing is excluded from the deterministic result hash.

Deterministic case result SHA-256: `ac3375fbf38f267e6164a72d4ec67f22e153691c1c1fa3c6b91200f78b2a8965`
