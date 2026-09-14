# RAG Evaluation and Deployment API

**A small retrieval API that makes missing evidence and misleading evaluation metrics visible.**

The service indexes 32 authored ML/MLOps notes with pure-Python TF-IDF. It returns ranked passages, constructs short extractive answers with source markers, and abstains when no passage passes its lexical checks. It runs without a model download, GPU, or paid API key.

[Workflow and code map](WORKFLOW.md) · [Development evaluation](outputs/EVALUATION_REPORT.md) · [Per-case failures](outputs/evaluation_failures.json)

## What changed

The earlier version returned zero-score documents for unrelated queries, then used the first snippet as a fallback answer. Its `citation_coverage` metric counted any nonempty source list as a success. That could report perfect coverage even when the answer was irrelevant.

The current implementation:

- Returns only positive-score retrieval candidates.
- Uses explicit `answered` or `abstained` status and an abstention reason.
- Keeps all ranked candidates in `retrieved_sources`; `sources` contains only snippets actually used in the answer, with matching `[document_id]` markers.
- Reports retrieval relevance, source presence, citation-ID validity, extractive traceability, and answerability separately, with numerator and denominator for every rate.
- Evaluates both answerable and unanswerable development cases, including questions that share vocabulary with the corpus but ask for absent facts.

This is an extractive retrieval baseline. It does not call an LLM or verify semantic entailment.

## Evidence policy

A passage must have cosine similarity >= 0.10, cover >= 30% of distinct query terms, and share at least two terms, or all terms for a one-term query. At most two passages are quoted. The defaults are simple development heuristics; no threshold search was performed on the reported cases.

This rejects unrelated queries and some weak matches. It still confuses topical similarity with answerability in several cases. A note defining API latency does not contain yesterday's measured latency, even though both questions use the same vocabulary.

## Development evaluation

The suite contains **52 cases**: 20 original answerable fixtures, 12 additional answerable cases, 12 overlapping-vocabulary unanswerable cases, and 8 out-of-domain unanswerable cases. All labels are **`pending_human_review`**. These are authored development fixtures; they are not human-reviewed gold labels, production questions, or a locked benchmark.

| Metric | Observed count | Meaning |
|---|---:|---|
| Retrieval hit at 1 | 31/32 | First candidate matches a provisionally relevant source on answerable cases |
| Retrieval hit at 3 | 32/32 | A labeled relevant source appears among the top three |
| Answerable cases receiving an answer | 31/32 | One answerable case is incorrectly refused |
| Correct abstentions on unanswerable cases | 12/20 | Eight unanswerable cases still receive answers |
| Abstentions when irrelevant candidates are present | 4/12 | Harder than simply detecting zero overlap |
| Citation relevance precision | 31/46 | Used citations match provisional relevant-source labels |
| Traceable extractive citations | 46/46 | Quoted snippets and markers can be traced to their named documents |

**Traceability is not answer correctness:** every used citation is traceable, yet **8/20 unanswerable questions are falsely answered**. The report preserves **16 cases with one or more diagnostics**, including unnecessary citations and one incorrect abstention. It does not hide failures behind the successful retrieval hit rate.

Precision@3 uses three slots per answerable case, including unfilled slots: 32/96. A separately named returned-candidate relevance metric uses actual candidates: 32/90. Empty denominators are `null`, not fabricated zeros or perfect scores.

## API contract

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and indexed-document count |
| `GET /search?q=...&top_k=3` | Positive-score ranked candidates; may return fewer than K |
| `POST /answer` | Extractive answer or explicit abstention |
| `GET /metrics` | Recompute the development evaluation, including per-case records |

`POST /answer` accepts the existing `question` and `top_k` fields. Existing response keys `question`, `answer`, `latency_ms`, and `sources` remain. Added fields are `status`, `abstention_reason`, and `retrieved_sources`. Clients that used `sources` as the complete ranking should switch to `retrieved_sources`.

The metrics response is **schema version 2**. The misleading `citation_coverage` field has been removed; use the explicitly named metrics. Citation-ID validity means the document exists, and quote traceability means the text was copied from it. Neither is semantic-support verification.

## Run locally

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m uvicorn rag_api.app:app --host 127.0.0.1 --port 8000
```

API documentation: `http://127.0.0.1:8000/docs`.

```powershell
.venv/Scripts/python.exe -m pytest -q --junitxml outputs/test_results.xml
.venv/Scripts/python.exe scripts/run_evaluation.py
```

The included corpus and cases are ready to use. `scripts/seed_corpus.py` regenerates the original 32 notes and 20 fixtures; `scripts/build_challenge.py` regenerates the additional 32 cases with their provenance.

## Verification and reproducibility

Local verification: **29 tests passed**, including API compatibility, abstention, exact source usage, irrelevant-source-present negatives, missing denominators, invalid labels, and deterministic results. Machine-readable results are in [outputs/test_results.xml](outputs/test_results.xml).

The evaluation writes:

- [evaluation_metrics.json](outputs/evaluation_metrics.json): metrics, confusion counts, all case records, policy, hashes, and timing scope.
- [evaluation_failures.json](outputs/evaluation_failures.json): every case with a diagnostic failure.
- [EVALUATION_REPORT.md](outputs/EVALUATION_REPORT.md): readable results and limitations.

Corpus, case-set, source-code, and deterministic-result hashes support reproducing the run. Timing is measured separately and excluded from the deterministic case hash. Latency describes local in-process retrieval and extraction; it is not hosted service or language-model latency.

## Scope and next evaluation step

Docker, Vercel, and Render configuration files remain available, but this verification did not build a container or deploy a service. The mirrored retriever and evaluator in the LLM Evaluation Workbench preserve that project's separate judge, replay, provider, and UI features. Retrieval metrics are not judge-quality metrics.

The next useful step is to review the provisional source/answerability labels, define a separate calibration set, and compare a semantic answerability checker against this lexical baseline. Real factual-support or LLM quality claims need their own reviewed references and evaluation protocol.

A defensible portfolio description:

> Built a FastAPI retrieval baseline with explicit abstention, traceable extractive citations, and reproducible development evaluation. Expanded testing to 52 answerable and unanswerable cases and separated retrieval success, source relevance, and false-answer behavior instead of treating source presence as correctness.
