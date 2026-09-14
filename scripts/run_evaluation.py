from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_api.evaluation import build_default_report


def main() -> None:
    report = build_default_report()
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    failures = [case for case in report["case_results"] if case["failure_reasons"]]
    (output / "evaluation_failures.json").write_text(json.dumps(failures, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Extractive retrieval development evaluation", "",
             f"**{report['documents_indexed']} documents; {report['eval_questions']} cases.** All reference labels are pending human review. No model API was called.",
             "", "| Metric | Count | Value |", "|---|---:|---:|"]
    for name, metric in report["metrics"].items():
        value = "unavailable" if metric["value"] is None else f"{metric['value']:.4f}"
        lines.append(f"| {name} | {metric['numerator']}/{metric['denominator']} | {value} |")
    lines += ["", "## Failure analysis", "",
              f"**{len(failures)} cases have one or more diagnostic failures.** These are retained in [evaluation_failures.json](evaluation_failures.json), including exact queries, retrieved candidates, cited snippets, statuses, and provisional expected sources.",
              "", "| Case | Category | Observed failure |", "|---|---|---|"]
    for case in failures:
        lines.append(f"| {case['question_id']} | {case['category']} | {', '.join(case['failure_reasons'])} |")
    lines += ["", "## Interpretation and limits", "",
              "Retrieval hit rates use answerable cases only. Citation relevance compares used source IDs against provisional relevance labels. Source presence, valid IDs, and traceable extractive quotes do not establish that an answer addresses the question or that its claims are semantically supported.",
              "", "The lexical abstention policy rejects zero-overlap and weak passages, but overlapping-vocabulary questions can still receive irrelevant answers. For example, a definition of API latency cannot answer a question about yesterday's measured latency. The per-case records preserve those failures.",
              "", "These authored development cases are not an independent locked benchmark. No threshold search, model training, paid model request, or deployment was performed by this evaluation. Timing covers a local in-process extractor, not hosted RAG or LLM latency.",
              "", "## Reproducibility", "", "Run `python scripts/run_evaluation.py`. The [full report](evaluation_metrics.json) records corpus, normalized source, case-set, and deterministic result hashes; timing is excluded from the deterministic result hash.",
              "", f"Deterministic case result SHA-256: `{report['reproducibility']['deterministic_case_results_sha256']}`", ""]
    (output / "EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"eval_questions": report["eval_questions"], "metrics": report["metrics"],
                      "failed_cases": len(failures), "deterministic_result_sha256": report["reproducibility"]["deterministic_case_results_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
