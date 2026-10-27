"""Plot the checked-in development snapshot; never rerun or alter its evaluation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "evaluation_metrics.json"
SOURCE_COMMIT = "be7221e27ff3b0d2d719a81729c0934ae87cde86"
SOURCE_BLOB = "47ba5d96b7878ec8219a35f7160f3a890801a19b"
SOURCE_SHA256 = "5a6c699a80a408ec1511c427f63a474f8b4f8264373dcb5c1e728f624f0bddb4"
STEM = "development-answer-status"
CATEGORIES = (
    ("original_answerable_fixture", "Original answerable fixtures"),
    ("additional_answerable", "Additional answerable cases"),
    ("overlapping_vocabulary_unanswerable", "Overlapping-vocabulary negatives"),
    ("out_of_domain_unanswerable", "Out-of-domain negatives"),
)


def load_snapshot(source: Path = SOURCE) -> dict:
    """Fail closed if the historical evidence changed; do not silently relabel it."""
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("evaluation snapshot hash changed; review provenance before updating the figure")
    report = json.loads(raw)
    if report["schema_version"] != 2 or report["eval_questions"] != 52:
        raise ValueError("expected the schema-2, 52-case development snapshot")
    if report["protocol"]["label_status_counts"] != {"pending_human_review": 52}:
        raise ValueError("the figure requires all labels to remain pending human review")
    if set(report["category_counts"]) != {key for key, _ in CATEGORIES}:
        raise ValueError("unexpected development categories")
    for key, _ in CATEGORIES:
        group = report["category_counts"][key]
        if group["answered"] + group["abstained"] != group["cases"]:
            raise ValueError("answer-status counts do not sum to category size")
    if sum(group["cases"] for group in report["category_counts"].values()) != 52:
        raise ValueError("category counts do not sum to 52")
    return report


def provenance(report: dict) -> dict:
    return {
        "source_path": "outputs/evaluation_metrics.json",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": SOURCE_BLOB,
        "source_sha256": SOURCE_SHA256,
        "deterministic_case_results_sha256": report["reproducibility"]["deterministic_case_results_sha256"],
        "category_counts": {key: report["category_counts"][key] for key, _ in CATEGORIES},
        "label_status_counts": report["protocol"]["label_status_counts"],
        "interpretation": "Answer emission is not answer correctness. Answers on provisionally unanswerable cases are false answers under those provisional labels.",
        "limits": "Authored synthetic development fixtures; no human-reviewed gold labels, held-out benchmark, semantic correctness measurement, or live-model evaluation.",
        "generator": "scripts/plot_development_results.py",
        "plot_dependency": "matplotlib==3.10.8 (optional, see requirements-plot.txt)",
        "reproduction": "python scripts/plot_development_results.py",
        "determinism": "Fixed SVG identifiers and no timestamp; byte-repeatability checked in the same Python/Matplotlib/font environment. Cross-platform byte identity is not promised.",
    }


def render(output_dir: Path) -> None:
    report = load_snapshot()
    # Keep the plotting dependency out of the API, evaluator and source-data checks.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "svg.hashsalt": "rag-development-answer-status-v1",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": "#888888",
        "text.color": "#202020", "axes.labelcolor": "#202020",
        "xtick.color": "#454545", "ytick.color": "#454545",
        "figure.facecolor": "white", "axes.facecolor": "white",
    }
    with plt.rc_context(settings):
        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=144)
        fig.subplots_adjust(left=0.39, right=0.97, top=0.78, bottom=0.17)
        answered_color, abstained_color = "#24618D", "#D9DEE3"
        positions = [0, 1, 2, 3]
        labels = ["Original answerable\nfixtures (n = 20)",
                  "Additional answerable\ncases (n = 12)",
                  "Overlapping-vocabulary\nunanswerable cases\n(8 false answers; n = 12)",
                  "Out-of-domain unanswerable\ncases (n = 8)"]
        for position, (key, label) in zip(positions, CATEGORIES):
            group = report["category_counts"][key]
            answered, abstained = group["answered"], group["abstained"]
            ax.barh(position, answered, height=0.55, color=answered_color, zorder=3)
            ax.barh(position, abstained, left=answered, height=0.55,
                    color=abstained_color, edgecolor="#8C969F", linewidth=0.6, zorder=3)
            if answered:
                ax.text(answered / 2, position, str(answered), color="white",
                        ha="center", va="center", fontsize=11, zorder=4)
            if abstained:
                ax.text(answered + abstained / 2, position, str(abstained),
                        ha="center", va="center", fontsize=11, zorder=4)
            if not answered or not abstained:
                zero = "0 answered" if not answered else "0 abstained"
                ax.text(group["cases"] + 0.4, position, zero,
                        va="center", fontsize=10, color="#454545")
        ax.set(yticks=positions, yticklabels=labels, xlim=(0, 21), ylim=(3.6, -0.6),
               xticks=[0, 5, 10, 15, 20], xlabel="Number of development cases")
        ax.tick_params(axis="x", labelsize=11, length=3)
        ax.tick_params(axis="y", labelsize=10.5, length=0, pad=8)
        ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        fig.suptitle("Answer status: 52 provisional development cases", y=0.96,
                     fontsize=12.5, weight="normal")
        ax.legend(handles=[Patch(facecolor=answered_color, label="Answered"),
                           Patch(facecolor=abstained_color, edgecolor="#8C969F", label="Abstained")],
                  loc="lower center", bbox_to_anchor=(0.5, 1.015), frameon=False,
                  ncol=2, fontsize=10.5, handlelength=1.2, columnspacing=1.7)
        title = "Answer emission and abstention across 52 provisional development cases"
        description = ("Original answerable: 19 answered, 1 abstained. Additional answerable: 12 answered, 0 abstained. "
                       "Overlapping-vocabulary unanswerable: 8 false answers, 4 abstentions. "
                       "Out-of-domain unanswerable: 0 answered, 8 abstained. "
                       "All labels are pending human review; answer emission is not answer correctness.")
        fig.savefig(output_dir / f"{STEM}.svg", metadata={
            "Title": title, "Description": description,
            "Creator": "scripts/plot_development_results.py", "Date": None})
        fig.savefig(output_dir / f"{STEM}.png", metadata={
            "Title": title, "Description": description,
            "Software": "scripts/plot_development_results.py"})
        plt.close(fig)
    (output_dir / f"{STEM}.provenance.json").write_text(
        json.dumps(provenance(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
