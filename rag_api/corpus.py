from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    path: str


def default_corpus_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "corpus"


def load_documents(corpus_dir: Path | None = None) -> list[Document]:
    root = corpus_dir or default_corpus_dir()
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first_line = text.splitlines()[0].strip("# ").strip()
        documents.append(
            Document(
                doc_id=path.stem,
                title=first_line or path.stem.replace("_", " ").title(),
                text=text,
                path=str(path),
            )
        )

    if not documents:
        raise ValueError(f"No markdown documents found in {root}")
    return documents

