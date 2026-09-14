from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from rag_api.corpus import Document

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "with", "what", "which", "how", "when", "where", "why",
}


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    score: float
    snippet: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[SearchResult]
    latency_ms: float
    status: str = "answered"
    abstention_reason: str | None = None
    retrieved_sources: list[SearchResult] = field(default_factory=list)


@dataclass(frozen=True)
class EvidencePolicy:
    """Lexical screening only; these thresholds are not a semantic verifier."""
    min_score: float = 0.10
    min_query_term_coverage: float = 0.30
    min_shared_terms: int = 2
    max_cited_sources: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be finite and in [0, 1]")
        if not math.isfinite(self.min_query_term_coverage) or not 0 <= self.min_query_term_coverage <= 1:
            raise ValueError("min_query_term_coverage must be finite and in [0, 1]")
        if self.min_shared_terms < 1 or self.max_cited_sources < 1:
            raise ValueError("shared-term and cited-source limits must be positive")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


class TfidfRagIndex:
    def __init__(self, documents: Iterable[Document], evidence_policy: EvidencePolicy | None = None) -> None:
        self.documents = list(documents)
        if len({doc.doc_id for doc in self.documents}) != len(self.documents):
            raise ValueError("document IDs must be unique")
        self.evidence_policy = evidence_policy or EvidencePolicy()
        self.doc_vectors: dict[str, dict[str, float]] = {}
        self.idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        doc_tokens = {doc.doc_id: tokenize(f"{doc.title} {doc.text}") for doc in self.documents}
        document_frequency: dict[str, int] = {}
        for tokens in doc_tokens.values():
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1

        total_docs = len(self.documents)
        self.idf = {
            term: math.log((1 + total_docs) / (1 + count)) + 1
            for term, count in document_frequency.items()
        }
        self.doc_vectors = {
            doc_id: self._normalize(self._tfidf(tokens))
            for doc_id, tokens in doc_tokens.items()
        }

    def _tfidf(self, tokens: list[str]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        total = max(len(tokens), 1)
        return {term: (count / total) * self.idf.get(term, 0.0) for term, count in counts.items()}

    @staticmethod
    def _normalize(vector: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return vector
        return {term: value / norm for term, value in vector.items()}

    @staticmethod
    def _dot(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(term, 0.0) for term, value in left.items())

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query_vector = self._normalize(self._tfidf(tokenize(query)))
        scored: list[tuple[float, Document]] = []
        for doc in self.documents:
            score = self._dot(query_vector, self.doc_vectors[doc.doc_id])
            if score > 0:
                scored.append((score, doc))

        results: list[SearchResult] = []
        for score, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]:
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    score=float(score),
                    snippet=self._best_snippet(query, doc.text),
                )
            )
        return results

    def answer(self, query: str, top_k: int = 3) -> AnswerResult:
        start = time.perf_counter()
        retrieved = self.search(query, top_k=top_k)
        query_terms = set(tokenize(query))
        policy = self.evidence_policy
        sources = []
        for source in retrieved:
            shared = len(query_terms.intersection(tokenize(source.snippet)))
            coverage = shared / len(query_terms) if query_terms else 0.0
            if (source.snippet.strip() and source.score >= policy.min_score
                    and coverage >= policy.min_query_term_coverage
                    and shared >= min(policy.min_shared_terms, len(query_terms))):
                sources.append(source)
        sources = sources[:policy.max_cited_sources]
        if sources:
            answer = " ".join(f"{source.snippet} [{source.doc_id}]" for source in sources)
            status, reason = "answered", None
        else:
            answer = "I cannot answer from the indexed corpus: no passage passed the lexical evidence checks."
            status = "abstained"
            reason = "no_matching_terms" if not retrieved else "weak_lexical_evidence"
        latency_ms = (time.perf_counter() - start) * 1000
        return AnswerResult(answer=answer, sources=sources, latency_ms=round(latency_ms, 3),
                            status=status, abstention_reason=reason, retrieved_sources=retrieved)

    @staticmethod
    def _best_snippet(query: str, text: str) -> str:
        query_terms = set(tokenize(query))
        sentences = [s.strip() for s in SENTENCE_RE.split(text.replace("\n", " ")) if s.strip()]
        if not sentences:
            return text[:240]

        def score(sentence: str) -> int:
            return len(query_terms.intersection(tokenize(sentence)))

        best = max(sentences, key=score)
        return best[:260]
