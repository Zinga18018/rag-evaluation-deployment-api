from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
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


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


class TfidfRagIndex:
    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
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
        query_vector = self._normalize(self._tfidf(tokenize(query)))
        scored: list[tuple[float, Document]] = []
        for doc in self.documents:
            scored.append((self._dot(query_vector, self.doc_vectors[doc.doc_id]), doc))

        results: list[SearchResult] = []
        for score, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]:
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    score=round(score, 4),
                    snippet=self._best_snippet(query, doc.text),
                )
            )
        return results

    def answer(self, query: str, top_k: int = 3) -> AnswerResult:
        start = time.perf_counter()
        sources = self.search(query, top_k=top_k)
        answer_parts = [source.snippet for source in sources if source.snippet and source.score >= 0.05]
        if not answer_parts and sources:
            answer_parts = [sources[0].snippet]
        if answer_parts:
            answer = " ".join(answer_parts[:2])
        else:
            answer = "No strong source match found in the indexed corpus."
        latency_ms = (time.perf_counter() - start) * 1000
        return AnswerResult(answer=answer, sources=sources, latency_ms=round(latency_ms, 3))

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
