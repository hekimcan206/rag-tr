"""Okapi BM25 and reciprocal rank fusion.

BM25 uses the Lucene/Elasticsearch variant of IDF, ``ln(1 + (N - df + 0.5) / (df + 0.5))``, which is never
negative (the classic Robertson-Spärck Jones IDF goes negative for terms in more than half the documents).
The PostgreSQL store computes the *same* formula in SQL; ``tests/integration`` asserts equal scores.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

K1 = 1.2
B = 0.75


def bm25_idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_term(tf: int, doc_len: int, avg_len: float, idf: float, k1: float = K1, b: float = B) -> float:
    return idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * doc_len / avg_len))


class BM25Index:
    """In-memory inverted index. ``search`` is exact BM25 over all documents containing a query term."""

    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1, self.b = k1, b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}
        self._doc_terms: dict[str, list[str]] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._lengths)

    def add(self, doc_id: str, terms: Mapping[str, int]) -> None:
        if doc_id in self._lengths:
            self.remove(doc_id)
        for term, tf in terms.items():
            self._postings[term][doc_id] = tf
        self._doc_terms[doc_id] = list(terms)
        length = sum(terms.values())
        self._lengths[doc_id] = length
        self._total_len += length

    def remove(self, doc_id: str) -> None:
        length = self._lengths.pop(doc_id, None)
        if length is None:
            return
        self._total_len -= length
        for term in self._doc_terms.pop(doc_id, []):
            del self._postings[term][doc_id]
            if not self._postings[term]:
                del self._postings[term]

    def search(self, query: Counter[str] | Iterable[str], k: int) -> list[tuple[str, float]]:
        qtf = query if isinstance(query, Counter) else Counter(query)
        n = len(self._lengths)
        if n == 0 or not qtf:
            return []
        avg = self._total_len / n or 1.0
        scores: dict[str, float] = defaultdict(float)
        for term, q_count in qtf.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = bm25_idf(n, len(postings))
            for doc_id, tf in postings.items():
                scores[doc_id] += q_count * bm25_term(tf, self._lengths[doc_id], avg, idf, self.k1, self.b)
        return top_k(scores, k)


def top_k(scores: Mapping[str, float], k: int) -> list[tuple[str, float]]:
    """Highest scores first; ties broken by id so results are deterministic across runs and backends."""
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """RRF (Cormack, Clarke & Büttcher, SIGIR 2009): ``score(d) = Σ 1 / (k + rank_i(d))``.

    Rank-based, so it needs no score calibration between BM25 (unbounded) and cosine similarity ([-1, 1]).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return top_k(scores, len(scores))
