"""Retrieval and answer metrics.

Retrieval metrics take a ranked list of chunk ids and the set of relevant chunk ids (binary relevance):

* ``recall@k`` — share of the relevant chunks found in the top k. Most questions have exactly one relevant
  chunk, so this is also the hit rate; with overlapping fixed windows a span can sit in two chunks.
* ``MRR@k`` — reciprocal rank of the first relevant chunk, 0 beyond k.
* ``nDCG@k`` — binary-gain DCG with the usual ``1 / log2(rank + 1)`` discount, normalised by the ideal
  ranking of ``min(|relevant|, k)`` hits (Järvelin & Kekäläinen 2002).

Answer metrics are computed without an LLM judge so they are cheap, deterministic and reproducible:

* ``accuracy`` — the answer is not an abstention and contains every required normalised fact string
  (``must``: the number, the weekday, …), compared after Turkish lowercasing and ASCII folding.
* ``citation precision`` — share of cited contexts that contain the answer span of the gold document.
* ``supported`` — at least one cited context contains the answer span.
* ``groundedness`` — share of the answer's content terms that occur in the cited contexts. This is a
  lexical proxy for faithfulness (in the spirit of RAGAS statement-level faithfulness, Es et al. 2023), not
  an LLM-judged score: it catches invented numbers and names, not a wrong paraphrase built from context words.
* ``abstention`` — on unanswerable questions, the share answered with the abstention sentence;
  ``false abstention`` is the share of answerable questions that were refused.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean

from ragtr.text.turkish import Analyzer, fold_diacritics, tr_lower

_MARKER = re.compile(r"\[\d+(?:\s*[,;]\s*\d+)*\]")


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for i, cid in enumerate(ranked[:k], 1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], 1) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float

    @staticmethod
    def of(ranked: Sequence[str], relevant: set[str]) -> RetrievalScores:
        return RetrievalScores(
            recall_at_k(ranked, relevant, 1),
            recall_at_k(ranked, relevant, 5),
            recall_at_k(ranked, relevant, 10),
            reciprocal_rank(ranked, relevant, 10),
            ndcg_at_k(ranked, relevant, 10),
        )

    @staticmethod
    def mean(items: Iterable[RetrievalScores]) -> RetrievalScores:
        rows = list(items)
        if not rows:
            return RetrievalScores(0.0, 0.0, 0.0, 0.0, 0.0)
        return RetrievalScores(
            fmean(r.recall_at_1 for r in rows),
            fmean(r.recall_at_5 for r in rows),
            fmean(r.recall_at_10 for r in rows),
            fmean(r.mrr_at_10 for r in rows),
            fmean(r.ndcg_at_10 for r in rows),
        )


def normalise_answer(text: str) -> str:
    """Lowercase (Turkish rules), fold diacritics, collapse whitespace; markers removed."""
    return re.sub(r"\s+", " ", fold_diacritics(tr_lower(_MARKER.sub(" ", text)))).strip()


def contains_all(answer: str, must: Sequence[str]) -> bool:
    norm = normalise_answer(answer)
    return all(m in norm for m in must)


def groundedness(analyzer: Analyzer, answer: str, cited_texts: Sequence[str]) -> float:
    """Share of the answer's analysed terms present in the cited contexts (1.0 for an empty answer)."""
    terms = set(analyzer.terms(_MARKER.sub(" ", answer)))
    if not terms:
        return 1.0
    support: set[str] = set()
    for text in cited_texts:
        support.update(analyzer.terms(text))
    return len(terms & support) / len(terms)


@dataclass(frozen=True, slots=True)
class AnswerScores:
    """Per-question answer judgement; ``None`` where a metric does not apply to the question."""

    answerable: bool
    abstained: bool
    correct: bool | None
    citation_precision: float | None
    supported: bool | None
    groundedness: float | None


def mean_or_none(values: Iterable[float | bool | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return fmean(present) if present else None
