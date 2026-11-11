"""Core value types shared by ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    text: str
    source: str = ""


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit. ``text`` is what is embedded and indexed (it may carry a heading prefix);
    ``body`` is the document text the chunk covers, ``start``/``end`` its offsets in the document."""

    id: str
    document_id: str
    ord: int
    text: str
    body: str
    start: int
    end: int
    title: str = ""
    section: str = ""


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    # Per-retriever ranks (1-based) before fusion/reranking; useful for debugging and for the API response.
    ranks: dict[str, int] = field(default_factory=dict)
