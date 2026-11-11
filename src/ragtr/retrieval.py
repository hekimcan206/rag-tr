"""Retrieval: dense, BM25 or hybrid (RRF), optionally reranked by a cross-encoder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from opentelemetry import trace

from ragtr.embeddings import Embedder
from ragtr.lexical import reciprocal_rank_fusion
from ragtr.models import ScoredChunk
from ragtr.rerank import Reranker
from ragtr.store.base import ChunkStore
from ragtr.text.turkish import Analyzer

_tracer = trace.get_tracer("ragtr.retrieval")


class Mode(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    mode: Mode = Mode.HYBRID
    k: int = 10
    candidates: int = 50  # per first-stage retriever, before fusion
    rrf_k: int = 60
    rerank: bool = True
    rerank_top: int = 20  # cross-encoder cost grows linearly with this

    def with_(self, **changes: object) -> RetrievalConfig:
        return replace(self, **changes)  # type: ignore[arg-type]


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        analyzer: Analyzer,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.analyzer = analyzer
        self.reranker = reranker

    def search(self, query: str, config: RetrievalConfig | None = None) -> list[ScoredChunk]:
        cfg = config or RetrievalConfig()
        with _tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("ragtr.mode", cfg.mode.value)
            ranks: dict[str, dict[str, int]] = {}
            lists: list[list[str]] = []
            scores: dict[str, float] = {}
            depth = max(cfg.candidates, cfg.k)
            if cfg.mode in (Mode.DENSE, Mode.HYBRID):
                with _tracer.start_as_current_span("retrieve.dense"):
                    dense = self.store.dense_search(self.embedder.embed_query(query), depth)
                lists.append([cid for cid, _ in dense])
                for r, (cid, s) in enumerate(dense, 1):
                    ranks.setdefault(cid, {})["dense"] = r
                    scores[cid] = s
            if cfg.mode in (Mode.BM25, Mode.HYBRID):
                with _tracer.start_as_current_span("retrieve.bm25"):
                    lexical = self.store.lexical_search(self.analyzer.term_counts(query), depth)
                lists.append([cid for cid, _ in lexical])
                for r, (cid, s) in enumerate(lexical, 1):
                    ranks.setdefault(cid, {})["bm25"] = r
                    scores[cid] = s
            fused = (
                reciprocal_rank_fusion(lists, cfg.rrf_k)
                if len(lists) > 1
                else [(cid, scores[cid]) for cid in lists[0]]
            )
            use_reranker = cfg.rerank and self.reranker is not None
            keep = max(cfg.rerank_top, cfg.k) if use_reranker else cfg.k
            fused = fused[:keep]
            chunks = self.store.get_chunks([cid for cid, _ in fused])
            results = [ScoredChunk(chunks[cid], s, ranks.get(cid, {})) for cid, s in fused if cid in chunks]
            if use_reranker and self.reranker is not None and results:
                with _tracer.start_as_current_span("retrieve.rerank") as rspan:
                    rspan.set_attribute("ragtr.rerank.pairs", len(results))
                    rr = self.reranker.score(query, [r.chunk.text for r in results])
                results = [
                    ScoredChunk(r.chunk, s, {**r.ranks, "fused": i})
                    for i, (r, s) in enumerate(zip(results, rr, strict=True), 1)
                ]
                results.sort(key=lambda r: (-r.score, r.chunk.id))
            span.set_attribute("ragtr.results", len(results[: cfg.k]))
            return results[: cfg.k]
