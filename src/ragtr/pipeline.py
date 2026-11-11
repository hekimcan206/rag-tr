"""Ingestion and question answering on top of the store, retriever and generator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from opentelemetry import trace

from ragtr.chunking import Chunker
from ragtr.embeddings import Embedder
from ragtr.generation import Answer, Generator
from ragtr.models import Chunk, Document, ScoredChunk
from ragtr.retrieval import RetrievalConfig, Retriever
from ragtr.store.base import ChunkStore
from ragtr.text.turkish import Analyzer

_tracer = trace.get_tracer("ragtr.pipeline")


class Indexer:
    def __init__(self, chunker: Chunker, embedder: Embedder, analyzer: Analyzer, store: ChunkStore) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.analyzer = analyzer
        self.store = store

    def index(self, document: Document) -> list[Chunk]:
        with _tracer.start_as_current_span("index") as span:
            chunks = self.chunker.chunk(document)
            span.set_attribute("ragtr.chunks", len(chunks))
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            terms = [self.analyzer.term_counts(c.text) for c in chunks]
            self.store.add(document, chunks, vectors, terms)
            return chunks


@dataclass(frozen=True, slots=True)
class RagResponse:
    question: str
    answer: Answer
    contexts: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        retrieval: RetrievalConfig | None = None,
        context_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.retrieval = retrieval or RetrievalConfig()
        self.context_k = context_k

    def ask(self, question: str, retrieval: RetrievalConfig | None = None) -> RagResponse:
        with _tracer.start_as_current_span("ask"):
            t0 = time.perf_counter()
            contexts = self.retriever.search(question, retrieval or self.retrieval)[: self.context_k]
            t1 = time.perf_counter()
            answer = self.generator.generate(question, contexts)
            t2 = time.perf_counter()
        return RagResponse(
            question,
            answer,
            contexts,
            {"retrieve": (t1 - t0) * 1000, "generate": (t2 - t1) * 1000, "total": (t2 - t0) * 1000},
        )
