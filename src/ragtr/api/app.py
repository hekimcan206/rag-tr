"""HTTP API: ingest documents, search, ask with citations."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Path, Request, status
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field

from ragtr import __version__
from ragtr.config import Components, Settings, build_components
from ragtr.generation import ABSTAIN
from ragtr.models import Document, ScoredChunk
from ragtr.pipeline import Indexer, RagPipeline
from ragtr.retrieval import Mode, Retriever
from ragtr.store.base import ChunkStore
from ragtr.telemetry import configure_logging, configure_tracing

log = logging.getLogger("ragtr.api")
_DOC_ID = r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,199}$"


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)
    source: str = Field(default="", max_length=500)


class Indexed(BaseModel):
    document_id: str
    chunks: int


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    mode: Mode | None = None
    rerank: bool | None = None


class Hit(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    section: str
    score: float
    ranks: dict[str, int]
    text: str


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    mode: Mode | None = None


class Citation(BaseModel):
    marker: int
    chunk_id: str
    document_id: str
    title: str
    section: str
    snippet: str


class AskOut(BaseModel):
    question: str
    answer: str
    abstained: bool
    model: str
    citations: list[Citation]
    contexts: list[Hit]
    timings_ms: dict[str, float]


def _hit(sc: ScoredChunk) -> Hit:
    c = sc.chunk
    return Hit(chunk_id=c.id, document_id=c.document_id, title=c.title, section=c.section,
               score=round(sc.score, 6), ranks=sc.ranks, text=c.body)  # fmt: skip


class AppState:
    def __init__(self, settings: Settings, components: Components, store: ChunkStore) -> None:
        self.settings = settings
        self.components = components
        self.store = store
        self.indexer = Indexer(components.chunker, components.embedder, components.analyzer, store)
        self.retriever = Retriever(store, components.embedder, components.analyzer, components.reranker)
        self.pipeline = RagPipeline(self.retriever, components.generator, components.retrieval,
                                    context_k=settings.top_k)  # fmt: skip


def _state(request: Request) -> AppState:
    rag: AppState = request.app.state.rag
    return rag


# Module level on purpose: with postponed annotations FastAPI resolves this name from module globals.
StateDep = Annotated[AppState, Depends(_state)]


def _postgres_store(settings: Settings, components: Components) -> ChunkStore:
    from ragtr.store.postgres import PostgresStore  # noqa: PLC0415 - keeps psycopg out of memory-only use

    return PostgresStore(
        settings.database_url.get_secret_value(),
        embedder=components.embedder.name,
        dim=components.embedder.dim,
        analyzer=components.analyzer.name,
        ef_search=settings.hnsw_ef_search,
    )


def create_app(
    settings: Settings | None = None,
    components: Components | None = None,
    store: ChunkStore | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        comps = components or build_components(settings)
        st = store or _postgres_store(settings, comps)
        app.state.rag = AppState(settings, comps, st)
        fields = {
            "embedder": comps.embedder.name,
            "generator": comps.generator.name,
            "reranker": comps.reranker.name if comps.reranker else "none",
        }
        log.info("ready", extra={"fields": fields})
        yield
        close = getattr(st, "close", None)
        if callable(close):
            close()

    app = FastAPI(
        title="rag-tr",
        version=__version__,
        summary="Turkish RAG: hybrid retrieval over pgvector + BM25 and cited answers.",
        lifespan=lifespan,
    )
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    def readyz(s: StateDep) -> dict[str, Any]:
        try:
            return {
                "status": "ready",
                "documents": s.store.count_documents(),
                "chunks": s.store.count_chunks(),
            }
        except Exception as exc:  # readiness must report, not raise
            log.warning("not ready", exc_info=exc)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "store unavailable") from exc

    @app.put("/v1/documents/{document_id}", tags=["documents"])
    def put_document(
        document_id: Annotated[str, Path(pattern=_DOC_ID)], body: DocumentIn, s: StateDep
    ) -> Indexed:
        if len(body.text) > s.settings.max_document_chars:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "document too large")
        chunks = s.indexer.index(Document(document_id, body.title.strip(), body.text, body.source))
        log.info("indexed", extra={"fields": {"document_id": document_id, "chunks": len(chunks)}})
        return Indexed(document_id=document_id, chunks=len(chunks))

    @app.delete("/v1/documents/{document_id}", status_code=204, tags=["documents"])
    def delete_document(document_id: Annotated[str, Path(pattern=_DOC_ID)], s: StateDep) -> None:
        if not s.store.delete_document(document_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")

    @app.post("/v1/search", tags=["query"])
    def search(body: SearchIn, s: StateDep) -> list[Hit]:
        _check_length(body.query, s.settings)
        cfg = s.components.retrieval.with_(k=body.k)
        if body.mode:
            cfg = cfg.with_(mode=body.mode)
        if body.rerank is not None:
            cfg = cfg.with_(rerank=body.rerank)
        return [_hit(h) for h in s.retriever.search(body.query, cfg)]

    @app.post("/v1/ask", tags=["query"])
    def ask(body: AskIn, s: StateDep) -> AskOut:
        _check_length(body.question, s.settings)
        cfg = s.components.retrieval.with_(mode=body.mode) if body.mode else None
        result = s.pipeline.ask(body.question, cfg)
        a = result.answer
        citations = [
            Citation(marker=n, chunk_id=c.chunk.id, document_id=c.chunk.document_id, title=c.chunk.title,
                     section=c.chunk.section, snippet=_snippet(c.chunk.body))
            for n in a.citations
            for c in [result.contexts[n - 1]]
        ]  # fmt: skip
        log.info("answered", extra={"fields": {"abstained": a.abstained, "citations": a.citations,
                                               "ms": round(result.timings_ms["total"], 1)}})  # fmt: skip
        return AskOut(
            question=body.question,
            answer=ABSTAIN if a.abstained else a.text,
            abstained=a.abstained,
            model=a.model,
            citations=citations,
            contexts=[_hit(c) for c in result.contexts],
            timings_ms={k: round(v, 2) for k, v in result.timings_ms.items()},
        )

    return app


def _check_length(text: str, settings: Settings) -> None:
    if len(text) > settings.max_question_chars:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "question too long")


def _snippet(text: str, limit: int = 280) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def main() -> FastAPI:
    """ASGI factory for ``uvicorn --factory ragtr.api.app:main``."""
    configure_logging()
    configure_tracing()
    return create_app()
