"""Runtime configuration from environment variables (prefix ``RAGTR_``) and component wiring."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragtr.chunking import Chunker, ChunkingStrategy, make_chunker
from ragtr.embeddings import Embedder, make_embedder
from ragtr.generation import Generator, make_generator
from ragtr.rerank import Reranker, make_reranker
from ragtr.retrieval import Mode, RetrievalConfig
from ragtr.text.turkish import TURKISH, Analyzer


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGTR_", env_file=".env", extra="ignore")

    database_url: SecretStr = SecretStr("postgresql://ragtr_app:ragtr-app-demo@localhost:58471/ragtr")
    embedder: str = "e5-small"
    reranker: str = "none"
    device: str = "cpu"
    generator: str = "extractive"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_timeout_s: float = 120.0
    chunking: ChunkingStrategy = ChunkingStrategy.STRUCTURE_CONTEXT
    chunk_words: int = Field(default=120, ge=20, le=400)
    mode: Mode = Mode.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    candidates: int = Field(default=50, ge=5, le=500)
    rerank_top: int = Field(default=20, ge=1, le=100)
    hnsw_ef_search: int = Field(default=100, ge=10, le=1000)
    max_document_chars: int = Field(default=500_000, ge=1_000)
    max_question_chars: int = Field(default=1_000, ge=10)


@dataclass(frozen=True, slots=True)
class Components:
    analyzer: Analyzer
    chunker: Chunker
    embedder: Embedder
    reranker: Reranker | None
    generator: Generator
    retrieval: RetrievalConfig


def build_components(settings: Settings) -> Components:
    reranker = make_reranker(settings.reranker, settings.device)
    return Components(
        analyzer=Analyzer(TURKISH),
        chunker=make_chunker(settings.chunking, settings.chunk_words),
        embedder=make_embedder(settings.embedder, settings.device),
        reranker=reranker,
        generator=make_generator(
            settings.generator, settings.ollama_url, settings.ollama_model, settings.ollama_timeout_s
        ),
        retrieval=RetrievalConfig(
            mode=settings.mode,
            k=settings.top_k,
            candidates=settings.candidates,
            rerank=reranker is not None,
            rerank_top=settings.rerank_top,
        ),
    )
