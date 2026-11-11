"""Storage interface: chunks, their embeddings and their BM25 term counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from ragtr.embeddings import Vector
from ragtr.models import Chunk, Document


class ChunkStore(Protocol):
    def add(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Vector,
        terms: Sequence[Counter[str]],
    ) -> None:
        """Insert or replace a document and all its chunks atomically."""
        ...

    def delete_document(self, document_id: str) -> bool: ...

    def dense_search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        """``(chunk_id, cosine similarity)`` best first."""
        ...

    def lexical_search(self, query: Counter[str], k: int) -> list[tuple[str, float]]:
        """``(chunk_id, BM25 score)`` best first."""
        ...

    def get_chunks(self, ids: Sequence[str]) -> dict[str, Chunk]: ...

    def count_chunks(self) -> int: ...

    def count_documents(self) -> int: ...
