"""In-memory store: exact cosine search over a NumPy matrix plus an in-memory BM25 index.

Used by the evaluation harness (exact search, so retrieval numbers are not blurred by ANN recall) and by
unit tests. Not thread-safe for concurrent writes; the API uses the PostgreSQL store.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from ragtr.embeddings import Vector
from ragtr.lexical import BM25Index, top_k
from ragtr.models import Chunk, Document


class MemoryStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._docs: dict[str, list[str]] = {}
        self._ids: list[str] = []
        self._matrix: Vector | None = None
        self._pending: dict[str, Vector] = {}
        self._vectors: dict[str, Vector] = {}
        self._bm25 = BM25Index()

    def add(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Vector,
        terms: Sequence[Counter[str]],
    ) -> None:
        if len(chunks) != len(embeddings) or len(chunks) != len(terms):
            raise ValueError("chunks, embeddings and terms must have the same length")
        self.delete_document(document.id)
        self._docs[document.id] = [c.id for c in chunks]
        for chunk, vec, tf in zip(chunks, embeddings, terms, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vec
            self._bm25.add(chunk.id, tf)
        self._matrix = None

    def delete_document(self, document_id: str) -> bool:
        ids = self._docs.pop(document_id, None)
        if ids is None:
            return False
        for cid in ids:
            self._chunks.pop(cid, None)
            self._vectors.pop(cid, None)
            self._bm25.remove(cid)
        self._matrix = None
        return True

    def _ensure_matrix(self) -> Vector:
        if self._matrix is None:
            self._ids = sorted(self._vectors)
            if self._ids:
                self._matrix = np.stack([self._vectors[i] for i in self._ids])
            else:
                self._matrix = np.zeros((0, 1), dtype=np.float32)
        return self._matrix

    def dense_search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        matrix = self._ensure_matrix()
        if matrix.shape[0] == 0:
            return []
        sims = matrix @ query
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        return top_k({self._ids[i]: float(sims[i]) for i in idx}, k)

    def lexical_search(self, query: Counter[str], k: int) -> list[tuple[str, float]]:
        return self._bm25.search(query, k)

    def get_chunks(self, ids: Sequence[str]) -> dict[str, Chunk]:
        return {i: self._chunks[i] for i in ids if i in self._chunks}

    def count_chunks(self) -> int:
        return len(self._chunks)

    def count_documents(self) -> int:
        return len(self._docs)
