"""Embedding models behind one small interface.

* ``HashingEmbedder`` — deterministic signed feature hashing of analysed terms and character trigrams.
  No model download, identical vectors on every machine: unit tests, integration tests and the CI demo
  path use it. It is a lexical model in dense clothing, so it is *not* used for reported dense numbers.
* ``SentenceTransformerEmbedder`` — any Hugging Face sentence-embedding model (default
  ``intfloat/multilingual-e5-small``), loaded lazily, CPU by default. E5 models were trained with
  ``"query: "`` / ``"passage: "`` prefixes and degrade without them; the registry below applies them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from ragtr.text.turkish import Analyzer, AnalyzerConfig, Stemmer

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Vector: ...

    def embed_query(self, text: str) -> Vector: ...


def _l2(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing (Weinberger et al. 2009) over stems and character trigrams."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self.name = f"hashing-{dim}"
        self._analyzer = Analyzer(AnalyzerConfig(stemmer=Stemmer.PREFIX5))

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[str]:
        terms = self._analyzer.terms(text)
        grams = [f"#{t[i : i + 3]}" for t in terms for i in range(max(len(t) - 2, 1))]
        return terms + grams

    def _vector(self, text: str) -> Vector:
        vec = np.zeros(self._dim, dtype=np.float32)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            vec[h % self._dim] += 1.0 if (h >> 63) & 1 else -1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return _l2(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> Vector:
        vec: Vector = _l2(self._vector(text)[None, :])[0]
        return vec


@dataclass(frozen=True, slots=True)
class ModelSpec:
    hf_id: str
    query_prefix: str = ""
    passage_prefix: str = ""


#: Short names used in configuration and in the evaluation tables.
MODELS: dict[str, ModelSpec] = {
    "e5-small": ModelSpec("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": ModelSpec("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "minilm-l12": ModelSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "bge-m3": ModelSpec("BAAI/bge-m3"),
}


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "e5-small", device: str = "cpu", batch_size: int = 32) -> None:
        self.spec = MODELS.get(model, ModelSpec(model))
        self.name = model
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - optional heavy import

            self._model = SentenceTransformer(self.spec.hf_id, device=self._device)
        return self._model

    @property
    def dim(self) -> int:
        dim = self._load().get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"{self.spec.hf_id} does not report an embedding dimension")
        return int(dim)

    def _encode(self, texts: Sequence[str]) -> Vector:
        out: Any = self._load().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([self.spec.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> Vector:
        vec: Vector = self._encode([self.spec.query_prefix + text])[0]
        return vec


def make_embedder(name: str, device: str = "cpu") -> Embedder:
    if name.startswith("hashing"):
        _, _, dim = name.partition("-")
        return HashingEmbedder(int(dim) if dim else 384)
    return SentenceTransformerEmbedder(name, device=device)
