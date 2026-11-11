"""Second-stage rerankers.

A cross-encoder reads query and passage together, so it can use word order and negation that a bi-encoder
compresses away; it is too slow for the whole corpus, so it rescores the top candidates of the first stage.
Default model: ``BAAI/bge-reranker-v2-m3`` (multilingual, XLM-RoBERTa based, Apache-2.0).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from ragtr.text.turkish import Analyzer

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

RERANKERS: dict[str, str] = {
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
    # 118M parameters, mMARCO-trained (machine-translated MS MARCO incl. Turkish); ~5x faster on CPU.
    "mmarco-minilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str = "bge-v2-m3", device: str = "cpu", max_length: int = 512) -> None:
        self.name = model
        self._hf_id = RERANKERS.get(model, model)
        self._device = device
        self._max_length = max_length
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415 - optional heavy import

            self._model = CrossEncoder(self._hf_id, device=self._device, max_length=self._max_length)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        out: Any = self._load().predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(x) for x in out]


class OverlapReranker:
    """Deterministic test double: share of analysed query terms present in the passage."""

    name = "overlap"

    def __init__(self, analyzer: Analyzer | None = None) -> None:
        self._analyzer = analyzer or Analyzer()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(self._analyzer.terms(query))
        if not q:
            return [0.0 for _ in passages]
        return [len(q & set(self._analyzer.terms(p))) / len(q) for p in passages]


def make_reranker(name: str, device: str = "cpu") -> Reranker | None:
    if name in {"", "none"}:
        return None
    if name == "overlap":
        return OverlapReranker()
    return CrossEncoderReranker(name, device=device)
