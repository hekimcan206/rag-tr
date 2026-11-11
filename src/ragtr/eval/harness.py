"""Evaluation harness: build an index per configuration, run every question, aggregate metrics.

Retrieval is evaluated on the in-memory store, which searches *exactly* (brute-force cosine, exhaustive
BM25). The PostgreSQL store returns the same BM25 scores (integration-tested) but its HNSW index is
approximate, and ANN recall is a property of the index parameters, not of the retrieval method being
compared here.

Embeddings and reranker scores are cached per text, so a suite that varies only the fusion mode or the
lexical analyser pays for each model call once.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ragtr.chunking import ChunkingStrategy, make_chunker
from ragtr.data.synth import QA
from ragtr.embeddings import Embedder, Vector, make_embedder
from ragtr.eval.metrics import (
    AnswerScores,
    RetrievalScores,
    contains_all,
    groundedness,
    mean_or_none,
)
from ragtr.generation import Generator
from ragtr.models import Chunk, Document, ScoredChunk
from ragtr.rerank import Reranker
from ragtr.retrieval import Mode, RetrievalConfig, Retriever
from ragtr.store.memory import MemoryStore
from ragtr.text.turkish import NAIVE, TURKISH, Analyzer, AnalyzerConfig, Stemmer

K = 10
CONTEXT_K = 5
# Context budget for the size-fair recall: chunking strategies produce chunks of very different length, so
# recall at a fixed k favours big chunks. R@300w asks whether the answer is inside the top-ranked chunks that
# fit in 300 words, roughly what a small local model gets as context.
WORD_BUDGET = 300
VARIANTS = ("canonical", "paraphrase", "ascii")


class CachedEmbedder:
    """Memoises an embedder by input text; the wrapped model is only called for unseen texts."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = inner.name
        self._docs: dict[str, Vector] = {}
        self._queries: dict[str, Vector] = {}
        self.seconds = 0.0

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        missing = list(dict.fromkeys(t for t in texts if t not in self._docs))
        if missing:
            t0 = time.perf_counter()
            for text, vec in zip(missing, self.inner.embed_documents(missing), strict=True):
                self._docs[text] = vec
            self.seconds += time.perf_counter() - t0
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._docs[t] for t in texts])

    def embed_query(self, text: str) -> Vector:
        vec = self._queries.get(text)
        if vec is None:
            t0 = time.perf_counter()
            vec = self._queries[text] = self.inner.embed_query(text)
            self.seconds += time.perf_counter() - t0
        return vec


class CachedReranker:
    def __init__(self, inner: Reranker) -> None:
        self.inner = inner
        self.name = inner.name
        self._scores: dict[tuple[str, str], float] = {}
        self.pairs = 0
        self.seconds = 0.0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        missing = list(dict.fromkeys(p for p in passages if (query, p) not in self._scores))
        if missing:
            t0 = time.perf_counter()
            for passage, s in zip(missing, self.inner.score(query, missing), strict=True):
                self._scores[(query, passage)] = s
            self.seconds += time.perf_counter() - t0
            self.pairs += len(missing)
        return [self._scores[(query, p)] for p in passages]


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    mode: Mode = Mode.HYBRID
    chunking: ChunkingStrategy = ChunkingStrategy.STRUCTURE_CONTEXT
    analyzer: AnalyzerConfig = TURKISH
    embedder: str | None = None  # None: the suite's default embedder
    rerank: bool = False
    answers: bool = False


@dataclass(slots=True)
class RunResult:
    name: str
    mode: str
    chunking: str
    analyzer: str
    embedder: str
    rerank: bool
    questions: int
    chunks: int
    chunk_words: float
    overall: dict[str, float]
    by_variant: dict[str, dict[str, float]]
    answers: dict[str, Any] | None = None
    seconds: float = 0.0


@dataclass(slots=True)
class SuiteResult:
    suite: str
    description: str
    environment: dict[str, Any]
    runs: list[RunResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relevant_chunks(qa: QA, documents: dict[str, Document], chunks: Sequence[Chunk]) -> set[str]:
    """Chunks of the gold document that fully contain the answer span (any overlap as a fallback).

    Spans are unique within their document (asserted by the generator), so this is unambiguous.
    """
    if not qa.answerable or qa.document_id is None or qa.span is None:
        return set()
    text = documents[qa.document_id].text
    s = text.index(qa.span)
    e = s + len(qa.span)
    own = [c for c in chunks if c.document_id == qa.document_id]
    full = {c.id for c in own if c.start <= s and e <= c.end}
    return full or {c.id for c in own if c.start < e and s < c.end}


@dataclass(slots=True)
class _Index:
    store: MemoryStore
    chunks: list[Chunk]
    relevant: dict[str, set[str]]


class Evaluator:
    def __init__(
        self,
        documents: Sequence[Document],
        qas: Sequence[QA],
        embedder_factory: Callable[[str], Embedder],
        default_embedder: str,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        answer_limit: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.documents = list(documents)
        self.by_id = {d.id: d for d in documents}
        self.qas = list(qas)
        self.default_embedder = default_embedder
        self._factory = embedder_factory
        self._embedders: dict[str, CachedEmbedder] = {}
        self._indexes: dict[tuple[str, str, str], _Index] = {}
        self.reranker = CachedReranker(reranker) if reranker else None
        self.generator = generator
        self.answer_qas = stratified_sample(self.qas, answer_limit) if answer_limit else self.qas
        self._progress = progress or (lambda _msg: None)

    def embedder(self, name: str) -> CachedEmbedder:
        if name not in self._embedders:
            self._embedders[name] = CachedEmbedder(self._factory(name))
        return self._embedders[name]

    def _index(self, chunking: ChunkingStrategy, analyzer: AnalyzerConfig, embedder: str) -> _Index:
        key = (chunking.value, analyzer.name, embedder)
        if key not in self._indexes:
            chunker = make_chunker(chunking)
            emb = self.embedder(embedder)
            an = Analyzer(analyzer)
            store = MemoryStore()
            all_chunks: list[Chunk] = []
            for doc in self.documents:
                chunks = chunker.chunk(doc)
                store.add(doc, chunks, emb.embed_documents([c.text for c in chunks]),
                          [an.term_counts(c.text) for c in chunks])  # fmt: skip
                all_chunks.extend(chunks)
            relevant = {q.id: relevant_chunks(q, self.by_id, all_chunks) for q in self.qas if q.answerable}
            self._indexes[key] = _Index(store, all_chunks, relevant)
        return self._indexes[key]

    def run(self, spec: RunSpec) -> RunResult:
        t0 = time.perf_counter()
        emb_name = spec.embedder or self.default_embedder
        index = self._index(spec.chunking, spec.analyzer, emb_name)
        if spec.rerank and self.reranker is None:
            raise ValueError(f"run {spec.name!r} needs a reranker")
        retriever = Retriever(index.store, self.embedder(emb_name), Analyzer(spec.analyzer),
                              self.reranker if spec.rerank else None)  # fmt: skip
        cfg = RetrievalConfig(mode=spec.mode, k=K, candidates=50, rerank=spec.rerank, rerank_top=20)
        scores: dict[str, list[RetrievalScores]] = defaultdict(list)
        in_budget: dict[str, list[bool]] = defaultdict(list)
        ranked: dict[str, list[ScoredChunk]] = {}
        for qa in self.qas:
            if not qa.answerable and not spec.answers:
                continue
            hits = retriever.search(qa.question, cfg)
            ranked[qa.id] = hits
            if qa.answerable:
                scores[qa.variant].append(
                    RetrievalScores.of([h.chunk.id for h in hits], index.relevant[qa.id])
                )
                in_budget[qa.variant].append(_within_budget(hits, index.relevant[qa.id]))
        overall = RetrievalScores.mean(s for rows in scores.values() for s in rows)
        result = RunResult(
            name=spec.name,
            mode=spec.mode.value,
            chunking=spec.chunking.value,
            analyzer=spec.analyzer.name,
            embedder=emb_name if spec.mode is not Mode.BM25 else "-",
            rerank=spec.rerank,
            questions=sum(len(v) for v in scores.values()),
            chunks=len(index.chunks),
            chunk_words=round(sum(len(c.body.split()) for c in index.chunks) / len(index.chunks), 1),
            overall=_round(
                asdict(overall)
                | {"recall_at_budget": float(np.mean([b for v in in_budget.values() for b in v]))}
            ),
            by_variant={
                v: _round(
                    asdict(RetrievalScores.mean(scores[v]))
                    | {"recall_at_budget": float(np.mean(in_budget[v]))}
                )
                for v in VARIANTS
                if v in scores
            },
        )
        if spec.answers:
            result.answers = self._answers(ranked, index, Analyzer(spec.analyzer))
        result.seconds = round(time.perf_counter() - t0, 2)
        self._progress(f"{spec.name}: R@5={result.overall['recall_at_5']:.3f} ({result.seconds}s)")
        return result

    def _answers(
        self, ranked: dict[str, list[ScoredChunk]], index: _Index, analyzer: Analyzer
    ) -> dict[str, Any]:
        if self.generator is None:
            raise ValueError("answer metrics need a generator")
        judged: list[tuple[QA, AnswerScores]] = []
        latencies: list[float] = []
        for qa in self.answer_qas:
            contexts = ranked[qa.id][:CONTEXT_K]
            t0 = time.perf_counter()
            answer = self.generator.generate(qa.question, contexts)
            latencies.append(time.perf_counter() - t0)
            cited = [contexts[n - 1] for n in answer.citations]
            if not qa.answerable:
                judged.append((qa, AnswerScores(False, answer.abstained, None, None, None, None)))
                continue
            rel = index.relevant[qa.id]
            if answer.abstained:
                judged.append((qa, AnswerScores(True, True, False, None, None, None)))
                continue
            judged.append((qa, AnswerScores(
                answerable=True,
                abstained=False,
                correct=contains_all(answer.text, qa.must),
                citation_precision=(sum(c.chunk.id in rel for c in cited) / len(cited)) if cited else 0.0,
                supported=any(c.chunk.id in rel for c in cited),
                groundedness=groundedness(analyzer, answer.text, [c.chunk.body for c in cited]),
            )))  # fmt: skip
        answerable = [s for _, s in judged if s.answerable]
        unanswerable = [s for _, s in judged if not s.answerable]
        by_variant = {
            v: _round({"accuracy": mean_or_none(bool(s.correct) for q, s in judged if q.variant == v) or 0.0})
            for v in VARIANTS
        }
        return {
            "generator": self.generator.name,
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            # Accuracy counts abstentions as wrong, over all answerable questions.
            "accuracy": _r(mean_or_none(bool(s.correct) for s in answerable)),
            "citation_precision": _r(mean_or_none(s.citation_precision for s in answerable)),
            "supported": _r(mean_or_none(s.supported for s in answerable)),
            "groundedness": _r(mean_or_none(s.groundedness for s in answerable)),
            "false_abstention": _r(mean_or_none(s.abstained for s in answerable)),
            "abstention_on_unanswerable": _r(mean_or_none(s.abstained for s in unanswerable)),
            "by_variant": by_variant,
            "latency_s_median": _r(float(np.median(latencies))) if latencies else None,
            "latency_s_p95": _r(float(np.percentile(latencies, 95))) if latencies else None,
        }


def stratified_sample(qas: Sequence[QA], limit: int, seed: int = 7) -> list[QA]:
    """Equal share per variant (canonical, paraphrase, ascii, unanswerable), seeded."""
    groups: dict[str, list[QA]] = defaultdict(list)
    for qa in qas:
        groups[qa.variant].append(qa)
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    per = max(limit // len(groups), 1)
    picked: list[QA] = []
    for name in sorted(groups):
        items = groups[name]
        picked.extend(rng.sample(items, min(per, len(items))))
    return sorted(picked, key=lambda q: q.id)


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _round(values: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in values.items()}


# ----------------------------------------------------------------------------------------- suites

NORMALISATION_LADDER: list[tuple[str, AnalyzerConfig]] = [
    ("naive (str.lower)", NAIVE),
    ("+ Turkish casing", AnalyzerConfig(True, False, False, Stemmer.NONE, False)),
    ("+ apostrophe suffix", AnalyzerConfig(True, True, False, Stemmer.NONE, False)),
    ("+ stopwords", AnalyzerConfig(True, True, True, Stemmer.NONE, False)),
    ("+ Snowball stemmer", AnalyzerConfig(True, True, True, Stemmer.SNOWBALL, False)),
    ("+ ASCII folding", AnalyzerConfig(True, True, True, Stemmer.SNOWBALL, True)),
    ("prefix-5 instead of Snowball (default)", TURKISH),
    (
        "Snowball, fold before stemming",
        AnalyzerConfig(True, True, True, Stemmer.SNOWBALL, True, fold_before_stem=True),
    ),
]


def suite_specs(suite: str, embedders: Sequence[str] = ()) -> tuple[str, list[RunSpec]]:
    sc = ChunkingStrategy.STRUCTURE_CONTEXT
    if suite == "main":
        return (
            "Retrieval mode x reranker on the default pipeline (structure+context chunks, Turkish analyser).",
            [
                RunSpec("BM25", Mode.BM25),
                RunSpec("dense", Mode.DENSE),
                RunSpec("hybrid (RRF)", Mode.HYBRID, answers=True),
                RunSpec("BM25 + rerank", Mode.BM25, rerank=True),
                RunSpec("dense + rerank", Mode.DENSE, rerank=True),
                RunSpec("hybrid + rerank", Mode.HYBRID, rerank=True, answers=True),
            ],
        )
    if suite == "chunking":
        return "Chunking strategy x retrieval mode (no reranker).", [
            RunSpec(f"{c.value} / {m.value}", m, chunking=c)
            for c in ChunkingStrategy
            for m in (Mode.BM25, Mode.DENSE, Mode.HYBRID)
        ]
    if suite == "normalisation":
        ladder = [RunSpec(label, Mode.BM25, chunking=sc, analyzer=cfg) for label, cfg in NORMALISATION_LADDER]
        return "Turkish-aware lexical normalisation, BM25 alone and inside hybrid retrieval.", [
            *ladder,
            RunSpec("hybrid, naive analyser", Mode.HYBRID, analyzer=NAIVE),
            RunSpec("hybrid, Turkish analyser", Mode.HYBRID, analyzer=TURKISH),
        ]
    if suite == "embedders":
        return "Dense retrieval alone, per embedding model.", [
            RunSpec(f"dense / {e}", Mode.DENSE, embedder=e) for e in embedders
        ]
    if suite == "ci":
        # Deterministic, download-free: the hashing embedder and the extractive generator. CI re-runs this
        # suite and fails when any metric differs from the committed result.
        return "Download-free regression suite (hashing embedder, extractive generator).", [
            RunSpec("BM25", Mode.BM25),
            RunSpec("dense (hashing)", Mode.DENSE),
            RunSpec("hybrid", Mode.HYBRID, answers=True),
            RunSpec("BM25, naive analyser", Mode.BM25, analyzer=NAIVE),
            *[
                RunSpec(f"hybrid / {c.value}", Mode.HYBRID, chunking=c)
                for c in ChunkingStrategy
                if c is not sc
            ],
        ]
    if suite == "generation":
        return "Answer quality of a generator on a stratified sample (hybrid + rerank contexts).", [
            RunSpec("hybrid + rerank", Mode.HYBRID, rerank=True, answers=True),
        ]
    raise ValueError(f"unknown suite {suite!r}")


SUITES = ("main", "chunking", "normalisation", "embedders", "generation", "ci")


def environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    def version(pkg: str) -> str | None:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {p: version(p) for p in ("numpy", "torch", "sentence-transformers", "transformers")},
    }
    if hasattr(os, "getloadavg"):
        env["loadavg_start"] = [round(x, 2) for x in os.getloadavg()]
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:  # noqa: PTH123 - Linux-only, optional
            mem = dict(line.split(":", 1) for line in fh)
        env["mem_total_gb"] = round(int(mem["MemTotal"].split()[0]) / 1024**2, 1)
        env["mem_available_gb_start"] = round(int(mem["MemAvailable"].split()[0]) / 1024**2, 1)
    except (OSError, KeyError, ValueError):
        pass
    env.update(extra or {})
    return env


def run_suite(
    suite: str,
    documents: Sequence[Document],
    qas: Sequence[QA],
    embedder: str,
    reranker: Reranker | None,
    generator: Generator | None,
    embedders: Sequence[str] = (),
    answer_limit: int | None = None,
    device: str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> SuiteResult:
    description, specs = suite_specs(suite, embedders)
    evaluator = Evaluator(
        documents, qas, lambda name: make_embedder(name, device), embedder, reranker, generator,
        answer_limit, progress,
    )  # fmt: skip
    env = environment({
        "embedder": embedder,
        "reranker": reranker.name if reranker else None,
        "generator": generator.name if generator else None,
        "documents": len(documents),
        "questions": len(qas),
    })  # fmt: skip
    result = SuiteResult(suite, description, env)
    t0 = time.perf_counter()
    for spec in specs:
        result.runs.append(evaluator.run(spec))
    env["wall_seconds"] = round(time.perf_counter() - t0, 1)
    env["embedding_seconds"] = {n: round(e.seconds, 1) for n, e in evaluator._embedders.items()}
    if evaluator.reranker:
        env["rerank_pairs"] = evaluator.reranker.pairs
        env["rerank_seconds"] = round(evaluator.reranker.seconds, 1)
    if hasattr(os, "getloadavg"):
        env["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
    return result


def _within_budget(hits: Sequence[ScoredChunk], relevant: set[str], budget: int = WORD_BUDGET) -> bool:
    """True when a relevant chunk is among the top hits within ``budget`` words (the first always fits)."""
    used = 0
    for i, h in enumerate(hits):
        used += len(h.chunk.body.split())
        if i > 0 and used > budget:
            return False
        if h.chunk.id in relevant:
            return True
    return False
