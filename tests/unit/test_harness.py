from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragtr.chunking import ChunkingStrategy, make_chunker
from ragtr.cli import _metric_diffs, app
from ragtr.data.synth import QA
from ragtr.embeddings import HashingEmbedder
from ragtr.eval.harness import (
    CachedEmbedder,
    CachedReranker,
    Evaluator,
    RunSpec,
    relevant_chunks,
    stratified_sample,
    suite_specs,
)
from ragtr.eval.report import render
from ragtr.generation import ExtractiveGenerator
from ragtr.models import Document
from ragtr.rerank import OverlapReranker
from ragtr.retrieval import Mode
from tests.conftest import ROOT


def test_relevant_chunk_is_the_one_containing_the_answer_span(
    synthetic: tuple[list[Document], list[QA]],
) -> None:
    docs, qas = synthetic
    by_id = {d.id: d for d in docs}
    qa = next(q for q in qas if q.id == "akcaburc-su-aboneligi:deposit:canonical")
    chunks = make_chunker(ChunkingStrategy.STRUCTURE_CONTEXT).chunk(by_id["akcaburc-su-aboneligi"])
    [cid] = relevant_chunks(qa, by_id, chunks)
    chunk = next(c for c in chunks if c.id == cid)
    assert qa.span is not None
    assert qa.span in chunk.body
    assert chunk.section == "Ücretler"


def test_unanswerable_questions_have_no_relevant_chunks(synthetic: tuple[list[Document], list[QA]]) -> None:
    docs, qas = synthetic
    qa = next(q for q in qas if not q.answerable)
    assert relevant_chunks(qa, {d.id: d for d in docs}, []) == set()


def test_cached_embedder_calls_the_model_once_per_text() -> None:
    calls: Counter[str] = Counter()

    class Spy(HashingEmbedder):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            calls.update(texts)
            return super().embed_documents(texts)

    emb = CachedEmbedder(Spy(64))
    emb.embed_documents(["a", "b", "a"])
    emb.embed_documents(["b", "c"])
    assert calls == Counter({"a": 1, "b": 1, "c": 1})
    assert emb.embed_documents([]).shape == (0, 64)
    assert emb.embed_query("q") is emb.embed_query("q")


def test_cached_reranker_counts_unique_pairs() -> None:
    rr = CachedReranker(OverlapReranker())
    rr.score("izin günü", ["izin", "gün"])
    rr.score("izin günü", ["izin", "ücret"])
    assert rr.pairs == 3


def test_stratified_sample_is_balanced_and_seeded(synthetic: tuple[list[Document], list[QA]]) -> None:
    _, qas = synthetic
    sample = stratified_sample(qas, 40)
    assert Counter(q.variant for q in sample) == {
        "ascii": 10,
        "canonical": 10,
        "paraphrase": 10,
        "unanswerable": 10,
    }
    assert sample == stratified_sample(qas, 40)


def test_every_suite_is_defined() -> None:
    for suite in ("main", "chunking", "normalisation", "ci", "generation"):
        _, specs = suite_specs(suite)
        assert specs
    assert len(suite_specs("embedders", ["a", "b"])[1]) == 2
    with pytest.raises(ValueError, match="unknown suite"):
        suite_specs("nope")


@pytest.fixture(scope="module")
def small_eval(synthetic: tuple[list[Document], list[QA]]) -> Evaluator:
    docs, qas = synthetic
    keep = {d.id for d in docs if d.id.startswith(("akcaburc", "yesilvadi"))}
    subset = [q for q in qas if q.document_id in keep or (not q.answerable and "akcaburc" in q.id)]
    return Evaluator(
        [d for d in docs if d.id in keep], subset, lambda n: HashingEmbedder(256), "hashing-256",
        OverlapReranker(), ExtractiveGenerator(),
    )  # fmt: skip


def test_evaluator_produces_bounded_metrics_and_answer_scores(small_eval: Evaluator) -> None:
    result = small_eval.run(RunSpec("hybrid", Mode.HYBRID, rerank=True, answers=True))
    assert result.questions == sum(1 for q in small_eval.qas if q.answerable)
    assert 0.5 < result.overall["recall_at_10"] <= 1
    assert set(result.by_variant) == {"canonical", "paraphrase", "ascii"}
    assert result.answers is not None
    assert result.answers["unanswerable"] == 4
    assert 0 <= result.answers["accuracy"] <= 1
    # The extractive generator copies a context sentence, so everything it says is in its citation.
    assert result.answers["groundedness"] == 1.0


def test_a_rerank_run_without_a_reranker_is_rejected(synthetic: tuple[list[Document], list[QA]]) -> None:
    docs, qas = synthetic
    ev = Evaluator(docs[:2], qas[:3], lambda n: HashingEmbedder(64), "hashing-64")
    with pytest.raises(ValueError, match="needs a reranker"):
        ev.run(RunSpec("x", rerank=True))


def test_report_renders_retrieval_and_answer_tables(small_eval: Evaluator) -> None:
    result = small_eval.run(RunSpec("hybrid", Mode.HYBRID, answers=True))
    md = render({"suite": "t", "description": "d", "environment": {}, "runs": [asdict(result)]})
    assert "| hybrid |" in md
    assert "| hybrid | extractive |" in md


def test_metric_diff_ignores_timings_but_not_metrics() -> None:
    run = {
        "name": "a",
        "seconds": 1.0,
        "overall": {"recall_at_5": 0.5},
        "answers": {"accuracy": 1, "latency_s_median": 1},
    }
    same = {**run, "seconds": 9.0, "answers": {"accuracy": 1, "latency_s_median": 3}}
    worse = {**run, "overall": {"recall_at_5": 0.4}}
    assert _metric_diffs({"runs": [run]}, {"runs": [same]}) == []
    assert _metric_diffs({"runs": [run]}, {"runs": [worse]})
    assert _metric_diffs({"runs": [run]}, {"runs": []}) == ["run 'a' missing"]


def test_cli_ci_suite_reproduces_the_committed_result(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, [
        "eval", "--suite", "ci", "--data", str(ROOT / "data"), "--embedder", "hashing-384",
        "--out", str(tmp_path), "--compare", str(ROOT / "eval" / "results" / "ci.json"),
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "ci.json").read_text(encoding="utf-8"))["suite"] == "ci"
    assert (tmp_path / "ci.md").read_text(encoding="utf-8").startswith("## Suite `ci`")


def test_cli_generate_data(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["generate-data", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert "59 documents" in result.output
