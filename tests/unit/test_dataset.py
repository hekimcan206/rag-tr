from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ragtr.data.dataset import load_corpus, load_qa
from ragtr.data.synth import QA, generate, write
from ragtr.models import Document
from tests.conftest import ROOT


def test_generator_is_deterministic(tmp_path: Path) -> None:
    write(tmp_path / "a")
    write(tmp_path / "b")
    files_a = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert files_a
    for rel in files_a:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_committed_data_set_is_exactly_what_the_generator_produces() -> None:
    docs, qas = generate()
    committed = {d.id: d for d in load_corpus(ROOT / "data")}
    assert set(committed) == {d.id for d in docs}
    for doc in docs:
        assert committed[doc.id].text == doc.text
        assert committed[doc.id].title == doc.title
    assert [asdict(q) for q in load_qa(ROOT / "data")] == [asdict(q) for q in qas]


def test_every_answer_span_is_unique_in_its_document(synthetic: tuple[list[Document], list[QA]]) -> None:
    docs, qas = synthetic
    by_id = {d.id: d for d in docs}
    for qa in qas:
        if qa.answerable:
            assert qa.document_id is not None
            assert qa.span is not None
            assert by_id[qa.document_id].text.count(qa.span) == 1, qa.id


def test_shape_of_the_question_set(synthetic: tuple[list[Document], list[QA]]) -> None:
    docs, qas = synthetic
    variants = Counter(q.variant for q in qas)
    assert variants["canonical"] == variants["paraphrase"] == variants["ascii"] > 150
    assert variants["unanswerable"] >= 40
    assert len(docs) == 59
    assert len({q.id for q in qas}) == len(qas)


def test_ascii_questions_have_no_turkish_letters_or_punctuation(
    synthetic: tuple[list[Document], list[QA]],
) -> None:
    _, qas = synthetic
    for qa in qas:
        if qa.variant == "ascii":
            assert not set(qa.question) & set("çğıöşüÇĞİÖŞÜ?'’")


def test_required_answer_strings_are_normalised(synthetic: tuple[list[Document], list[QA]]) -> None:
    _, qas = synthetic
    for qa in qas:
        for m in qa.must:
            assert m == m.lower()
            assert not set(m) & set("çğıöşü")


def test_qa_jsonl_is_utf8_and_keeps_turkish_letters() -> None:
    first = (ROOT / "data" / "qa.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert "\\u" not in first
    assert json.loads(first)["variant"] == "canonical"
