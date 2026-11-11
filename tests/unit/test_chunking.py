from __future__ import annotations

import itertools
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ragtr.chunking import (
    ChunkingStrategy,
    FixedChunker,
    SentenceChunker,
    StructureChunker,
    make_chunker,
)
from ragtr.models import Document

DOC = Document(
    "akcaburc-su",
    "Akçaburç Belediyesi Su Aboneliği Rehberi",
    """# Akçaburç Belediyesi Su Aboneliği Rehberi

Belediye içme suyu aboneliği hizmeti verir. Bu rehber başvuru ve ücretler hakkında bilgi verir.

## Başvuru

Başvuru hizmet binasının 3. katında yapılır. Birim 08.30 ile 17.00 saatleri arasında açıktır.

## Ücretler

Yeni konut aboneliklerinde 1.450 TL güvence bedeli alınır. Bedel abonelik bitince iade edilir.
""",
)


def test_structure_chunks_never_cross_a_heading() -> None:
    chunks = StructureChunker(target=200, maximum=300).chunk(DOC)
    assert [c.section for c in chunks] == ["", "Başvuru", "Ücretler"]
    assert all("##" not in c.body for c in chunks)


def test_contextual_header_prefixes_title_and_section_but_body_stays_document_text() -> None:
    chunk = StructureChunker(target=200, maximum=300).chunk(DOC)[2]
    assert chunk.text.startswith("Akçaburç Belediyesi Su Aboneliği Rehberi > Ücretler\n")
    assert chunk.body == DOC.text[chunk.start : chunk.end]
    assert "1.450 TL" in chunk.body


def test_plain_structure_chunker_indexes_the_body_only() -> None:
    chunks = make_chunker(ChunkingStrategy.STRUCTURE).chunk(DOC)
    assert all(c.text == c.body for c in chunks)


def test_sentence_chunker_does_not_split_mid_sentence() -> None:
    chunks = SentenceChunker(target=12, maximum=18).chunk(DOC)
    for c in chunks:
        assert c.body.rstrip().endswith((".", "Rehberi", "Başvuru", "Ücretler")), c.body


def test_fixed_chunker_windows_overlap() -> None:
    chunks = FixedChunker(size=10, overlap=3).chunk(DOC)
    words = [c.body.split() for c in chunks]
    for a, b in itertools.pairwise(words):
        assert a[-3:] == b[:3]


def test_fixed_chunker_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FixedChunker(size=10, overlap=10)


def test_chunk_ids_are_stable_and_ordered() -> None:
    chunks = make_chunker("structure+context").chunk(DOC)
    assert [c.id for c in chunks] == [f"akcaburc-su#{i:04d}" for i in range(len(chunks))]
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_a_sentence_longer_than_the_cap_is_split_not_dropped() -> None:
    long = Document("long", "Uzun", " ".join(["kelime"] * 500) + ".")
    chunks = SentenceChunker(target=100, maximum=150).chunk(long)
    assert sum(len(c.body.split()) for c in chunks) == 500
    assert max(len(c.body.split()) for c in chunks) <= 150


paragraph = st.lists(
    st.sampled_from(
        ["Başvuru yapılır.", "Ücret 450 TL.", "3. katta açıktır.", "Dr. Can gelir.", "izin", "gün"]
    ),
    min_size=1,
    max_size=40,
).map(" ".join)
markdown = st.lists(
    st.tuples(
        st.sampled_from(["#", "##", "###"]), st.sampled_from(["Başvuru", "Ücretler", "Saatler"]), paragraph
    ),
    min_size=1,
    max_size=6,
).map(lambda secs: "\n\n".join(f"{h} {t}\n\n{p}" for h, t, p in secs))


@settings(max_examples=150)
@given(markdown, st.sampled_from(list(ChunkingStrategy)), st.integers(min_value=5, max_value=60))
def test_every_word_of_every_section_body_is_in_some_chunk(
    text: str, strategy: ChunkingStrategy, size: int
) -> None:
    doc = Document("d", "Başlık", text)
    chunks = make_chunker(strategy, size).chunk(doc)
    for c in chunks:
        assert doc.text[c.start : c.end] == c.body
        assert c.body.strip()
    covered = [False] * len(text)
    for c in chunks:
        for i in range(c.start, c.end):
            covered[i] = True
    headings = {m.start() + i for m in re.finditer(r"^#+ .*$", text, re.M) for i in range(len(m.group(0)))}
    lost = [i for i, ch in enumerate(text) if not ch.isspace() and not covered[i] and i not in headings]
    assert not lost, text[min(lost) : min(lost) + 40]
