from __future__ import annotations

from pathlib import Path

import pytest

from ragtr.chunking import make_chunker
from ragtr.data.synth import QA, generate
from ragtr.embeddings import HashingEmbedder
from ragtr.models import Document
from ragtr.pipeline import Indexer
from ragtr.store.memory import MemoryStore
from ragtr.text.turkish import Analyzer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic() -> tuple[list[Document], list[QA]]:
    docs, qas = generate()
    return [Document(d.id, d.title, d.text) for d in docs], qas


@pytest.fixture(scope="session")
def municipal_docs(synthetic: tuple[list[Document], list[QA]]) -> list[Document]:
    """The 25 municipal guides: five near-identical families, the hard case for entity disambiguation."""
    docs, _ = synthetic
    return [
        d
        for d in docs
        if d.id.split("-")[0] in {"yesilvadi", "akcaburc", "isikliova", "ikizpinar", "camliyaka"}
    ]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(256)


@pytest.fixture
def memory_store(municipal_docs: list[Document], embedder: HashingEmbedder) -> MemoryStore:
    store = MemoryStore()
    indexer = Indexer(make_chunker("structure+context"), embedder, Analyzer(), store)
    for doc in municipal_docs:
        indexer.index(doc)
    return store
