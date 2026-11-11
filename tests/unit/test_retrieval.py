from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from ragtr.embeddings import MODELS, HashingEmbedder, make_embedder
from ragtr.models import Chunk, Document
from ragtr.rerank import OverlapReranker, make_reranker
from ragtr.retrieval import Mode, RetrievalConfig, Retriever
from ragtr.store.memory import MemoryStore
from ragtr.text.turkish import Analyzer


def test_hashing_embedder_is_deterministic_and_unit_length(embedder: HashingEmbedder) -> None:
    a = embedder.embed_documents(["Su aboneliği güvence bedeli", "Nikâh salonu kapasitesi"])
    b = embedder.embed_documents(["Su aboneliği güvence bedeli", "Nikâh salonu kapasitesi"])
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    q = embedder.embed_query("su abonesi güvence bedeli")
    assert float(a[0] @ q) > float(a[1] @ q)


def test_empty_batch_has_the_right_shape(embedder: HashingEmbedder) -> None:
    assert embedder.embed_documents([]).shape == (0, 256)


def test_e5_models_carry_their_query_and_passage_prefixes() -> None:
    assert MODELS["e5-small"].query_prefix == "query: "
    assert MODELS["e5-small"].passage_prefix == "passage: "
    assert make_embedder("hashing-128").dim == 128


@pytest.mark.parametrize("mode", [Mode.BM25, Mode.HYBRID])
def test_lexical_and_hybrid_find_the_right_entity_among_near_duplicates(
    memory_store: MemoryStore, mode: Mode
) -> None:
    retriever = Retriever(memory_store, HashingEmbedder(256), Analyzer())
    hits = retriever.search(
        "Işıklıova Belediyesi'nde su güvence bedeli ne kadardır?", RetrievalConfig(mode, k=3)
    )
    assert hits[0].chunk.document_id == "isikliova-su-aboneligi"
    assert "güvence bedeli" in hits[0].chunk.body


def test_hybrid_records_per_retriever_ranks(memory_store: MemoryStore) -> None:
    hits = Retriever(memory_store, HashingEmbedder(256), Analyzer()).search(
        "akcaburc nikah hafta sonu ucreti", RetrievalConfig(Mode.HYBRID, k=5)
    )
    assert hits[0].ranks.keys() <= {"dense", "bm25"}
    assert any(len(h.ranks) == 2 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_reranker_reorders_the_candidates(memory_store: MemoryStore) -> None:
    retriever = Retriever(memory_store, HashingEmbedder(256), Analyzer(), OverlapReranker())
    hits = retriever.search(
        "Çamlıyaka hacimli atık ayda kaç kez", RetrievalConfig(Mode.HYBRID, k=3, rerank=True)
    )
    assert "fused" in hits[0].ranks
    assert hits[0].chunk.document_id == "camliyaka-atik-toplama"


def test_reranker_off_by_config_even_when_installed(memory_store: MemoryStore) -> None:
    retriever = Retriever(memory_store, HashingEmbedder(256), Analyzer(), OverlapReranker())
    hits = retriever.search("Çamlıyaka pazar", RetrievalConfig(Mode.BM25, k=3, rerank=False))
    assert all("fused" not in h.ranks for h in hits)


def test_make_reranker() -> None:
    assert make_reranker("none") is None
    assert isinstance(make_reranker("overlap"), OverlapReranker)


def test_memory_store_replace_and_delete(embedder: HashingEmbedder) -> None:
    store = MemoryStore()
    doc = Document("d1", "T", "metin")
    chunk = Chunk("d1#0000", "d1", 0, "metin", "metin", 0, 5)
    store.add(doc, [chunk], embedder.embed_documents(["metin"]), [Counter({"metin": 1})])
    store.add(doc, [chunk], embedder.embed_documents(["metin"]), [Counter({"metin": 1})])
    assert store.count_documents() == 1
    assert store.count_chunks() == 1
    assert store.delete_document("d1")
    assert not store.delete_document("d1")
    assert store.dense_search(embedder.embed_query("metin"), 3) == []
    with pytest.raises(ValueError, match="same length"):
        store.add(doc, [chunk], embedder.embed_documents([]), [])


def test_dense_alone_confuses_near_duplicate_entities_and_hybrid_recovers(memory_store: MemoryStore) -> None:
    # The five municipal water guides differ mostly in the municipality name. A bag-of-n-grams vector sees
    # them as near-identical, so dense-only retrieval picks the wrong town; BM25 matches the rare name
    # token and fusion carries it to the top. This is the motivating case for hybrid search.
    retriever = Retriever(memory_store, HashingEmbedder(256), Analyzer())
    question = "Işıklıova Belediyesi'nde su güvence bedeli ne kadardır?"
    dense = retriever.search(question, RetrievalConfig(Mode.DENSE, k=10))
    hybrid = retriever.search(question, RetrievalConfig(Mode.HYBRID, k=10))
    assert dense[0].chunk.document_id != "isikliova-su-aboneligi"
    assert "isikliova-su-aboneligi" in {h.chunk.document_id for h in dense}
    assert hybrid[0].chunk.document_id == "isikliova-su-aboneligi"
