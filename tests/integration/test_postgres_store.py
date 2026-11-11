"""PostgreSQL + pgvector store against a real database (Testcontainers)."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from ragtr.api.app import create_app
from ragtr.chunking import make_chunker
from ragtr.config import Components, Settings
from ragtr.embeddings import HashingEmbedder
from ragtr.generation import ExtractiveGenerator
from ragtr.models import Document
from ragtr.pipeline import Indexer
from ragtr.retrieval import Mode, RetrievalConfig, Retriever
from ragtr.store.memory import MemoryStore
from ragtr.store.postgres import IndexMismatchError, PostgresStore, migrate
from ragtr.text.turkish import Analyzer

pytestmark = pytest.mark.integration

IMAGE = "pgvector/pgvector:0.8.1-pg17"
DIM = 256
EMBEDDER = HashingEmbedder(DIM)
ANALYZER = Analyzer()


@pytest.fixture(scope="module")
def admin_url() -> Iterator[str]:
    with PostgresContainer(IMAGE, username="owner", password="owner-pw", dbname="ragtr", driver=None) as pg:
        url = pg.get_connection_url()
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("CREATE ROLE ragtr_app LOGIN PASSWORD 'app-pw'")
        migrate(url, DIM, EMBEDDER.name, ANALYZER.name, app_role="ragtr_app")
        yield url


@pytest.fixture(scope="module")
def app_url(admin_url: str) -> str:
    return admin_url.replace("owner:owner-pw", "ragtr_app:app-pw")


@pytest.fixture(scope="module")
def stores(app_url: str, municipal_docs: list[Document]) -> Iterator[tuple[PostgresStore, MemoryStore]]:
    pg = PostgresStore(app_url, EMBEDDER.name, DIM, ANALYZER.name, ef_search=200)
    mem = MemoryStore()
    chunker = make_chunker("structure+context")
    for store in (pg, mem):
        indexer = Indexer(chunker, EMBEDDER, ANALYZER, store)
        for doc in municipal_docs:
            indexer.index(doc)
    yield pg, mem
    pg.close()


QUERIES = [
    "Akçaburç Belediyesi'nde yapı ruhsatı için projeler kaç takım teslim edilir?",
    "isikliova nikah hafta sonu ucreti",
    "Çamlıyaka'da geri dönüşüm atıkları hangi günler toplanıyor?",
    "su sayacı kaç iş günü içinde takılır",
]


@pytest.mark.parametrize("query", QUERIES)
def test_sql_bm25_equals_the_in_memory_reference(
    stores: tuple[PostgresStore, MemoryStore], query: str
) -> None:
    pg, mem = stores
    terms = ANALYZER.term_counts(query)
    # Every matching chunk (no cut-off), so a group of tied scores can never straddle the limit.
    expected = mem.lexical_search(terms, 1000)
    actual = pg.lexical_search(terms, 1000)

    # Near-duplicate documents tie on BM25. PostgreSQL and NumPy sum in a different order, so tied scores can
    # differ in the 15th digit and swap places. Compare rankings with ties resolved by id, scores to 1e-9.
    def ranking(hits: list[tuple[str, float]]) -> list[str]:
        return [c for c, _ in sorted(hits, key=lambda h: (-round(h[1], 9), h[0]))]

    assert ranking(actual) == ranking(expected)
    scores = dict(expected)
    for chunk_id, score in actual:
        assert math.isclose(score, scores[chunk_id], rel_tol=1e-9)


@pytest.mark.parametrize("query", QUERIES)
def test_hnsw_dense_search_agrees_with_exact_search_at_the_top(
    stores: tuple[PostgresStore, MemoryStore], query: str
) -> None:
    pg, mem = stores
    q = EMBEDDER.embed_query(query)
    exact = mem.dense_search(q, 10)
    approx = pg.dense_search(q, 10)
    assert approx[0][0] == exact[0][0]
    assert len({c for c, _ in approx} & {c for c, _ in exact}) >= 8
    assert math.isclose(approx[0][1], exact[0][1], abs_tol=1e-5)


def test_hybrid_retrieval_end_to_end_on_postgres(stores: tuple[PostgresStore, MemoryStore]) -> None:
    pg, _ = stores
    hits = Retriever(pg, EMBEDDER, ANALYZER).search(QUERIES[0], RetrievalConfig(Mode.HYBRID, k=3))
    assert hits[0].chunk.document_id == "akcaburc-yapi-ruhsati"
    assert hits[0].ranks.keys() == {"dense", "bm25"}


def test_reingest_replaces_chunks_atomically(app_url: str) -> None:
    store = PostgresStore(app_url, EMBEDDER.name, DIM, ANALYZER.name)
    try:
        indexer = Indexer(make_chunker("sentence", 20), EMBEDDER, ANALYZER, store)
        long = Document("tmp-doc", "Geçici", " ".join(f"Cümle numarası {i} burada." for i in range(40)))
        n_long = len(indexer.index(long))
        n_short = len(indexer.index(Document("tmp-doc", "Geçici", "Tek cümle.")))
        assert n_long > 1
        assert n_short == 1
        with psycopg.connect(app_url) as conn:
            row = conn.execute("SELECT count(*) FROM chunks WHERE document_id = 'tmp-doc'").fetchone()
            terms = conn.execute(
                "SELECT count(*) FROM chunk_terms t JOIN chunks c ON c.id = t.chunk_id "
                "WHERE c.document_id = 'tmp-doc'"
            ).fetchone()
        assert row == (1,)
        assert terms is not None
        assert terms[0] == len(ANALYZER.term_counts("Tek cümle."))
        assert store.delete_document("tmp-doc")
        assert not store.delete_document("tmp-doc")
    finally:
        store.close()


def test_a_failed_ingest_leaves_the_previous_version_in_place(app_url: str) -> None:
    store = PostgresStore(app_url, EMBEDDER.name, DIM, ANALYZER.name)
    try:
        doc = Document("atomic-doc", "Atomik", "Birinci sürüm.")
        Indexer(make_chunker("sentence"), EMBEDDER, ANALYZER, store).index(doc)
        chunks = make_chunker("sentence").chunk(doc)
        bad_vectors = np.zeros((1, DIM + 1), dtype=np.float32)  # wrong dimension: the INSERT fails
        with pytest.raises(psycopg.Error):
            store.add(
                Document("atomic-doc", "Atomik", "İkinci."), chunks, bad_vectors, [ANALYZER.term_counts("x")]
            )
        assert store.get_chunks(["atomic-doc#0000"])["atomic-doc#0000"].body == "Birinci sürüm."
    finally:
        store.close()


def test_app_role_cannot_change_the_schema(app_url: str) -> None:
    with psycopg.connect(app_url) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("DROP TABLE chunk_terms")
    with psycopg.connect(app_url) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("UPDATE ragtr_meta SET value = 'x' WHERE key = 'embedder'")


def test_opening_an_index_with_a_different_embedder_or_analyser_fails(app_url: str, admin_url: str) -> None:
    with pytest.raises(IndexMismatchError, match="embedder"):
        PostgresStore(app_url, "e5-small", DIM, ANALYZER.name)
    with pytest.raises(IndexMismatchError, match="analyzer"):
        PostgresStore(app_url, EMBEDDER.name, DIM, "naive")
    with pytest.raises(IndexMismatchError):
        migrate(admin_url, 384, "e5-small", ANALYZER.name)


def test_migrate_is_idempotent(admin_url: str) -> None:
    migrate(admin_url, DIM, EMBEDDER.name, ANALYZER.name, app_role="ragtr_app")
    migrate(admin_url, DIM, EMBEDDER.name, ANALYZER.name, app_role="ragtr_app")
    with pytest.raises(ValueError, match="dimension"):
        migrate(admin_url, 0, EMBEDDER.name, ANALYZER.name)


def test_api_on_postgres(app_url: str, stores: tuple[PostgresStore, MemoryStore]) -> None:
    comps = Components(ANALYZER, make_chunker("structure+context"), EMBEDDER, None, ExtractiveGenerator(),
                       RetrievalConfig(Mode.HYBRID, k=5, rerank=False))  # fmt: skip
    settings = Settings(database_url=app_url, embedder=EMBEDDER.name)  # type: ignore[arg-type]
    with TestClient(create_app(settings, comps)) as client:
        assert client.get("/readyz").json()["documents"] >= 25
        body = client.post("/v1/ask", json={"question": QUERIES[0]}).json()
        assert not body["abstained"]
        assert body["citations"][0]["document_id"] == "akcaburc-yapi-ruhsati"
