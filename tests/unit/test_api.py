from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ragtr.api.app import create_app
from ragtr.chunking import make_chunker
from ragtr.config import Components, Settings
from ragtr.embeddings import HashingEmbedder
from ragtr.generation import ABSTAIN, ExtractiveGenerator
from ragtr.models import Document
from ragtr.rerank import OverlapReranker
from ragtr.retrieval import Mode, RetrievalConfig
from ragtr.store.memory import MemoryStore
from ragtr.text.turkish import Analyzer


@pytest.fixture
def client(municipal_docs: list[Document]) -> Iterator[TestClient]:
    comps = Components(
        analyzer=Analyzer(),
        chunker=make_chunker("structure+context"),
        embedder=HashingEmbedder(256),
        reranker=OverlapReranker(),
        generator=ExtractiveGenerator(),
        retrieval=RetrievalConfig(Mode.HYBRID, k=5, rerank=True),
    )
    app = create_app(Settings(max_question_chars=200, max_document_chars=20_000), comps, MemoryStore())
    with TestClient(app) as c:
        for doc in municipal_docs:
            r = c.put(f"/v1/documents/{doc.id}", json={"title": doc.title, "text": doc.text})
            assert r.status_code == 200
        yield c


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["documents"] == 25
    assert ready["chunks"] > 25


def test_ask_returns_a_cited_answer(client: TestClient) -> None:
    r = client.post(
        "/v1/ask", json={"question": "Akçaburç Belediyesi'nde hacimli atıklar ayda kaç kez ücretsiz alınır?"}
    )
    body = r.json()
    assert r.status_code == 200
    assert not body["abstained"]
    assert body["citations"]
    first = body["citations"][0]
    assert first["document_id"] == "akcaburc-atik-toplama"
    assert body["answer"].endswith(f"[{first['marker']}]")
    assert set(body["timings_ms"]) == {"retrieve", "generate", "total"}


def test_ask_abstains_on_an_unanswerable_question(client: TestClient) -> None:
    body = client.post(
        "/v1/ask", json={"question": "Yeşilvadi'de otopark aylık abonman ücreti ne kadar?"}
    ).json()
    assert body["abstained"]
    assert body["answer"] == ABSTAIN
    assert body["citations"] == []


def test_search_modes_and_rerank_toggle(client: TestClient) -> None:
    for mode in ("dense", "bm25", "hybrid"):
        hits = client.post("/v1/search", json={"query": "camliyaka pazar yeri", "k": 3, "mode": mode,
                                               "rerank": False}).json()  # fmt: skip
        assert len(hits) == 3
        # Dense-only with the hashing test embedder finds the right topic but may pick a sibling town.
        assert hits[0]["document_id"].endswith("-semt-pazarlari")
        if mode != "dense":
            assert hits[0]["document_id"] == "camliyaka-semt-pazarlari"


def test_reingest_replaces_and_delete_removes(client: TestClient) -> None:
    before = client.get("/readyz").json()["chunks"]
    r = client.put("/v1/documents/ikizpinar-semt-pazarlari", json={"title": "Kısa", "text": "Tek cümle."})
    assert r.json() == {"document_id": "ikizpinar-semt-pazarlari", "chunks": 1}
    after = client.get("/readyz").json()
    assert after["documents"] == 25
    assert after["chunks"] < before
    assert client.delete("/v1/documents/ikizpinar-semt-pazarlari").status_code == 204
    assert client.delete("/v1/documents/ikizpinar-semt-pazarlari").status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "payload", "status"),
    [
        ("put", "/v1/documents/../etc", {"title": "t", "text": "x"}, 404),
        ("put", "/v1/documents/bad id", {"title": "t", "text": "x"}, 422),
        ("put", "/v1/documents/ok", {"title": "", "text": "x"}, 422),
        ("put", "/v1/documents/ok", {"title": "t", "text": "x" * 20_001}, 413),
        ("post", "/v1/ask", {"question": "x" * 201}, 422),
        ("post", "/v1/ask", {"question": ""}, 422),
        ("post", "/v1/search", {"query": "x", "k": 50}, 422),
    ],
)
def test_input_validation(
    client: TestClient, method: str, path: str, payload: dict[str, object], status: int
) -> None:
    assert getattr(client, method)(path, json=payload).status_code == status


def test_readiness_reports_503_when_the_store_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> int:
        raise ConnectionError("db down")

    monkeypatch.setattr(client.app.state.rag.store, "count_documents", boom)  # type: ignore[attr-defined]
    assert client.get("/readyz").status_code == 503


def test_openapi_lists_the_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/ask", "/v1/search", "/v1/documents/{document_id}", "/healthz", "/readyz"} <= set(paths)
