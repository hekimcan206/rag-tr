"""PostgreSQL store: pgvector HNSW for dense search, BM25 computed in SQL over a term table.

Why BM25 in SQL instead of ``tsvector``/``ts_rank``: ``ts_rank`` is not BM25 (no IDF, no length
normalisation against the collection average), and PostgreSQL's ``turkish`` text-search configuration
lowercases ``I`` to ``i``, not ``ı``. Storing the analyser's terms in ``chunk_terms`` means the Python and
SQL paths index *identical* terms and score with the *identical* formula, which the integration tests
check to 1e-9. See ``docs/adr/0003-hybrid-retrieval-pgvector-bm25-rrf.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection, sql
from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

from ragtr.embeddings import Vector
from ragtr.lexical import K1, B
from ragtr.models import Chunk, Document

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ragtr_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id         text PRIMARY KEY CHECK (length(id) BETWEEN 1 AND 200),
    title      text NOT NULL,
    source     text NOT NULL DEFAULT '',
    indexed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           text PRIMARY KEY,
    document_id  text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ord          integer NOT NULL,
    text         text NOT NULL,
    body         text NOT NULL,
    start_offset integer NOT NULL,
    end_offset   integer NOT NULL,
    title        text NOT NULL DEFAULT '',
    section      text NOT NULL DEFAULT '',
    n_terms      integer NOT NULL CHECK (n_terms >= 0),
    embedding    vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS chunk_terms (
    term     text NOT NULL,
    chunk_id text NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    tf       integer NOT NULL CHECK (tf > 0),
    PRIMARY KEY (term, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunk_terms_chunk_idx ON chunk_terms (chunk_id);
"""

# Exact BM25 over every chunk that contains at least one query term; same formula as ragtr.lexical.
_BM25_SQL = """
WITH q AS (
    SELECT term, qtf FROM unnest(%(terms)s::text[], %(qtf)s::integer[]) AS u(term, qtf)
), stats AS (
    SELECT count(*)::float8 AS n, greatest(avg(n_terms), 1e-9)::float8 AS avgdl FROM chunks
), df AS (
    SELECT t.term, count(*)::float8 AS df FROM chunk_terms t JOIN q USING (term) GROUP BY t.term
)
SELECT t.chunk_id,
       sum(q.qtf * ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
           * t.tf * (%(k1)s + 1) / (t.tf + %(k1)s * (1 - %(b)s + %(b)s * c.n_terms / s.avgdl))) AS score
FROM chunk_terms t
JOIN q USING (term)
JOIN df USING (term)
JOIN chunks c ON c.id = t.chunk_id
CROSS JOIN stats s
GROUP BY t.chunk_id
ORDER BY score DESC, t.chunk_id
LIMIT %(k)s
"""


class IndexMismatchError(RuntimeError):
    """The database was indexed with a different embedder or analyser than the one configured."""


def migrate(conninfo: str, dim: int, embedder: str, analyzer: str, app_role: str | None = None) -> None:
    """Create the schema (idempotent) and record which embedder/analyser the index belongs to.

    Runs as the schema owner; the API connects as ``app_role`` which only receives DML rights.
    """
    if not 1 <= dim <= 16000:
        raise ValueError("embedding dimension out of range")
    with Connection.connect(conninfo, autocommit=True) as conn:
        conn.execute(sql.SQL(_SCHEMA.format(dim=int(dim))))  # int-formatted DDL, no user input
        conn.execute(
            "INSERT INTO ragtr_meta (key, value) VALUES ('schema_version', %s) ON CONFLICT (key) DO NOTHING",
            (SCHEMA_VERSION,),
        )
        _check_or_set_meta(conn, {"embedder": embedder, "dim": str(dim), "analyzer": analyzer})
        if app_role:
            role = sql.Identifier(app_role)
            conn.execute(sql.SQL("GRANT SELECT ON ragtr_meta TO {}").format(role))
            conn.execute(
                sql.SQL(
                    "GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks, chunk_terms TO {}"
                ).format(role)
            )


def _check_or_set_meta(conn: Connection[Any], expected: dict[str, str]) -> None:
    for key, value in expected.items():
        conn.execute(
            "INSERT INTO ragtr_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING", (key, value)
        )
    _verify_meta(conn, expected)


def _verify_meta(conn: Connection[Any], expected: dict[str, str]) -> None:
    rows = dict(conn.execute("SELECT key, value FROM ragtr_meta").fetchall())
    wrong = {k: (rows.get(k), v) for k, v in expected.items() if rows.get(k) != v}
    if wrong:
        detail = ", ".join(
            f"{k}: index has {have!r}, configured {want!r}" for k, (have, want) in wrong.items()
        )
        raise IndexMismatchError(f"re-index required ({detail})")


class PostgresStore:
    def __init__(
        self,
        conninfo: str,
        embedder: str,
        dim: int,
        analyzer: str,
        ef_search: int = 100,
        pool_size: int = 8,
    ) -> None:
        self._ef_search = int(ef_search)
        self._pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=pool_size,
            configure=self._configure,
            kwargs={"autocommit": False},
            open=True,
        )
        with self._pool.connection() as conn:
            _verify_meta(conn, {"embedder": embedder, "dim": str(dim), "analyzer": analyzer})

    @staticmethod
    def _configure(conn: Connection[Any]) -> None:
        register_vector(conn)
        conn.commit()

    def close(self) -> None:
        self._pool.close()

    def ping(self) -> bool:
        with self._pool.connection() as conn:
            return conn.execute("SELECT 1").fetchone() == (1,)

    def add(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Vector,
        terms: Sequence[Counter[str]],
    ) -> None:
        if len(chunks) != len(embeddings) or len(chunks) != len(terms):
            raise ValueError("chunks, embeddings and terms must have the same length")
        with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            # Replace semantics: re-ingesting a document swaps all its chunks in one transaction.
            cur.execute("DELETE FROM documents WHERE id = %s", (document.id,))
            cur.execute(
                "INSERT INTO documents (id, title, source) VALUES (%s, %s, %s)",
                (document.id, document.title, document.source),
            )
            cur.executemany(
                """INSERT INTO chunks (id, document_id, ord, text, body, start_offset, end_offset, title,
                                       section, n_terms, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        c.id,
                        c.document_id,
                        c.ord,
                        c.text,
                        c.body,
                        c.start,
                        c.end,
                        c.title,
                        c.section,
                        sum(tf.values()),
                        np.asarray(vec, dtype=np.float32),
                    )
                    for c, vec, tf in zip(chunks, embeddings, terms, strict=True)
                ],
            )
            cur.executemany(
                "INSERT INTO chunk_terms (term, chunk_id, tf) VALUES (%s, %s, %s)",
                [(term, c.id, n) for c, tf in zip(chunks, terms, strict=True) for term, n in tf.items()],
            )

    def delete_document(self, document_id: str) -> bool:
        with self._pool.connection() as conn, conn.transaction():
            cur = conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            return cur.rowcount > 0

    def dense_search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        vec = np.asarray(query, dtype=np.float32)
        with self._pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=tuple_row) as cur:
            # ef_search must be >= k or HNSW returns fewer than k rows.
            cur.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(max(self._ef_search, k))))
            cur.execute(
                """SELECT id, 1 - (embedding <=> %(q)s) AS sim
                   FROM chunks ORDER BY embedding <=> %(q)s, id LIMIT %(k)s""",
                {"q": vec, "k": k},
            )
            return [(str(r[0]), float(r[1])) for r in cur.fetchall()]

    def lexical_search(self, query: Counter[str], k: int) -> list[tuple[str, float]]:
        if not query:
            return []
        terms = sorted(query)
        with self._pool.connection() as conn, conn.transaction():
            rows = conn.execute(
                _BM25_SQL,
                {"terms": terms, "qtf": [query[t] for t in terms], "k1": K1, "b": B, "k": k},
            ).fetchall()
            return [(str(r[0]), float(r[1])) for r in rows]

    def get_chunks(self, ids: Sequence[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        with self._pool.connection() as conn, conn.transaction():
            rows = conn.execute(
                """SELECT id, document_id, ord, text, body, start_offset, end_offset, title, section
                   FROM chunks WHERE id = ANY(%s)""",
                (list(ids),),
            ).fetchall()
        return {r[0]: Chunk(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in rows}

    def count_chunks(self) -> int:
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute("SELECT count(*) FROM chunks").fetchone()
            return int(row[0]) if row else 0

    def count_documents(self) -> int:
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute("SELECT count(*) FROM documents").fetchone()
            return int(row[0]) if row else 0
