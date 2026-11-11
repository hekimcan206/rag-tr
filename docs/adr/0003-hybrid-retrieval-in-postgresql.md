# 0003 — Hybrid retrieval in one PostgreSQL: pgvector HNSW, BM25 in SQL, reciprocal rank fusion

- Status: accepted
- Date: 2026-11-10

## Context

Questions about institutional documents mix two kinds of signal:

- **Exact tokens**: an entity name (`Işıklıova`), an error code (`E-14`), a form name. These are rare terms that a
  lexical index matches precisely and a small embedding model blurs.
- **Meaning**: `depozito` for `güvence bedeli`, `ne zaman` for a weekday. A dense model matches these, and
  lexical search misses them unless the stemmer happens to help.

The store also has to be operated: re-ingesting a document must replace its chunks atomically, deletes must remove
vectors and terms together, and the API must run with least privilege.

## Options considered

**Lexical index**

1. **PostgreSQL full-text search** (`tsvector`, `ts_rank_cd`). It is built in, but `ts_rank` is not BM25: it
   has no IDF and no length normalisation against the collection average. The `turkish` configuration also
   lowercases `I` to `i` (not `ı`) and uses its own Snowball stemmer, which cannot be aligned with the analyser in
   ADR 0001.
2. **OpenSearch/Elasticsearch next to a vector database.** It gives real BM25 and an ICU Turkish analyser, but it
   adds two more stateful services, dual writes, and consistency between them on every re-ingest.
3. **ParadeDB `pg_search` or `pg_bm25`.** It gives BM25 inside PostgreSQL, but it needs a non-standard extension
   image and its own tokeniser.
4. **A term table plus BM25 in one SQL statement** (chosen). `chunk_terms(term, chunk_id, tf)` stores
   exactly the terms the Python analyser produced. The query computes Okapi BM25 (k1 = 1.2, b = 0.75), with
   document frequency from the term table and average length over all chunks, and scores every chunk that contains a query term. The same formula
   runs in `ragtr.lexical` for the in-memory store, and an integration test checks that both rank the same
   chunks with scores equal to 1e-9.

**Dense index**

pgvector 0.8 HNSW on cosine distance (`m = 16`, `ef_construction = 64`, `hnsw.ef_search = 100`, raised to
at least k per query). An integration test checks that approximate search returns the exact top hit and at
least 8 of the exact top 10.

**Fusion**

1. **Weighted score sum.** It needs score normalisation across two unrelated scales, and the weights have to be tuned.
2. **Reciprocal rank fusion** (Cormack, Clarke & Büttcher 2009): `Σ 1 / (60 + rank)`. It uses ranks only,
   has no tuned weights, and is robust when one retriever's scores drift (a swapped embedder).
3. **A cross-encoder over the fused candidates** (optional second stage). It reads question and chunk together; the
   multilingual mMiniLM model rescores the top 20.

## Decision

- One PostgreSQL 17 database with pgvector. The tables are `documents`, `chunks` (body, offsets, header, `n_terms`,
  `embedding vector(d)`) and `chunk_terms`, with `ON DELETE CASCADE`. Re-ingest is delete-then-insert in one
  transaction, so a failed ingest leaves the previous version in place (integration-tested).
- Default mode `hybrid`: the top 50 from each retriever, fused with RRF (k = 60), then the top 5 go to the generator.
  The cross-encoder is opt-in (`RAGTR_RERANKER`).
- `ragtr_meta` stores the embedder name, dimension and analyser. The store refuses to open on a mismatch.
- Roles: `ragtr_owner` runs DDL in the one-shot `migrate` job; the API connects as `ragtr_app` with DML only.

## Measurements

`main` suite (multilingual-e5-small, structure+context chunks, 558 answerable questions) and the download-free
`ci` suite (hashing embedder):

| first stage | e5-small R@1 | R@5 | MRR@10 | hashing R@1 | R@5 | MRR@10 |
|---|---|---|---|---|---|---|
| BM25 | 0.701 | 0.955 | 0.802 | 0.701 | 0.955 | 0.802 |
| dense | 0.828 | 0.996 | 0.906 | 0.489 | 0.797 | 0.615 |
| hybrid (RRF) | 0.776 | 0.998 | 0.870 | 0.599 | 0.910 | 0.717 |
| hybrid + cross-encoder | 0.961 | 1.000 | 0.980 | – | – | – |

- Which single retriever wins depends on the embedder: dense with E5, BM25 with the hashing baseline. Hybrid is never the worst
  and has the best E5 recall at 5, but equal-weight RRF ranks below dense on R@1 when the embedder is strong.
- The cross-encoder over the fused top 20 removes that gap (R@1 0.961). It costs about 27 ms per pair on CPU,
  roughly 0.55 s per question.
- The near-duplicate case that motivated hybrid search is pinned by a unit test: with the hashing embedder, dense-only picks the
  wrong municipality, and BM25's match on the rare name carries the right one to the top after fusion.

## Consequences

- One stateful service to back up, secure and run in compose and CI. Atomic re-ingest and deletes come
  free from transactions.
- BM25 in SQL scans the postings of the query terms. That is fine at thousands of chunks, but at millions a dedicated
  inverted index (ParadeDB, OpenSearch) or a materialised IDF table would be needed. Measuring that was out of
  scope for a 191-chunk corpus.
- Analyser and index are coupled: any analyser change requires re-indexing, and the meta check enforces it.
- RRF with equal weights is not always better than the best single retriever (see Measurements). A weighted
  RRF or a learned fusion should be tuned on a held-out split, never on the evaluation set.
