# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-10

### Added

- Turkish text handling: locale-correct casing (`I`→`ı`, `İ`→`i`), apostrophe suffix splitting
  (`Ankara'da` → `ankara`), stopwords, Snowball and 5-character-prefix stemming, ASCII folding so
  questions typed without Turkish letters match; a sentence splitter that knows Turkish abbreviations,
  ordinal numbers, decimals, times and Markdown structure.
- Four chunking strategies (`fixed`, `sentence`, `structure`, `structure+context`), all with character
  offsets back into the source document.
- Embedders: deterministic hashing baseline and multilingual sentence-transformers models (E5 small/base,
  MiniLM-L12, bge-m3) with the correct `query:`/`passage:` prefixes; CPU-only PyTorch by default.
- Retrieval: BM25, dense, and hybrid with reciprocal rank fusion; optional cross-encoder reranking
  (`mmarco-minilm`, `bge-v2-m3`).
- Stores: in-memory reference store and PostgreSQL 17 + pgvector (HNSW) store with BM25 computed in SQL
  over a term table, checked to give identical scores to the reference implementation.
- Generation: local LLM through Ollama with numbered citations and a fixed abstention sentence, and a
  download-free extractive generator. The Ollama client retries transient 5xx and connection failures with
  bounded backoff.
- FastAPI service (`/v1/documents`, `/v1/search`, `/v1/ask`, health and readiness), structured JSON logs and
  OpenTelemetry traces; least-privilege database role; schema migration as a separate job.
- Seeded synthetic Turkish data set: 59 documents in near-duplicate entity families and 602 questions
  (canonical, paraphrase, ASCII-typed and unanswerable), byte-reproducible.
- Evaluation harness and CLI: recall@k, MRR, nDCG, a context-budget recall (R@300w), answer accuracy,
  citation precision, lexical groundedness and abstention rates; suites for retrieval mode × reranker,
  chunking, normalisation ablation, embedders and generation. A download-free suite is a CI regression gate.
- `docker compose up` demo (PostgreSQL, migrate, API, seed; opt-in Ollama profile), non-root image,
  CI with lint, strict typing, tests, dependency audit, image check, compose demo and CodeQL.

[1.0.0]: https://github.com/hekimcan206/rag-tr/releases/tag/v1.0.0
