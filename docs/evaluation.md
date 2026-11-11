# Evaluation

Every number in the README and the ADRs comes from the result files in [`eval/results/`](../eval/results), which
the `ragtr eval` command writes as JSON (all metrics, environment, timings) and Markdown. This page explains how
they were produced and how to reproduce them.

## Data set

`ragtr generate-data` (seed 2026) writes [`data/corpus/*.md`](../data/corpus) and [`data/qa.jsonl`](../data/qa.jsonl).
A unit test checks that the committed files are byte-identical to the generator's output.

| | |
|---|---|
| Documents | 59 Markdown documents about 15 fictional entities: 5 municipalities × 5 service guides, 4 companies × 4 HR policies, 6 appliance models × 3 manual sections |
| Near-duplicates | Every entity has the same document types with different values (fees, days, phone numbers, error codes), so retrieval must find the right entity, not just the topic |
| Facts | 186, each with an answer span that is unique within its document |
| Questions | 602: 186 × {canonical, paraphrase, ascii} + 44 unanswerable (topics no document covers) |
| Chunks (default) | 191 structure+context chunks, 34.7 words on average |

**Relevance is mechanical.** A chunk is relevant to a question when it belongs to the gold document and contains the gold
answer span (with partial overlap as a fallback). Nobody labels anything, and the definition is the same for every chunking
strategy.

## Metrics

Retrieval, over the 558 answerable questions:

- **R@k**: the share of relevant chunks in the top k. Almost every question has exactly one relevant chunk, so this is the hit rate.
- **MRR@10** and **nDCG@10** (binary gain).
- **R@300w**: whether a relevant chunk is inside the top-ranked chunks that fit in 300 words. Strategies with
  different chunk sizes are compared at equal context cost (ADR 0002).

Answers (`ragtr/eval/metrics.py`, ADR 0004):

- **accuracy**: not an abstention, and contains every required normalised fact string. Abstentions count as wrong.
- **citation precision** and **supported**: whether the cited chunks contain the gold span.
- **groundedness**: the share of the answer's analysed terms found in the cited chunks. This is lexical, a proxy for faithfulness.
- **false abstention** (answerable questions refused) and **abstention** on the unanswerable questions.

## Environment

| | |
|---|---|
| Host | Intel Core i7-13620H (10 cores, 16 threads), 15.7 GB RAM, Windows 11 |
| Runtime | Docker Desktop 29.9.1, WSL2 VM with 16 vCPUs and 7.6 GB RAM; `ghcr.io/astral-sh/uv:0.9.5-python3.13-bookworm-slim` |
| Python stack | Python 3.13.9, PyTorch 2.15.0+cpu, sentence-transformers 6.2.0, transformers 5.22.2, NumPy 2.5.4 |
| Models | `intfloat/multilingual-e5-small` / `-base`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Load | The machine ran other containers; load averages at the start and end of each run are in the JSON files. Latencies are indicative, not benchmarks |

Retrieval metrics are deterministic: re-running a suite gives the same numbers, and CI checks this for the `ci` suite.

## Results

### Retrieval mode × reranker (`main`)

Retrieval mode x reranker on the default pipeline (structure+context chunks, Turkish analyser).

| run | chunks | words/chunk | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | R@300w | R@5 canonical | R@5 paraphrase | R@5 ascii |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 191 | 34.7 | 0.701 | 0.955 | 0.989 | 0.802 | 0.848 | 0.988 | 1.000 | 0.866 | 1.000 |
| dense | 191 | 34.7 | 0.828 | 0.996 | 1.000 | 0.906 | 0.930 | 1.000 | 1.000 | 0.989 | 1.000 |
| hybrid (RRF) | 191 | 34.7 | 0.776 | 0.998 | 1.000 | 0.870 | 0.903 | 1.000 | 1.000 | 0.995 | 1.000 |
| BM25 + rerank | 191 | 34.7 | 0.961 | 1.000 | 1.000 | 0.980 | 0.985 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense + rerank | 191 | 34.7 | 0.961 | 1.000 | 1.000 | 0.980 | 0.985 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid + rerank | 191 | 34.7 | 0.961 | 1.000 | 1.000 | 0.980 | 0.985 | 1.000 | 1.000 | 1.000 | 1.000 |

| contexts from | generator | n | accuracy | citation precision | supported | groundedness | false abstention | abstention (unanswerable) | median s |
|---|---|---|---|---|---|---|---|---|---|
| hybrid (RRF) | extractive | 602 | 0.473 | 0.996 | 0.996 | 1.000 | 0.523 | 1.000 | 0.001 |
| hybrid + rerank | extractive | 602 | 0.473 | 1.000 | 1.000 | 1.000 | 0.525 | 1.000 | 0.001 |

_Run 2026-11-11T09:22:34+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [1.76, 1.41, 1.29] end [9.69, 7.86, 4.53] · wall 464.8 s · embedder `e5-small`._

Answer rows use the extractive generator on all 602 questions.

### Chunking (`chunking`)

Chunking strategy x retrieval mode (no reranker).

| run | chunks | words/chunk | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | R@300w | R@5 canonical | R@5 paraphrase | R@5 ascii |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed / bm25 | 84 | 94.4 | 0.872 | 0.946 | 0.970 | 0.937 | 0.930 | 0.959 | 0.970 | 0.898 | 0.970 |
| fixed / dense | 84 | 94.4 | 0.845 | 0.951 | 0.986 | 0.918 | 0.925 | 0.944 | 0.979 | 0.938 | 0.935 |
| fixed / hybrid | 84 | 94.4 | 0.855 | 0.955 | 0.981 | 0.931 | 0.931 | 0.962 | 0.979 | 0.917 | 0.970 |
| sentence / bm25 | 73 | 101.8 | 0.909 | 0.977 | 0.982 | 0.936 | 0.948 | 0.964 | 1.000 | 0.930 | 1.000 |
| sentence / dense | 73 | 101.8 | 0.892 | 0.971 | 0.989 | 0.925 | 0.941 | 0.959 | 0.989 | 0.946 | 0.979 |
| sentence / hybrid | 73 | 101.8 | 0.909 | 0.977 | 0.993 | 0.939 | 0.952 | 0.968 | 0.989 | 0.946 | 0.995 |
| structure / bm25 | 191 | 34.7 | 0.158 | 0.597 | 0.801 | 0.326 | 0.438 | 0.710 | 0.758 | 0.269 | 0.763 |
| structure / dense | 191 | 34.7 | 0.124 | 0.572 | 0.846 | 0.297 | 0.425 | 0.778 | 0.720 | 0.430 | 0.565 |
| structure / hybrid | 191 | 34.7 | 0.143 | 0.582 | 0.891 | 0.323 | 0.455 | 0.783 | 0.720 | 0.339 | 0.688 |
| structure+context / bm25 | 191 | 34.7 | 0.701 | 0.955 | 0.989 | 0.802 | 0.848 | 0.988 | 1.000 | 0.866 | 1.000 |
| structure+context / dense | 191 | 34.7 | 0.828 | 0.996 | 1.000 | 0.906 | 0.930 | 1.000 | 1.000 | 0.989 | 1.000 |
| structure+context / hybrid | 191 | 34.7 | 0.776 | 0.998 | 1.000 | 0.870 | 0.903 | 1.000 | 1.000 | 0.995 | 1.000 |

_Run 2026-11-11T09:30:28+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [8.57, 7.69, 4.51] end [6.91, 7.34, 4.55] · wall 47.4 s · embedder `e5-small`._

### Normalisation ablation (`normalisation`)

Turkish-aware lexical normalisation, BM25 alone and inside hybrid retrieval.

| run | chunks | words/chunk | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | R@300w | R@5 canonical | R@5 paraphrase | R@5 ascii |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive (str.lower) | 191 | 34.7 | 0.509 | 0.837 | 0.932 | 0.634 | 0.706 | 0.914 | 0.995 | 0.581 | 0.935 |
| + Turkish casing | 191 | 34.7 | 0.514 | 0.844 | 0.939 | 0.640 | 0.712 | 0.925 | 0.995 | 0.586 | 0.952 |
| + apostrophe suffix | 191 | 34.7 | 0.516 | 0.844 | 0.941 | 0.643 | 0.714 | 0.934 | 0.995 | 0.586 | 0.952 |
| + stopwords | 191 | 34.7 | 0.511 | 0.844 | 0.934 | 0.634 | 0.706 | 0.923 | 1.000 | 0.608 | 0.925 |
| + Snowball stemmer | 191 | 34.7 | 0.588 | 0.918 | 0.970 | 0.702 | 0.767 | 0.968 | 1.000 | 0.796 | 0.957 |
| + ASCII folding | 191 | 34.7 | 0.695 | 0.932 | 0.979 | 0.787 | 0.833 | 0.977 | 1.000 | 0.796 | 1.000 |
| prefix-5 instead of Snowball (default) | 191 | 34.7 | 0.701 | 0.955 | 0.989 | 0.802 | 0.848 | 0.988 | 1.000 | 0.866 | 1.000 |
| Snowball, fold before stemming | 191 | 34.7 | 0.695 | 0.910 | 0.977 | 0.781 | 0.828 | 0.975 | 0.995 | 0.742 | 0.995 |
| hybrid, naive analyser | 191 | 34.7 | 0.663 | 0.955 | 0.996 | 0.784 | 0.837 | 0.995 | 1.000 | 0.898 | 0.968 |
| hybrid, Turkish analyser | 191 | 34.7 | 0.776 | 0.998 | 1.000 | 0.870 | 0.903 | 1.000 | 1.000 | 0.995 | 1.000 |

_Run 2026-11-11T09:31:23+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [6.43, 7.24, 4.54] end [5.54, 6.89, 4.52] · wall 37.2 s · embedder `e5-small`._

Rows above the default are cumulative; `Snowball, fold before stemming` is a variant of the Snowball row.

### Embedders (`embedders`)

Dense retrieval alone, per embedding model.

| run | chunks | words/chunk | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | R@300w | R@5 canonical | R@5 paraphrase | R@5 ascii |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense / hashing-384 | 191 | 34.7 | 0.489 | 0.797 | 0.898 | 0.615 | 0.683 | 0.880 | 0.855 | 0.677 | 0.860 |
| dense / minilm-l12 | 191 | 34.7 | 0.441 | 0.817 | 0.898 | 0.593 | 0.667 | 0.871 | 0.946 | 0.882 | 0.624 |
| dense / e5-small | 191 | 34.7 | 0.828 | 0.996 | 1.000 | 0.906 | 0.930 | 1.000 | 1.000 | 0.989 | 1.000 |
| dense / e5-base | 191 | 34.7 | 0.846 | 0.995 | 0.996 | 0.913 | 0.934 | 0.996 | 1.000 | 1.000 | 0.984 |

_Run 2026-11-09T22:29:38+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [1.48, 0.48, 0.17] end [5.25, 2.42, 1.09] · wall 362.9 s · embedder `e5-small`._

### Download-free regression suite (`ci`)

Download-free regression suite (hashing embedder, extractive generator).

| run | chunks | words/chunk | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | R@300w | R@5 canonical | R@5 paraphrase | R@5 ascii |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 191 | 34.7 | 0.701 | 0.955 | 0.989 | 0.802 | 0.848 | 0.988 | 1.000 | 0.866 | 1.000 |
| dense (hashing) | 191 | 34.7 | 0.489 | 0.797 | 0.898 | 0.615 | 0.683 | 0.880 | 0.855 | 0.677 | 0.860 |
| hybrid | 191 | 34.7 | 0.599 | 0.910 | 0.986 | 0.717 | 0.781 | 0.975 | 0.957 | 0.817 | 0.957 |
| BM25, naive analyser | 191 | 34.7 | 0.509 | 0.837 | 0.932 | 0.634 | 0.706 | 0.914 | 0.995 | 0.581 | 0.935 |
| hybrid / fixed | 84 | 94.4 | 0.797 | 0.949 | 0.979 | 0.895 | 0.903 | 0.946 | 0.970 | 0.906 | 0.970 |
| hybrid / sentence | 73 | 101.8 | 0.833 | 0.980 | 0.988 | 0.896 | 0.919 | 0.952 | 1.000 | 0.941 | 1.000 |
| hybrid / structure | 191 | 34.7 | 0.147 | 0.604 | 0.740 | 0.317 | 0.418 | 0.695 | 0.747 | 0.323 | 0.742 |

| contexts from | generator | n | accuracy | citation precision | supported | groundedness | false abstention | abstention (unanswerable) | median s |
|---|---|---|---|---|---|---|---|---|---|
| hybrid | extractive | 602 | 0.446 | 0.996 | 0.996 | 1.000 | 0.550 | 1.000 | 0.001 |

_Run 2026-11-11T09:20:59+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [2.07, 1.3, 1.24] end [2.07, 1.31, 1.25] · wall 2.2 s · embedder `hashing-384`._

CI re-runs this suite and fails on any difference from `eval/results/ci.json`.


## Reproduce

```bash
uv sync --extra ml
uv run ragtr eval --suite ci --embedder hashing-384                    # no download, seconds
uv run ragtr eval --suite main --embedder e5-small --reranker mmarco-minilm
uv run ragtr eval --suite chunking --embedder e5-small
uv run ragtr eval --suite normalisation --embedder e5-small
uv run ragtr eval --suite embedders --embedders hashing-384,minilm-l12,e5-small,e5-base --generator none
docker run -d -p 11434:11434 -v ollama:/root/.ollama ollama/ollama && docker exec <id> ollama pull gemma3:4b
uv run ragtr eval --suite generation --embedder e5-small --reranker mmarco-minilm \
    --generator ollama --ollama-model gemma3:4b --limit 120
```

## Caveats

- **Synthetic and small.** 191 chunks from template-generated documents. Strong embedders saturate R@5, so R@1, MRR
  and the weak-embedder runs carry the signal. Absolute numbers will be lower on real documents. The relative
  findings (normalisation ladder, header vs no header, reranker gain) are what this set can support.
- **Designed with knowledge of the set.** The analyser default (prefix-5 over Snowball) and the extractive
  generator's coverage rule (ADR 0001, commit history) were each changed once after seeing results on this set.
  Both changes are principled, and the alternatives' numbers are published, but there is no held-out split.
- **Answer metrics are string checks.** They are exact for these facts and blind to fluency. Groundedness is lexical.
- **No LLM answer metrics are published.** The `generation` suite with `gemma3:4b` on Ollama (CPU) was started three
  times on this machine. Each time the model runner was killed for memory inside a 7.6 GB Docker VM shared with other
  workloads, and the last attempt ended when the Docker engine stopped responding. The suite and the retry logic are in place
  (`ragtr eval --suite generation --generator ollama`), but no complete run exists, so no number is reported. Manual spot checks
  with gemma3:4b (a cited correct fee and a correct abstention) are anecdotes, not measurements.
