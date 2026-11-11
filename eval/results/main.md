## Suite `main`

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

Run 2026-11-11T09:22:34+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [1.76, 1.41, 1.29] end [9.69, 7.86, 4.53] · wall 464.8 s · embedder `e5-small` · reranker `mmarco-minilm` · generator `extractive`
