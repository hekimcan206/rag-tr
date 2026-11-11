## Suite `embedders`

Dense retrieval alone, per embedding model.

| run | chunks | words/chunk | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | R@300w | R@5 canonical | R@5 paraphrase | R@5 ascii |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense / hashing-384 | 191 | 34.7 | 0.489 | 0.797 | 0.898 | 0.615 | 0.683 | 0.880 | 0.855 | 0.677 | 0.860 |
| dense / minilm-l12 | 191 | 34.7 | 0.441 | 0.817 | 0.898 | 0.593 | 0.667 | 0.871 | 0.946 | 0.882 | 0.624 |
| dense / e5-small | 191 | 34.7 | 0.828 | 0.996 | 1.000 | 0.906 | 0.930 | 1.000 | 1.000 | 0.989 | 1.000 |
| dense / e5-base | 191 | 34.7 | 0.846 | 0.995 | 0.996 | 0.913 | 0.934 | 0.996 | 1.000 | 1.000 | 0.984 |

Run 2026-11-09T22:29:38+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [1.48, 0.48, 0.17] end [5.25, 2.42, 1.09] · wall 362.9 s · embedder `e5-small` · reranker `None` · generator `None`
