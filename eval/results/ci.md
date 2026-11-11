## Suite `ci`

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

Run 2026-11-11T09:20:59+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [2.07, 1.3, 1.24] end [2.07, 1.31, 1.25] · wall 2.2 s · embedder `hashing-384` · reranker `None` · generator `extractive`
