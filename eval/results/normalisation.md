## Suite `normalisation`

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

Run 2026-11-11T09:31:23+00:00 · Python 3.13.9 · 16 CPUs visible · load avg start [6.43, 7.24, 4.54] end [5.54, 6.89, 4.52] · wall 37.2 s · embedder `e5-small` · reranker `mmarco-minilm` · generator `extractive`
