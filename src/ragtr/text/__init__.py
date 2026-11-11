"""Turkish text processing: normalisation, analysis for BM25, sentence splitting."""

from ragtr.text.sentences import Span, split_sentences
from ragtr.text.turkish import (
    NAIVE,
    TURKISH,
    Analyzer,
    AnalyzerConfig,
    Stemmer,
    fold_diacritics,
    tr_lower,
    tr_upper,
)

__all__ = [
    "NAIVE",
    "TURKISH",
    "Analyzer",
    "AnalyzerConfig",
    "Span",
    "Stemmer",
    "fold_diacritics",
    "split_sentences",
    "tr_lower",
    "tr_upper",
]
