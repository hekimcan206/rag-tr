"""Turkish-aware text normalisation and analysis for lexical retrieval.

Three properties of Turkish break the defaults of most IR toolkits:

* **Casing.** Turkish has dotted ``İ/i`` and dotless ``I/ı``. Python's ``str.lower`` follows the
  language-neutral Unicode mapping, so ``"İSTANBUL".lower()`` is ``"i̇stanbul"`` (``i`` + U+0307
  COMBINING DOT ABOVE) and ``"IŞIK".lower()`` is ``"işik"`` instead of ``"ışık"``. Neither matches the
  same word typed in lowercase. ``casefold`` does not help, and the ``locale`` module does not affect
  ``str`` methods.
* **Diacritics in queries.** Users often type without Turkish letters (``"yillik izin kac gun"``), while
  documents are written correctly. Folding ``ç ğ ı ö ş ü`` to ASCII on both sides makes the two meet.
* **Agglutination.** One lemma surfaces in many forms (``izin, izni, izinler, izninizi …``) and proper
  nouns carry suffixes after an apostrophe (``Akçaburç'taki``). Stemming or prefix truncation collapses
  the forms; the apostrophe suffix carries no topical information and is dropped.

See ``docs/adr/0001-turkish-text-normalisation.md`` for the measured effect of each step.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import snowballstemmer

_UPPER_TO_LOWER = str.maketrans({"İ": "i", "I": "ı"})
_LOWER_TO_UPPER = str.maketrans({"i": "İ", "ı": "I"})
_ASCII_FOLD = str.maketrans(
    {
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
        "â": "a", "î": "i", "û": "u",
        "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
        "Â": "A", "Î": "I", "Û": "U",
    }
)  # fmt: skip
_APOSTROPHES = "'’ʼ`´"

# A token is a number (with Turkish thousands/decimal separators), a code such as ``E-14`` or ``K-200``,
# or a word that may carry an apostrophe suffix (``Akçaburç'taki``).
_TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)*"
    r"|[^\W\d_]+(?:[-/]\w+)+"
    r"|[^\W_]+(?:[" + _APOSTROPHES + r"][^\W\d_]+)?"
)

# High-frequency function words. Deliberately short: content words such as "gün" or "saat" stay, because
# questions about durations and opening hours depend on them.
TURKISH_STOPWORDS: frozenset[str] = frozenset(
    [
        "acaba",
        "ama",
        "ancak",
        "artık",
        "aslında",
        "az",
        "bana",
        "bazen",
        "bazı",
        "belki",
        "ben",
        "beni",
        "benim",
        "bile",
        "bir",
        "birkaç",
        "biri",
        "birçok",
        "bu",
        "buna",
        "bunda",
        "bundan",
        "bunu",
        "bunun",
        "burada",
        "çok",
        "çünkü",
        "da",
        "daha",
        "de",
        "defa",
        "diye",
        "en",
        "gibi",
        "hangi",
        "hangisi",
        "hem",
        "hep",
        "hepsi",
        "her",
        "hiç",
        "için",
        "ile",
        "ilgili",
        "ise",
        "işte",
        "kadar",
        "kendi",
        "ki",
        "kim",
        "kimi",
        "kimin",
        "mi",
        "mı",
        "mu",
        "mü",
        "nasıl",
        "ne",
        "neden",
        "nerde",
        "nerede",
        "nereye",
        "niye",
        "niçin",
        "o",
        "olan",
        "olarak",
        "oldu",
        "olduğu",
        "olsa",
        "ona",
        "onda",
        "ondan",
        "onlar",
        "onu",
        "onun",
        "öyle",
        "sadece",
        "sanki",
        "sen",
        "siz",
        "şey",
        "şu",
        "şuna",
        "şunda",
        "şunu",
        "tüm",
        "ve",
        "veya",
        "ya",
        "yani",
        "yine",
    ]
)


def tr_lower(text: str) -> str:
    """Lowercase with Turkish rules: ``İ → i`` and ``I → ı``.

    NFC first so a decomposed ``I`` + U+0307 (how some PDFs encode ``İ``) becomes ``İ`` before mapping.
    """
    return unicodedata.normalize("NFC", text).translate(_UPPER_TO_LOWER).lower()


def tr_upper(text: str) -> str:
    """Uppercase with Turkish rules: ``i → İ`` and ``ı → I``."""
    return unicodedata.normalize("NFC", text).translate(_LOWER_TO_UPPER).upper()


def fold_diacritics(text: str) -> str:
    """Map Turkish letters (and circumflex vowels) to ASCII: ``ç ğ ı ö ş ü â î û → c g i o s u a i u``."""
    return text.translate(_ASCII_FOLD)


def strip_apostrophe_suffix(token: str) -> str:
    """``akçaburç'taki → akçaburç``. Turkish writes inflections of proper nouns after an apostrophe."""
    for mark in _APOSTROPHES:
        head, sep, _ = token.partition(mark)
        if sep:
            return head
    return token


def raw_tokens(text: str) -> list[str]:
    """Split into number, code and word tokens without changing case."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text))


class Stemmer(StrEnum):
    NONE = "none"
    SNOWBALL = "snowball"  # Snowball Turkish suffix stripper (Eryiğit & Adalı 2004 lineage)
    PREFIX5 = "prefix5"  # fixed 5-character prefix (Can et al. 2008, JASIST 59(3))


@lru_cache(maxsize=1)
def _snowball() -> snowballstemmer.TurkishStemmer:
    return snowballstemmer.stemmer("turkish")


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    """Lexical analysis pipeline. Every flag is an ablation knob in the evaluation."""

    turkish_case: bool = True
    apostrophe: bool = True
    stopwords: bool = True
    stemmer: Stemmer = Stemmer.PREFIX5
    fold: bool = True
    fold_before_stem: bool = False

    @property
    def name(self) -> str:
        if self == NAIVE:
            return "naive"
        parts = [
            "trcase" if self.turkish_case else "lower",
            "apos" if self.apostrophe else "",
            "stop" if self.stopwords else "",
            self.stemmer.value if self.stemmer is not Stemmer.NONE else "",
            ("fold<stem" if self.fold_before_stem else "fold") if self.fold else "",
        ]
        return "+".join(p for p in parts if p)


#: What a generic, language-neutral pipeline does: ``str.lower`` and whitespace/punctuation tokens.
NAIVE = AnalyzerConfig(
    turkish_case=False, apostrophe=False, stopwords=False, stemmer=Stemmer.NONE, fold=False
)
#: The default: Turkish casing, apostrophe suffixes dropped, stopwords removed, 5-character prefixes, ASCII-folded.
#: Prefix-5 replaced Snowball after the normalisation suite (see docs/adr/0001-turkish-lexical-normalisation.md).
TURKISH = AnalyzerConfig()


class Analyzer:
    """Turns text into index terms. The same analyzer must be used at index and at query time."""

    def __init__(self, config: AnalyzerConfig = TURKISH) -> None:
        self.config = config
        self._stem_cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self.config.name

    def normalise(self, text: str) -> str:
        """Case (and optionally diacritic) normalisation without tokenising; used for span matching."""
        cfg = self.config
        out = tr_lower(text) if cfg.turkish_case else text.lower()
        return fold_diacritics(out) if cfg.fold else out

    def terms(self, text: str) -> list[str]:
        cfg = self.config
        result: list[str] = []
        for raw in raw_tokens(text):
            token = tr_lower(raw) if cfg.turkish_case else raw.lower()
            if cfg.apostrophe:
                token = strip_apostrophe_suffix(token)
            elif any(m in token for m in _APOSTROPHES):
                # Without apostrophe handling the suffix is just split off as a separate token.
                result.extend(self._finish(t) for t in re.split(f"[{_APOSTROPHES}]", token) if t)
                continue
            if cfg.stopwords and token in TURKISH_STOPWORDS:
                continue
            result.append(self._finish(token))
        return result

    def term_counts(self, text: str) -> Counter[str]:
        return Counter(self.terms(text))

    def _finish(self, token: str) -> str:
        cached = self._stem_cache.get(token)
        if cached is not None:
            return cached
        cfg = self.config
        out = token
        if cfg.fold and cfg.fold_before_stem:
            out = fold_diacritics(out)
        if not out[:1].isdigit():
            if cfg.stemmer is Stemmer.SNOWBALL:
                out = _snowball().stemWord(out)
            elif cfg.stemmer is Stemmer.PREFIX5:
                out = out[:5]
        if cfg.fold and not cfg.fold_before_stem:
            out = fold_diacritics(out)
        if len(self._stem_cache) < 200_000:
            self._stem_cache[token] = out
        return out


def content_terms(analyzer: Analyzer, texts: Iterable[str]) -> set[str]:
    """Union of analysed terms over several texts (used by the extractive generator and metrics)."""
    terms: set[str] = set()
    for text in texts:
        terms.update(analyzer.terms(text))
    return terms
