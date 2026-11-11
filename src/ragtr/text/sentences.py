"""Rule-based Turkish sentence splitter.

A period in Turkish text often does not end a sentence:

* **Ordinals** are written as digits plus a period: ``3. kat`` (3rd floor), ``19. yüzyıl``, ``2. madde``.
  A splitter that treats every ``digit + period`` as a boundary cuts these sentences in half.
* **Abbreviations**: ``Dr.``, ``Prof.``, ``vb.``, ``vs.``, ``örn.``, ``bkz.``, ``No.``, ``Mah.``, ``Cad.``,
  ``Sok.``, ``T.C.``, ``A.Ş.``, ``Ltd. Şti.``
* **Initials**: ``M. Kemal``.

The rule set: a candidate boundary is ``[.!?…]`` followed by whitespace. It is rejected when the token
before it is a known abbreviation or a single-letter initial, or when it is a number and the next word starts
in lowercase (an ordinal). Blank lines, Markdown headings and list items always start a new sentence.
This trades the last percent of accuracy of statistical splitters (Punkt, Zemberek's perceptron) for zero
dependencies and explainable behaviour; the golden tests in ``tests/unit/test_sentences.py`` pin it down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragtr.text.turkish import tr_lower

ABBREVIATIONS: frozenset[str] = frozenset(
    [
        "a.ş",
        "alb",
        "apt",
        "av",
        "bkz",
        "blv",
        "bşk",
        "cad",
        "cd",
        "doç",
        "dr",
        "gen",
        "gn",
        "hz",
        "kom",
        "koop",
        "ltd",
        "mad",
        "mah",
        "md",
        "mim",
        "muh",
        "no",
        "nr",
        "ord",
        "örn",
        "öğr",
        "prof",
        "s",
        "sf",
        "sn",
        "sok",
        "sk",
        "st",
        "şti",
        "t.c",
        "tel",
        "tic",
        "vb",
        "vd",
        "vs",
        "yay",
        "yrd",
        "yy",
        "gör",
        "har",
        "müd",
        "yük",
    ]
)

_CANDIDATE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")
_STRUCTURAL_LINE = re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\|)")
_WORD_BEFORE = re.compile(r"([^\s(\[\"'“]+)$")


@dataclass(frozen=True, slots=True)
class Span:
    """A sentence as a half-open character range into the source text."""

    start: int
    end: int
    text: str


def _is_boundary(text: str, dot_start: int, after: int) -> bool:
    before = _WORD_BEFORE.search(text, 0, dot_start)
    word = before.group(1) if before else ""
    nxt = text[after:].lstrip()
    if not nxt:
        return True
    if text[dot_start] != ".":
        return True  # ! ? … always end a sentence when followed by whitespace
    key = tr_lower(word).rstrip(".")
    if key in ABBREVIATIONS:
        return False
    if len(word) == 1 and word.isalpha() and word.isupper():
        return False  # initial: "M. Kemal"
    if word.isdigit() and nxt[:1].islower():
        return False  # ordinal: "3. kat", "19. yüzyıl"
    return not nxt[:1].islower()


def split_sentences(text: str) -> list[Span]:
    """Split ``text`` into sentence spans; offsets index into the original string."""
    spans: list[Span] = []
    line_start = 0
    for line in text.splitlines(keepends=True):
        _split_line(text, line_start, line_start + len(line.rstrip("\r\n")), spans)
        line_start += len(line)
    return spans


def _split_line(text: str, start: int, end: int, out: list[Span]) -> None:
    line = text[start:end]
    if not line.strip():
        return
    # A structural line (heading, list item, table row) is its own unit, and a line that does not start a
    # new structure continues the previous sentence only when that sentence did not end.
    continues = (
        out
        and not _STRUCTURAL_LINE.match(line)
        and not out[-1].text.rstrip().endswith((".", "!", "?", "…", ":"))
        and not _STRUCTURAL_LINE.match(out[-1].text)
        and text[out[-1].end : start].count("\n") == 1
    )
    cursor = start
    if continues:
        prev = out.pop()
        cursor = prev.start
    if _STRUCTURAL_LINE.match(line) and line.lstrip().startswith(("#", "|")):
        out.append(_make(text, cursor, end))
        return
    marker = _STRUCTURAL_LINE.match(line)
    search_from = start + marker.end() if marker else start  # "1. Başvuru …" — the list number is no ordinal
    for match in _CANDIDATE.finditer(text, search_from, end):
        dot_start, after = match.start(), match.end()
        if _is_boundary(text[:end], dot_start, after):
            out.append(_make(text, cursor, after))
            cursor = after
    if text[cursor:end].strip():
        out.append(_make(text, cursor, end))


def _make(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end, text[start:end])
