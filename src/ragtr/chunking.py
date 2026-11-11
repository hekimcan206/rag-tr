"""Document chunking strategies.

Four strategies, all measured in ``docs/evaluation.md``:

* ``fixed`` — sliding window of N words with overlap, blind to sentences (the common baseline).
* ``sentence`` — Turkish sentence splitter, sentences packed up to a word budget, never split mid-sentence.
* ``structure`` — like ``sentence`` but never crosses a Markdown heading, so a chunk belongs to one section.
* ``structure+context`` — ``structure`` plus the document title and heading path prepended to the indexed
  text (``Akçaburç Belediyesi Su Aboneliği > Ücretler``). Chunks from near-identical documents of different
  entities are otherwise indistinguishable, because the entity name only appears in the title.

Sizes are counted in words, not model tokens, so chunk boundaries do not move when the embedding model
changes. Turkish averages ~1.6-1.9 subword tokens per word in multilingual tokenizers; 120 words stays
well below the 512-token limit of the E5 and MiniLM encoders.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ragtr.models import Chunk, Document
from ragtr.text.sentences import Span, split_sentences

_WORD = re.compile(r"\S+")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    STRUCTURE = "structure"
    STRUCTURE_CONTEXT = "structure+context"


class Chunker(Protocol):
    @property
    def name(self) -> str: ...

    def chunk(self, document: Document) -> list[Chunk]: ...


def _chunk_id(document_id: str, ord_: int) -> str:
    return f"{document_id}#{ord_:04d}"


@dataclass(frozen=True, slots=True)
class FixedChunker:
    """Sliding word window. ``overlap`` words are repeated between neighbouring chunks."""

    size: int = 120
    overlap: int = 20
    name: str = "fixed"

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be in [0, size)")

    def chunk(self, document: Document) -> list[Chunk]:
        words = [(m.start(), m.end()) for m in _WORD.finditer(document.text)]
        chunks: list[Chunk] = []
        step = self.size - self.overlap
        for i in range(0, max(len(words), 1), step):
            window = words[i : i + self.size]
            if not window:
                break
            start, end = window[0][0], window[-1][1]
            body = document.text[start:end]
            chunks.append(
                Chunk(_chunk_id(document.id, len(chunks)), document.id, len(chunks), body, body, start, end,
                      title=document.title)
            )  # fmt: skip
            if i + self.size >= len(words):
                break
        return chunks


@dataclass(frozen=True, slots=True)
class _Section:
    path: tuple[str, ...]
    start: int
    end: int


def _sections(text: str) -> list[_Section]:
    """Split Markdown into sections at headings; the heading line itself is not part of the section body."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    pos = 0
    path: tuple[str, ...] = ()
    for m in _HEADING.finditer(text):
        if text[pos : m.start()].strip():
            sections.append(_Section(path, pos, m.start()))
        level = len(m.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, m.group(2).strip()))
        path = tuple(h for _, h in stack)
        pos = m.end()
    if text[pos:].strip():
        sections.append(_Section(path, pos, len(text)))
    return sections


def _pack(spans: Sequence[Span], target: int, maximum: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Greedy packing of sentences into ranges of about ``target`` words (hard cap ``maximum``).

    A single sentence longer than ``maximum`` words is split on word boundaries rather than dropped.
    """
    units: list[tuple[int, int, int]] = []  # (start, end, words)
    for s in spans:
        words = [(m.start(), m.end()) for m in _WORD.finditer(s.text)]
        if len(words) <= maximum:
            units.append((s.start, s.end, len(words)))
            continue
        for i in range(0, len(words), maximum):
            part = words[i : i + maximum]
            units.append((s.start + part[0][0], s.start + part[-1][1], len(part)))
    i = 0
    while i < len(units):
        j, count = i, 0
        while j < len(units) and (j == i or count + units[j][2] <= target):
            count += units[j][2]
            j += 1
        # Absorb one more short unit rather than leaving a tiny orphan at the end of a section.
        if j == len(units) - 1 and count + units[j][2] <= maximum and units[j][2] < target // 4:
            j += 1
        yield units[i][0], units[j - 1][1]
        i = j - overlap if overlap and j - overlap > i else j


@dataclass(frozen=True, slots=True)
class SentenceChunker:
    target: int = 120
    maximum: int = 180
    overlap_sentences: int = 0
    name: str = "sentence"

    def chunk(self, document: Document) -> list[Chunk]:
        spans = split_sentences(document.text)
        chunks: list[Chunk] = []
        for start, end in _pack(spans, self.target, self.maximum, self.overlap_sentences):
            body = document.text[start:end]
            chunks.append(
                Chunk(_chunk_id(document.id, len(chunks)), document.id, len(chunks), body, body, start, end,
                      title=document.title)
            )  # fmt: skip
        return chunks


@dataclass(frozen=True, slots=True)
class StructureChunker:
    """Sentence packing inside Markdown sections; optionally prefixes the title and heading path."""

    target: int = 120
    maximum: int = 180
    contextual_headers: bool = True
    name: str = "structure+context"

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        title = document.title
        for section in _sections(document.text):
            path = section.path
            # The H1 usually repeats the title; drop it from the path to avoid "Title > Title".
            if path and path[0] == title:
                path = path[1:]
            heading = " > ".join(path)
            spans = [
                Span(s.start + section.start, s.end + section.start, s.text)
                for s in split_sentences(document.text[section.start : section.end])
            ]
            for start, end in _pack(spans, self.target, self.maximum, 0):
                body = document.text[start:end]
                if self.contextual_headers:
                    prefix = f"{title} > {heading}" if heading else title
                    text = f"{prefix}\n{body}"
                else:
                    text = body
                chunks.append(
                    Chunk(_chunk_id(document.id, len(chunks)), document.id, len(chunks), text, body,
                          start, end, title=title, section=heading)
                )  # fmt: skip
        return chunks


def make_chunker(strategy: ChunkingStrategy | str, target: int = 120) -> Chunker:
    match ChunkingStrategy(strategy):
        case ChunkingStrategy.FIXED:
            return FixedChunker(size=target, overlap=max(target // 6, 1))
        case ChunkingStrategy.SENTENCE:
            return SentenceChunker(target=target, maximum=target * 3 // 2)
        case ChunkingStrategy.STRUCTURE:
            return StructureChunker(target, target * 3 // 2, contextual_headers=False, name="structure")
        case ChunkingStrategy.STRUCTURE_CONTEXT:
            return StructureChunker(target, target * 3 // 2, contextual_headers=True)
