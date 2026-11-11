# 0002 — Structure-aware chunking with contextual headers, compared at equal context cost

- Status: accepted
- Date: 2026-11-10

## Context

Institutional knowledge bases (municipal service guides, HR policies, appliance manuals) share two
properties that decide how chunking must work:

- **They are structured.** A guide has a title, sections (`## Ücretler`, `## Başvuru`) and short
  paragraphs or list items. An answer almost always lives inside one section.
- **They come in near-duplicate families.** Every municipality publishes the same "water subscription guide" with
  different numbers. The entity name (`Işıklıova`) appears in the title and perhaps in the first paragraph, but
  not in the section that holds the deposit amount. A chunk cut from the middle of the document can be
  indistinguishable from four sibling chunks of other towns.

Turkish adds one constraint. Sentence boundaries are hard because `3. kat`, `19. yüzyıl` and `Dr.` contain
periods, so chunkers that split on `. ` cut sentences in half (see `ragtr/text/sentences.py`).

## Options considered

1. **Fixed word window with overlap** (`fixed`: 120 words, 20 overlap). This is the common baseline. It is blind to
   sentences and sections, and chunks straddle topics.
2. **Sentence packing** (`sentence`). Turkish-aware sentences are packed greedily up to about 120 words. It never cuts a
   sentence, but it still crosses section boundaries.
3. **Section-bounded sentence packing** (`structure`). It never crosses a heading. Chunks are small (35 words on
   average here) and on-topic, but they lose the entity name.
4. **Section-bounded packing with a contextual header** (`structure+context`). Each chunk's indexed text is
   `Title > Section path` followed by the body. The header is embedded and BM25-indexed, but the stored body
   and its character offsets stay those of the source, so citations point at real text. This is the same idea
   as Anthropic's "contextual retrieval", without an LLM writing the context: the document structure already
   holds it.

Chunk sizes are counted in words, not model tokens, so boundaries do not move when the embedding model changes.

## How the options were compared

Recall@k at a fixed k favours big chunks. With roughly 100-word chunks, a top-5 list holds 500 words of
context; with 35-word chunks it holds 175. The harness therefore also reports **R@300w**: whether the answer
span is inside the top-ranked chunks that fit in a 300-word context budget (`WORD_BUDGET`). This is roughly what
a small local model receives, and it compares strategies at equal cost to the generator.

Download-free `ci` suite (hashing embedder, hybrid retrieval, 558 answerable questions):

| chunking | chunks | words/chunk | R@1 | R@5 | MRR@10 | R@300w |
|---|---|---|---|---|---|---|
| fixed | 84 | 94.4 | 0.797 | 0.949 | 0.895 | 0.946 |
| sentence | 73 | 101.8 | 0.833 | 0.980 | 0.896 | 0.952 |
| structure | 191 | 34.7 | 0.147 | 0.604 | 0.317 | 0.695 |
| **structure+context** | 191 | 34.7 | 0.599 | 0.910 | 0.717 | **0.975** |

With multilingual E5 the picture is the same; see `docs/evaluation.md` for the full chunking × mode table.

## Decision

`structure+context` is the default (`RAGTR_CHUNKING`), with a target of 120 and a maximum of 180 words per chunk.

## Consequences

- At a fixed k, the larger fixed and sentence chunks look better on R@1. At an equal context budget,
  structure+context retrieves the answer most often. The first comparison mostly measures chunk size, and the second
  is the one that matters for generation. Both are published so the trade-off is visible.
- Plain `structure` chunking is the worst option on near-duplicate families (R@1 0.147). The header, not the
  section split, is what makes small chunks work. This is the clearest result in the chunking suite.
- Answers are cited at section granularity, and `/v1/ask` returns title and section for every citation.
- Documents without Markdown headings degrade to sentence packing under the title, which is still better than a blind
  window.
- The header repeats the title in every chunk. This adds index size and slightly inflates BM25 document frequency
  for title words, which is acceptable at this scale.
