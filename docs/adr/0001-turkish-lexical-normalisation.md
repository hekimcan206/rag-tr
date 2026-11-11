# 0001 — Turkish lexical normalisation: locale casing, apostrophe suffixes, prefix-5 stems, ASCII folding

- Status: accepted
- Date: 2026-11-10

## Context

BM25 only matches terms that are spelled identically at index and query time. Turkish breaks the usual
English-centric pipeline in four places:

1. **Casing.** Turkish has dotted and dotless i as separate letters: `I`→`ı` and `İ`→`i`. Python's
   `str.lower()` maps `I`→`i` and turns `İ` into `i̇` (i plus a combining dot), so `IŞIKLIOVA` and
   `Işıklıova` index as different words, and `İzin` gains an invisible character. `casefold()` does the same, and
   the `locale` module does not change string methods.
2. **Proper nouns with case suffixes** are written with an apostrophe: `Işıklıova'da`, `Akçaburç'un`,
   `TL'dir`. Without handling, every inflected form of a name is a separate term.
3. **Agglutination.** One noun has dozens of surface forms (`ücret`, `ücreti`, `ücretin`, `ücretler`,
   `ücretlerinden` ...). Paraphrased questions rarely use the document's exact inflection.
4. **ASCII typing.** People type into search boxes without Turkish letters: `isikliova nikah ucreti`.

## Options considered

**Stemming**

1. **No stemming.** It is exact but brittle. On the paraphrase questions BM25 R@5 is 0.608 with stopwords and no stemming.
2. **Snowball Turkish** (affix stripping, Eryiğit & Adalı 2004 lineage). This is a dependency-free Python
   package. It strips one suffix layer at a time and is inconsistent on some lemmas: `izin`→`iz` but
   `izinler`→`izin` (pinned in `test_snowball_strips_one_suffix_layer_at_a_time`).
3. **Fixed 5-character prefix** (Can et al. 2008, *JASIST* 59(3), found prefix truncation competitive with
   morphological stemmers for Turkish IR). It is trivial and fast, and it is consistent by construction. It over-conflates
   different words that share five letters, and it splits roots shorter than five letters from their
   inflections (`atık`→`atik`, `atıklar`→`atikl`).
4. **Full morphological analysis** (Zemberek, TRmorph). It is the most accurate, but it is a JVM or FST dependency and
   slow at index time. Rejected for this scope.

**Order of folding and stemming**

Folding (`ş`→`s`, `ı`→`i` ...) before stemming lets ASCII-typed and correctly spelled words meet earlier, but
the Snowball rules are written for Turkish letters.

## Decision

The analyser (`ragtr.text.turkish.Analyzer`) applies, in order:

1. Unicode NFC (some PDFs encode `İ` as `I` + U+0307), then Turkish casing (`tr_lower`: `I`→`ı`, `İ`→`i`).
2. Apostrophe suffix removal: `Işıklıova'da` → `ışıklıova`.
3. Turkish stopword removal (a short list of function words and question words).
4. **Prefix-5 stemming**: numbers are kept whole.
5. ASCII folding after stemming.

The same analyser runs at index and query time. Its name (`trcase+apos+stop+prefix5+fold`) is stored in
`ragtr_meta`, and the PostgreSQL store refuses to open an index built with another analyser.

Snowball was the first default. The normalisation suite (BM25 alone, structure+context chunks, 602 questions)
replaced it with prefix-5:

| step | BM25 R@1 | R@5 | MRR@10 | R@5 paraphrase | R@5 ascii |
|---|---|---|---|---|---|
| naive `str.lower` | 0.509 | 0.837 | 0.634 | 0.581 | 0.935 |
| + Turkish casing | 0.514 | 0.844 | 0.640 | 0.586 | 0.952 |
| + apostrophe suffix | 0.516 | 0.844 | 0.643 | 0.586 | 0.952 |
| + stopwords | 0.511 | 0.844 | 0.634 | 0.608 | 0.925 |
| + Snowball | 0.588 | 0.918 | 0.702 | 0.796 | 0.957 |
| + ASCII folding (Snowball) | 0.695 | 0.932 | 0.787 | 0.796 | 1.000 |
| **prefix-5 instead of Snowball** | **0.701** | **0.955** | **0.802** | **0.866** | **1.000** |
| Snowball, fold before stemming | 0.695 | 0.910 | 0.781 | 0.742 | 0.995 |

Stemming contributes the largest single gain, mostly on paraphrases. ASCII folding is what makes ASCII-typed
questions work. Folding before Snowball hurts because Snowball's rules expect Turkish letters.

## Consequences

- Prefix-5 is now the index default. The extractive generator picks answer sentences with a Snowball analyser of its
  own, because one question matched against one sentence is where the short-root split
  (`atık`/`atıklar`) costs most (`ragtr/generation.py`).
- Prefix-5 will merge unrelated words that share a prefix (`karar` "decision" and `karartma` "blackout" both become `karar`). This is harmless here, and
  it is a known risk on larger vocabularies. Re-run the normalisation suite on any new corpus before trusting it.
- The ladder was measured on a synthetic corpus in three domains. The ranking of options is the finding, not the
  third decimal place.
- Changing the analyser requires re-indexing. `ragtr migrate` records the analyser, and a mismatch raises
  `IndexMismatchError` instead of silently returning poor results.
