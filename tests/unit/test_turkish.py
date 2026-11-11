from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragtr.text.turkish import (
    NAIVE,
    TURKISH,
    Analyzer,
    AnalyzerConfig,
    Stemmer,
    fold_diacritics,
    raw_tokens,
    strip_apostrophe_suffix,
    tr_lower,
    tr_upper,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("İSTANBUL", "istanbul"),
        ("IŞIK", "ışık"),
        ("Işıklıova", "ışıklıova"),
        ("İkizpınar", "ikizpınar"),
        ("İSTANBUL", "istanbul"),  # decomposed İ as some PDF extractors emit it
    ],
)
def test_tr_lower_applies_turkish_dotted_and_dotless_i(text: str, expected: str) -> None:
    assert tr_lower(text) == expected


def test_python_lower_is_wrong_for_turkish_which_is_why_tr_lower_exists() -> None:
    assert "İSTANBUL".lower() != "istanbul"
    assert "IŞIK".lower() == "işik"


def test_tr_upper() -> None:
    assert tr_upper("istanbul ılık") == "İSTANBUL ILIK"


def test_fold_diacritics_maps_turkish_letters_and_circumflex() -> None:
    assert fold_diacritics("çğıöşü ÇĞİÖŞÜ âîû") == "cgiosu CGIOSU aiu"


@given(st.text(alphabet="abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ", max_size=60))
def test_tr_lower_upper_round_trip_on_turkish_alphabet(text: str) -> None:
    assert tr_lower(tr_upper(tr_lower(text))) == tr_lower(text)


@given(st.text(max_size=80))
def test_fold_is_idempotent(text: str) -> None:
    assert fold_diacritics(fold_diacritics(text)) == fold_diacritics(text)


def test_apostrophe_suffix_is_dropped() -> None:
    assert strip_apostrophe_suffix("akçaburç'taki") == "akçaburç"
    assert strip_apostrophe_suffix("ankara’da") == "ankara"
    assert strip_apostrophe_suffix("izin") == "izin"


def test_raw_tokens_keeps_numbers_codes_and_apostrophe_words() -> None:
    assert raw_tokens("E-14 kodu, 1.450 TL ve Akçaburç'taki 3. kat") == [
        "E-14", "kodu", "1.450", "TL", "ve", "Akçaburç'taki", "3", "kat",
    ]  # fmt: skip


def test_turkish_analyzer_matches_ascii_query_to_document_wording() -> None:
    an = Analyzer(TURKISH)
    doc = set(an.terms("Işıklıova Belediyesi'nde güvence bedeli alınır."))
    query = set(an.terms("isikliova belediyesinde guvence bedeli"))
    assert len(doc & query) >= 3  # entity, "güvence", "bedel" meet despite case, suffixes and diacritics


def test_naive_analyzer_misses_the_same_match() -> None:
    an = Analyzer(NAIVE)
    doc = set(an.terms("IŞIKLIOVA Belediyesi'nde güvence bedeli"))
    query = set(an.terms("ışıklıova belediyesinde güvence bedeli"))
    assert "ışıklıova" not in doc  # str.lower turned IŞIKLIOVA into "işikliova"
    assert len(doc & query) == 2  # only "güvence" and "bedeli"


def test_stopwords_removed_but_content_words_kept() -> None:
    terms = Analyzer(AnalyzerConfig(stemmer=Stemmer.NONE, fold=False)).terms("Bu izin için kaç gün ve saat?")
    assert terms == ["izin", "kaç", "gün", "saat"]


def test_inflected_forms_share_a_stem() -> None:
    an = Analyzer(TURKISH)
    for forms in [("ücret", "ücretler", "ücretin", "ücreti"), ("abonelik", "aboneliği", "abonelikler")]:
        assert len({an.terms(w)[0] for w in forms}) == 1, forms


def test_snowball_strips_one_suffix_layer_at_a_time() -> None:
    # Why Snowball is not the default: the stemmer is not a lemmatiser. Some lemmas
    # land on two stems ("izin" -> "iz", "izinler" -> "izin"), which costs lexical recall on those words.
    an = Analyzer(AnalyzerConfig(stemmer=Stemmer.SNOWBALL, fold=False))
    assert {an.terms(w)[0] for w in ("izin", "izinler", "izinlerin", "izinleri")} == {"iz", "izin"}


def test_prefix5_truncates_words_but_not_numbers() -> None:
    an = Analyzer(AnalyzerConfig(stemmer=Stemmer.PREFIX5, stopwords=False))
    assert an.terms("belediyesinde 123456") == ["beled", "123456"]


def test_analyzer_names_are_stable_because_the_index_records_them() -> None:
    assert TURKISH.name == "trcase+apos+stop+prefix5+fold"
    assert NAIVE.name == "naive"


def test_without_apostrophe_handling_the_suffix_becomes_a_separate_token() -> None:
    an = Analyzer(AnalyzerConfig(apostrophe=False, stopwords=False, stemmer=Stemmer.NONE, fold=False))
    assert an.terms("Akçaburç'ta") == ["akçaburç", "ta"]
