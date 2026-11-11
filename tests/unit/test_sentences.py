"""Golden tests for the Turkish sentence splitter."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragtr.text.sentences import split_sentences


def texts(source: str) -> list[str]:
    return [s.text for s in split_sentences(source)]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Başvuru 3. katta yapılır. Birim 09.00'da açılır.",
            ["Başvuru 3. katta yapılır.", "Birim 09.00'da açılır."],
        ),
        (
            "19. yüzyılda kuruldu. Bugün müzedir.",
            ["19. yüzyılda kuruldu.", "Bugün müzedir."],
        ),
        (
            "Dr. Ayşe Yılmaz ve Prof. Can Demir katıldı. Toplantı bitti.",
            ["Dr. Ayşe Yılmaz ve Prof. Can Demir katıldı.", "Toplantı bitti."],
        ),
        (
            "Kalem, defter vb. malzemeler verilir. Ayrıntılar için bkz. ek liste.",
            ["Kalem, defter vb. malzemeler verilir.", "Ayrıntılar için bkz. ek liste."],
        ),
        (
            "Adres: Cumhuriyet Mah. Gazi Cad. No. 5. Bina yenidir.",
            ["Adres: Cumhuriyet Mah. Gazi Cad. No. 5.", "Bina yenidir."],
        ),
        ("M. Kemal geldi. Herkes ayağa kalktı.", ["M. Kemal geldi.", "Herkes ayağa kalktı."]),
        ("Gerçekten mi? Evet! Tamam…  Bitti.", ["Gerçekten mi?", "Evet!", "Tamam…", "Bitti."]),
        (
            "Mavi Liman Lojistik Ltd. Şti. kuruldu. T.C. kimlik numarası istenir.",
            ["Mavi Liman Lojistik Ltd. Şti. kuruldu.", "T.C. kimlik numarası istenir."],
        ),
        (
            "Yıl 2024. Yeni yönetmelik yürürlükte.",
            ["Yıl 2024.", "Yeni yönetmelik yürürlükte."],
        ),
    ],
)
def test_golden(source: str, expected: list[str]) -> None:
    assert texts(source) == expected


def test_markdown_headings_and_list_items_are_their_own_units() -> None:
    source = (
        "# Başlık\n\nİlk paragraf burada. İkinci cümle.\n\n"
        "- E-14: Su girişi yok. Çözüm: musluğu açınız.\n- E-20: Kapı açık.\n"
    )
    assert texts(source) == [
        "# Başlık",
        "İlk paragraf burada.",
        "İkinci cümle.",
        "- E-14: Su girişi yok.",
        "Çözüm: musluğu açınız.",
        "- E-20: Kapı açık.",
    ]


def test_numbered_list_marker_is_not_an_ordinal() -> None:
    assert texts("1. Başvuru yapılır.\n2. Belge teslim edilir.") == [
        "1. Başvuru yapılır.",
        "2. Belge teslim edilir.",
    ]


def test_hard_wrapped_line_continues_the_sentence() -> None:
    assert texts("Bu cümle iki satıra\nbölünmüş durumda. Yeni cümle.") == [
        "Bu cümle iki satıra\nbölünmüş durumda.",
        "Yeni cümle.",
    ]


@given(st.text(alphabet="abcçdeğıiİoöşuü .!?\n3#-", max_size=200))
def test_spans_point_into_the_source_and_do_not_overlap(source: str) -> None:
    spans = split_sentences(source)
    last_end = 0
    for span in spans:
        assert source[span.start : span.end] == span.text
        assert span.text == span.text.strip()
        assert span.text
        assert span.start >= last_end
        last_end = span.end


@given(st.text(alphabet="abcçdeğıiİoöşuü .!?\n3", max_size=200))
def test_no_non_whitespace_character_is_lost(source: str) -> None:
    joined = "".join(s.text for s in split_sentences(source))
    assert "".join(joined.split()) == "".join(source.split())
