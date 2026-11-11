from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragtr.eval.metrics import (
    RetrievalScores,
    contains_all,
    groundedness,
    mean_or_none,
    ndcg_at_k,
    normalise_answer,
    recall_at_k,
    reciprocal_rank,
)
from ragtr.text.turkish import Analyzer


def test_recall_mrr_ndcg_on_a_known_ranking() -> None:
    ranked = ["x", "a", "y", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(ranked, relevant, 1) == 0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1
    assert reciprocal_rank(ranked, relevant, 10) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(ranked, relevant, 10), expected)


def test_metrics_are_zero_without_relevant_items_or_beyond_k() -> None:
    assert recall_at_k(["a"], set(), 5) == 0
    assert ndcg_at_k(["a"], set(), 5) == 0
    assert reciprocal_rank(["x", "y", "a"], {"a"}, 2) == 0


def test_perfect_ranking_scores_one() -> None:
    s = RetrievalScores.of(["a", "b"], {"a"})
    assert (s.recall_at_1, s.mrr_at_10, s.ndcg_at_10) == (1, 1, 1)


def test_mean_of_nothing_is_zero() -> None:
    assert RetrievalScores.mean([]).recall_at_5 == 0


@given(
    st.lists(st.sampled_from("abcdefghij"), unique=True, max_size=10),
    st.sets(st.sampled_from("abcdefghij"), min_size=1),
    st.integers(1, 10),
)
def test_metric_bounds_and_monotonicity(ranked: list[str], relevant: set[str], k: int) -> None:
    for value in (
        recall_at_k(ranked, relevant, k),
        reciprocal_rank(ranked, relevant, k),
        ndcg_at_k(ranked, relevant, k),
    ):
        assert 0 <= value <= 1 + 1e-12
    assert recall_at_k(ranked, relevant, k) <= recall_at_k(ranked, relevant, k + 1)


def test_answer_normalisation_handles_turkish_case_diacritics_and_markers() -> None:
    assert normalise_answer("İKİ GÜN  içinde [1, 2]") == "iki gun icinde"
    assert contains_all("Pazar yeri ÇARŞAMBA günü kurulur [1].", ["carsamba"])
    assert not contains_all("1.450 TL", ["1.450", "gun"])


def test_groundedness_flags_invented_content() -> None:
    an = Analyzer()
    context = ["Yeni konut aboneliklerinde 1.450 TL güvence bedeli alınır."]
    assert groundedness(an, "Güvence bedeli 1.450 TL [1].", context) == 1.0
    assert groundedness(an, "Güvence bedeli 2.000 TL [1].", context) == pytest.approx(3 / 4)
    assert groundedness(an, "[1]", context) == 1.0


def test_mean_or_none_skips_missing_values() -> None:
    assert mean_or_none([True, False, None]) == 0.5
    assert mean_or_none([None]) is None
