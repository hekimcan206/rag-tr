from __future__ import annotations

import math
from collections import Counter

from hypothesis import given
from hypothesis import strategies as st

from ragtr.lexical import BM25Index, bm25_idf, bm25_term, reciprocal_rank_fusion, top_k


def test_idf_is_never_negative_even_for_a_term_in_every_document() -> None:
    assert bm25_idf(10, 10) > 0
    assert bm25_idf(10, 1) > bm25_idf(10, 5)


def test_bm25_matches_hand_computed_score() -> None:
    index = BM25Index()
    index.add("a", Counter({"izin": 2, "gün": 1}))
    index.add("b", Counter({"ücret": 1, "gün": 3}))
    index.add("c", Counter({"ücret": 2}))
    [(doc, score)] = index.search(["izin"], k=1)
    avg = (3 + 4 + 2) / 3
    expected = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)) * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / avg))
    assert doc == "a"
    assert math.isclose(score, expected, rel_tol=1e-12)
    assert math.isclose(expected, bm25_term(2, 3, avg, bm25_idf(3, 1)), rel_tol=1e-12)


def test_remove_and_replace_keep_statistics_consistent() -> None:
    index = BM25Index()
    index.add("a", Counter({"x": 1}))
    index.add("b", Counter({"x": 1, "y": 1}))
    index.add("a", Counter({"y": 5}))  # replace
    assert len(index) == 2
    assert [d for d, _ in index.search(["x"], 5)] == ["b"]
    index.remove("b")
    index.remove("missing")
    assert index.search(["x"], 5) == []
    assert [d for d, _ in index.search(["y"], 5)] == ["a"]


def test_empty_index_and_empty_query() -> None:
    assert BM25Index().search(["x"], 3) == []
    index = BM25Index()
    index.add("a", Counter({"x": 1}))
    assert index.search([], 3) == []


def test_top_k_breaks_ties_by_id() -> None:
    assert top_k({"b": 1.0, "a": 1.0, "c": 2.0}, 3) == [("c", 2.0), ("a", 1.0), ("b", 1.0)]


def test_rrf_rewards_agreement_between_rankers() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [d for d, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert fused[0][1] == fused[1][1] == 1 / 61 + 1 / 62
    assert {d for d, _ in fused} == {"a", "b", "c", "d"}


@given(st.lists(st.lists(st.sampled_from("abcdefgh"), unique=True, max_size=8), max_size=4))
def test_rrf_scores_are_bounded_by_number_of_rankings(rankings: list[list[str]]) -> None:
    for _, score in reciprocal_rank_fusion(rankings):
        assert 0 < score <= len(rankings) / 61
