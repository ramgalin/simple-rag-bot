"""Unit tests for the metrics themselves.

A metric with a bug does not fail loudly — it reports a plausible number and
quietly stops protecting anything. So the metrics get tested before they are
trusted to judge the retriever.
"""

import pytest

from ragbot.evaluation import (
    distinct_sources,
    evaluate,
    mean,
    recall_at_k,
    reciprocal_rank,
)

# a.md at chunk positions 2 and 3, b.md at 1 and 4
RANKED = ["b.md", "a.md", "a.md", "b.md"]


def test_recall_at_k_counts_a_hit_anywhere_in_the_top_k():
    assert recall_at_k(RANKED, ["a.md"], 1) == 0.0     # position 1 is b.md
    assert recall_at_k(RANKED, ["a.md"], 2) == 1.0     # position 2 is a.md
    assert recall_at_k(RANKED, ["a.md"], 4) == 1.0


def test_recall_at_k_with_several_relevant_sources_is_a_fraction():
    assert recall_at_k(RANKED, ["a.md", "b.md"], 1) == 0.5
    assert recall_at_k(RANKED, ["a.md", "b.md"], 2) == 1.0


def test_recall_at_k_is_zero_when_nothing_relevant_was_retrieved():
    assert recall_at_k(RANKED, ["missing.md"], 4) == 0.0


def test_recall_at_k_rejects_an_empty_relevant_set():
    # silently returning 1.0 (or 0.0) here would hide a broken dataset
    with pytest.raises(ValueError):
        recall_at_k(RANKED, [], 4)


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert reciprocal_rank(RANKED, ["b.md"]) == 1.0        # position 1
    assert reciprocal_rank(RANKED, ["a.md"]) == 0.5        # position 2
    assert reciprocal_rank(["x", "y", "a.md"], ["a.md"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_the_source_never_appears():
    assert reciprocal_rank(RANKED, ["missing.md"]) == 0.0


def test_mrr_distinguishes_rank_one_from_rank_five():
    """The point of MRR: recall@8 cannot tell these two retrievers apart."""
    good = ["a.md"] + ["x"] * 7
    poor = ["x"] * 4 + ["a.md"] + ["x"] * 3
    assert recall_at_k(good, ["a.md"], 8) == recall_at_k(poor, ["a.md"], 8) == 1.0
    assert reciprocal_rank(good, ["a.md"]) == 1.0
    assert reciprocal_rank(poor, ["a.md"]) == 0.2


def test_distinct_sources_keeps_first_appearance_order():
    assert distinct_sources(RANKED) == ["b.md", "a.md"]


def test_mean_of_nothing_is_zero_not_a_crash():
    assert mean([]) == 0.0


def test_evaluate_aggregates_and_reports_misses():
    report = evaluate(
        [
            ("hit", ["a.md"], ["a.md", "b.md"]),
            ("miss", ["c.md"], ["a.md", "b.md"]),
        ],
        ks=(1, 2),
    )
    assert report["n"] == 2
    assert report["recall@1"] == 0.5
    assert report["mrr"] == 0.5
    assert [m["question"] for m in report["misses"]] == ["miss"]
