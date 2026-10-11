"""Lossy Counting: the three guarantees, checked against exact counts."""

import random
from collections import Counter

import pytest

from mba.scaling.streaming import (
    LossyCounter,
    compare_with_exact,
    stream_frequent_items,
)


@pytest.fixture
def skewed_stream():
    """20k baskets from a Zipf-ish catalogue — a few hot items, a long tail."""
    rng = random.Random(1234)
    weights = [1.0 / (i + 1) ** 1.1 for i in range(600)]
    total = sum(weights)
    weights = [w / total for w in weights]
    items = [f"item{i:04d}" for i in range(600)]
    return [
        rng.choices(items, weights=weights, k=rng.randint(2, 12))
        for _ in range(20_000)
    ]


# ------------------------------------------------------- the guarantees
def test_no_false_negatives(skewed_stream):
    """Guarantee 1: every genuinely frequent item is reported."""
    comparison = compare_with_exact(skewed_stream, min_support=0.02, epsilon=0.002)
    assert int(comparison["false_negative"].sum()) == 0
    assert comparison["truly_frequent"].sum() > 0, "test data has no frequent items"


def test_counts_are_never_overestimates(skewed_stream):
    """Guarantee 2a: an approximate count never exceeds the true one."""
    comparison = compare_with_exact(skewed_stream, min_support=0.02, epsilon=0.002)
    reported = comparison.loc[comparison["reported"]]
    assert (reported["error"] >= 0).all()


def test_undercount_respects_the_error_bound(skewed_stream):
    """Guarantee 2b: the shortfall never exceeds epsilon * N."""
    epsilon = 0.002
    comparison = compare_with_exact(skewed_stream, min_support=0.02, epsilon=epsilon)
    reported = comparison.loc[comparison["reported"]]
    bound = epsilon * len(skewed_stream)
    assert (reported["error"] <= bound).all()
    assert reported["within_error_bound"].all()


def test_classical_space_bound_holds_for_a_single_item_stream():
    """Guarantee 3, in the setting where it actually applies.

    The (1/eps)*log(eps*N) bound assumes one item per observation. The
    transaction variant inserts a whole basket between prunes and can exceed
    it, which is why `space_bound` scales it — see the module docstring.
    """
    rng = random.Random(7)
    counter = LossyCounter(epsilon=0.001)
    for _ in range(100_000):
        counter.update(f"k{int(rng.paretovariate(1.2))}")
    stats = counter.stats()
    assert stats.mean_items_per_observation == pytest.approx(1.0)
    assert stats.peak_entries <= stats.classical_space_bound


def test_transaction_stream_respects_the_scaled_bound(skewed_stream):
    _, stats = stream_frequent_items(skewed_stream, min_support=0.02, epsilon=0.002)
    assert stats.mean_items_per_observation > 1.0
    assert stats.within_space_bound


# ------------------------------------------------------------ behaviour
def test_memory_is_far_below_the_distinct_item_count(skewed_stream):
    _, stats = stream_frequent_items(skewed_stream, min_support=0.05, epsilon=0.005)
    assert stats.peak_entries < stats.distinct_seen


def test_tighter_epsilon_costs_memory_and_buys_accuracy(skewed_stream):
    _, loose = stream_frequent_items(skewed_stream, min_support=0.02, epsilon=0.01)
    _, tight = stream_frequent_items(skewed_stream, min_support=0.02, epsilon=0.0005)
    assert tight.peak_entries > loose.peak_entries
    assert tight.max_count_error < loose.max_count_error


def test_counts_are_per_transaction_not_per_item(skewed_stream):
    """The bug this guards: bucketing on item occurrences broke the scale."""
    frame, stats = stream_frequent_items(skewed_stream, min_support=0.02, epsilon=0.002)
    assert stats.n_observations == len(skewed_stream)
    assert not frame.empty, "nothing reported — threshold is on the wrong scale"
    exact = Counter()
    for transaction in skewed_stream:
        exact.update(set(transaction))
    top_item = frame.iloc[0]["item"]
    assert frame.iloc[0]["count"] <= exact[top_item]
    assert frame.iloc[0]["support"] == pytest.approx(
        frame.iloc[0]["count"] / len(skewed_stream)
    )


def test_duplicate_items_in_one_basket_count_once():
    counter = LossyCounter(epsilon=0.01)
    counter.update_transaction(["a", "a", "a", "b"])
    assert counter.entries["a"][0] == 1
    assert counter.n_observations == 1


def test_incremental_use_matches_one_shot(skewed_stream):
    one_shot = LossyCounter(epsilon=0.002)
    for transaction in skewed_stream:
        one_shot.update_transaction(transaction)

    incremental = LossyCounter(epsilon=0.002)
    for start in range(0, len(skewed_stream), 1000):
        for transaction in skewed_stream[start : start + 1000]:
            incremental.update_transaction(transaction)

    assert one_shot.entries == incremental.entries


def test_invalid_epsilon_rejected():
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            LossyCounter(epsilon=bad)


def test_warns_when_epsilon_is_not_below_min_support(skewed_stream, caplog):
    counter = LossyCounter(epsilon=0.05)
    counter.update_transaction(["a", "b"])
    with caplog.at_level("WARNING"):
        counter.frequent(min_support=0.01)
    assert "min_support" in caplog.text


def test_empty_stream():
    frame, stats = stream_frequent_items([], min_support=0.01)
    assert frame.empty
    assert stats.n_observations == 0
