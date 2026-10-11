"""Maximal and closed itemsets."""

from itertools import combinations

import pytest

from mba.data.basket import baskets_from_transactions
from mba.mining.compact import (
    closed_itemsets,
    compact_representations,
    maximal_itemsets,
    recover_support,
)
from mba.mining.fpgrowth import fpgrowth


@pytest.fixture
def dense_result():
    """Dense data, where closure actually compresses.

    Every basket of the first group holds all of ``{a,b,c,d,e}``, so all 31
    of its subsets share support 100 and collapse to one closed itemset.
    """
    transactions = [["a", "b", "c", "d", "e"]] * 100 + [["a", "b", "x"]] * 40 + [["p", "q"]] * 30
    basket = baskets_from_transactions(transactions)
    return fpgrowth(basket, min_support=0.1, max_len=5, track_memory=False)


def test_containment_chain(dense_result):
    """maximal subset-of closed subset-of frequent, always."""
    frequent = set(dense_result.itemsets["itemsets"])
    closed = set(closed_itemsets(dense_result)["itemsets"])
    maximal = set(maximal_itemsets(dense_result)["itemsets"])
    assert maximal <= closed <= frequent


def test_closure_compresses_dense_data(dense_result):
    result = compact_representations(dense_result)
    assert result.n_closed < result.n_frequent
    assert result.closed_compression > 2.0


def test_closed_set_is_exactly_right(dense_result):
    closed = {tuple(sorted(s)) for s in closed_itemsets(dense_result)["itemsets"]}
    assert closed == {
        ("a", "b"),                      # support 140 — distinct from both supersets
        ("a", "b", "c", "d", "e"),       # support 100
        ("a", "b", "x"),                 # support 40
        ("p", "q"),                      # support 30
    }


def test_maximal_set_is_exactly_right(dense_result):
    maximal = {tuple(sorted(s)) for s in maximal_itemsets(dense_result)["itemsets"]}
    assert maximal == {("a", "b", "c", "d", "e"), ("a", "b", "x"), ("p", "q")}


def test_closure_is_lossless_for_support(dense_result):
    """The defining property: every frequent support recoverable from closed."""
    closed = closed_itemsets(dense_result)
    for itemset, count in zip(
        dense_result.itemsets["itemsets"], dense_result.itemsets["count"]
    ):
        assert recover_support(itemset, closed) == int(count), (
            f"support of {set(itemset)} not recoverable from the closed set"
        )


def test_closure_is_lossless_on_toy_data(toy_transactions):
    basket = baskets_from_transactions(toy_transactions)
    result = fpgrowth(basket, min_support=0.1, max_len=3, track_memory=False)
    closed = closed_itemsets(result)
    for itemset, count in zip(result.itemsets["itemsets"], result.itemsets["count"]):
        assert recover_support(itemset, closed) == int(count)


def test_no_closed_itemset_has_an_equal_support_superset(dense_result):
    """The definition of closure, checked directly."""
    closed = closed_itemsets(dense_result)
    supports = dict(zip(dense_result.itemsets["itemsets"], dense_result.itemsets["count"]))
    for itemset, count in zip(closed["itemsets"], closed["count"]):
        for other, other_count in supports.items():
            if itemset < other:
                assert int(other_count) < int(count), (
                    f"{set(itemset)} is not closed: {set(other)} has the same support"
                )


def test_no_maximal_itemset_has_a_frequent_superset(dense_result):
    maximal = maximal_itemsets(dense_result)
    frequent = set(dense_result.itemsets["itemsets"])
    for itemset in maximal["itemsets"]:
        assert not any(itemset < other for other in frequent)


def test_every_frequent_itemset_is_inside_some_maximal_one(dense_result):
    """The maximal set preserves the frequent/infrequent boundary."""
    maximal = list(maximal_itemsets(dense_result)["itemsets"])
    for itemset in dense_result.itemsets["itemsets"]:
        assert any(itemset <= m for m in maximal)


def test_truncation_boundary_is_marked(toy_transactions):
    """max_len makes the longest mined length artificially maximal."""
    basket = baskets_from_transactions(toy_transactions)
    result = fpgrowth(basket, min_support=0.1, max_len=2, track_memory=False)
    maximal = maximal_itemsets(result)
    assert "maximal_by_truncation" in maximal.columns
    assert maximal.loc[maximal["length"] == 2, "maximal_by_truncation"].all()


def test_recover_support_returns_none_for_infrequent(dense_result):
    closed = closed_itemsets(dense_result)
    assert recover_support(frozenset({"not-an-item"}), closed) is None


def test_empty_input():
    from mba.mining.base import empty_itemsets_frame

    empty = empty_itemsets_frame()
    assert closed_itemsets(empty).empty
    assert maximal_itemsets(empty).empty
