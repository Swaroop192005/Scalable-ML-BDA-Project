"""Apriori and FP-Growth: correctness, agreement and edge cases.

The central property is that both algorithms must return *exactly* what an
exhaustive search returns. Everything else about them is an optimisation.
"""

import random

import pytest

from conftest import brute_force_itemsets
from mba.data.basket import baskets_from_transactions
from mba.mining.apriori import apriori, apriori_from_transactions
from mba.mining.base import available_miners, mine
from mba.mining.fpgrowth import FPTree, fpgrowth, fpgrowth_from_transactions


def as_dict(result) -> dict:
    return {row.itemsets: row.count for row in result.itemsets.itertuples()}


# ------------------------------------------------------------ correctness
@pytest.mark.parametrize("algorithm", ["apriori", "fpgrowth"])
@pytest.mark.parametrize("min_support", [0.1, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("max_len", [1, 2, 3])
def test_matches_brute_force(toy_transactions, algorithm, min_support, max_len):
    basket = baskets_from_transactions(toy_transactions)
    result = mine(basket, algorithm=algorithm, min_support=min_support,
                  max_len=max_len, track_memory=False)
    assert as_dict(result) == brute_force_itemsets(toy_transactions, min_support, max_len)


def test_algorithms_agree_on_random_data():
    """50 random datasets; both miners must match brute force exactly."""
    rng = random.Random(12345)
    for trial in range(50):
        n_items = rng.randint(4, 10)
        transactions = [
            [f"i{x}" for x in rng.sample(range(n_items), rng.randint(1, min(n_items, 6)))]
            for _ in range(rng.randint(8, 50))
        ]
        min_support = rng.choice([0.05, 0.1, 0.2, 0.35])
        max_len = rng.randint(1, 4)

        reference = brute_force_itemsets(transactions, min_support, max_len)
        a = as_dict(apriori_from_transactions(transactions, min_support, max_len, track_memory=False))
        f = as_dict(fpgrowth_from_transactions(transactions, min_support, max_len, track_memory=False))

        assert a == reference, f"apriori diverged on trial {trial}"
        assert f == reference, f"fpgrowth diverged on trial {trial}"


def test_downward_closure_holds(toy_basket):
    """Every subset of a frequent itemset must itself be frequent."""
    from itertools import combinations

    result = fpgrowth(toy_basket, min_support=0.2, max_len=3, track_memory=False)
    found = as_dict(result)
    for itemset, count in found.items():
        for size in range(1, len(itemset)):
            for subset in combinations(sorted(itemset), size):
                key = frozenset(subset)
                assert key in found, f"{subset} missing though superset {set(itemset)} is frequent"
                assert found[key] >= count, "a subset cannot be rarer than its superset"


def test_support_equals_count_over_transactions(toy_basket):
    result = fpgrowth(toy_basket, min_support=0.1, max_len=3, track_memory=False)
    for row in result.itemsets.itertuples():
        assert row.support == pytest.approx(row.count / toy_basket.n_transactions)


def test_max_len_is_respected(toy_basket):
    for limit in (1, 2, 3):
        result = fpgrowth(toy_basket, min_support=0.1, max_len=limit, track_memory=False)
        assert result.itemsets["length"].max() <= limit


def test_min_count_overrides_min_support(toy_basket):
    result = apriori(toy_basket, min_support=0.9, min_count=2, max_len=2, track_memory=False)
    assert (result.itemsets["count"] >= 2).all()
    assert len(result) > 0


# ------------------------------------------------------------- edge cases
@pytest.mark.parametrize("algorithm", ["apriori", "fpgrowth"])
def test_empty_input(algorithm):
    basket = baskets_from_transactions([])
    result = mine(basket, algorithm=algorithm, min_support=0.1, track_memory=False)
    assert result.empty


@pytest.mark.parametrize("algorithm", ["apriori", "fpgrowth"])
def test_support_of_one_keeps_only_universal_items(algorithm):
    transactions = [["a", "b"], ["a", "c"], ["a", "d"]]
    result = mine(baskets_from_transactions(transactions), algorithm=algorithm,
                  min_support=1.0, max_len=2, track_memory=False)
    assert set(result.itemsets["itemsets"]) == {frozenset({"a"})}


@pytest.mark.parametrize("algorithm", ["apriori", "fpgrowth"])
def test_identical_transactions(algorithm):
    transactions = [["a", "b", "c"]] * 7
    result = mine(baskets_from_transactions(transactions), algorithm=algorithm,
                  min_support=0.5, max_len=3, track_memory=False)
    assert len(result) == 7  # 3 singles + 3 pairs + 1 triple
    assert (result.itemsets["count"] == 7).all()


def test_unknown_algorithm_raises(toy_basket):
    with pytest.raises(ValueError, match="unknown mining algorithm"):
        mine(toy_basket, algorithm="not-a-miner")


def test_both_miners_are_registered():
    assert {"apriori", "fpgrowth"} <= set(available_miners())


# ---------------------------------------------------------------- FP-tree
def test_fptree_shares_prefix_paths():
    tree = FPTree()
    tree.add_transaction([0, 1, 2])
    tree.add_transaction([0, 1, 3])
    # 0 -> 1 is shared; only the leaves differ, so 4 nodes not 6.
    assert tree.n_nodes == 4
    assert tree.item_counts[0] == 2 and tree.item_counts[1] == 2


def test_fptree_header_chain_links_every_occurrence():
    tree = FPTree()
    tree.add_transaction([0, 1])
    tree.add_transaction([2, 1])
    assert len(list(tree.nodes_for(1))) == 2


def test_fptree_detects_a_single_path():
    tree = FPTree()
    tree.add_transaction([0, 1, 2])
    assert tree.single_path() is not None

    tree.add_transaction([0, 3])
    assert tree.single_path() is None


def test_conditional_pattern_base_carries_counts():
    tree = FPTree()
    tree.add_transaction([0, 1], count=3)
    tree.add_transaction([0, 2, 1], count=2)
    base = tree.conditional_pattern_base(1)
    assert sorted((sorted(prefix), count) for prefix, count in base) == [
        ([0], 3), ([0, 2], 2)
    ]


def test_stats_are_recorded(toy_basket):
    result = fpgrowth(toy_basket, min_support=0.2, max_len=3)
    stats = result.stats
    assert stats.algorithm == "fpgrowth"
    assert stats.runtime_sec > 0
    assert stats.n_itemsets == len(result)
    assert stats.extra["fp_tree_nodes"] > 0
    assert "min_count_threshold" in stats.extra


def test_apriori_records_candidates_per_level(toy_basket):
    result = apriori(toy_basket, min_support=0.2, max_len=3, track_memory=False)
    assert result.stats.candidates_per_level[1] == toy_basket.n_items
    # Pruning must mean fewer frequent sets than candidates at level 2.
    assert result.stats.itemsets_per_level.get(2, 0) <= result.stats.candidates_per_level.get(2, 0)
