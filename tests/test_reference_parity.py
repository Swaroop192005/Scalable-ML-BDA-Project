"""Cross-validation against mlxtend, an independent reference implementation.

Agreeing with our own brute-force checker proves internal consistency. This
file proves the *definitions* match a widely-used third-party library — that
support is counted the same way, that `max_len` means the same thing, and
that our rule metrics reproduce theirs.

Skipped entirely when mlxtend is absent, so it is a bonus check rather than
a dependency.
"""

import pytest

mlxtend = pytest.importorskip("mlxtend", reason="mlxtend is an optional extra")

import pandas as pd
from mlxtend.frequent_patterns import apriori as mlx_apriori
from mlxtend.frequent_patterns import association_rules as mlx_rules
from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth

from mba.data.basket import baskets_from_transactions
from mba.mining.apriori import apriori
from mba.mining.fpgrowth import fpgrowth
from mba.mining.rules import association_rules


def to_dense(basket) -> pd.DataFrame:
    return pd.DataFrame(basket.matrix.toarray(), columns=[str(i) for i in basket.items])


def as_support_map(frame, column="itemsets") -> dict:
    return {
        frozenset(map(str, row[column])): round(float(row["support"]), 9)
        for _, row in frame.iterrows()
    }


@pytest.fixture
def basket(toy_transactions):
    return baskets_from_transactions(toy_transactions)


@pytest.mark.parametrize("min_support", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("max_len", [2, 3])
def test_our_apriori_matches_mlxtend(basket, min_support, max_len):
    ours = apriori(basket, min_support=min_support, max_len=max_len, track_memory=False)
    theirs = mlx_apriori(
        to_dense(basket), min_support=min_support, max_len=max_len, use_colnames=True
    )
    assert as_support_map(ours.itemsets) == as_support_map(theirs)


@pytest.mark.parametrize("min_support", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("max_len", [2, 3])
def test_our_fpgrowth_matches_mlxtend(basket, min_support, max_len):
    ours = fpgrowth(basket, min_support=min_support, max_len=max_len, track_memory=False)
    theirs = mlx_fpgrowth(
        to_dense(basket), min_support=min_support, max_len=max_len, use_colnames=True
    )
    assert as_support_map(ours.itemsets) == as_support_map(theirs)


def test_our_rule_metrics_match_mlxtend(basket):
    ours_itemsets = fpgrowth(basket, min_support=0.1, max_len=3, track_memory=False)
    ours = association_rules(ours_itemsets, min_confidence=0.0, min_lift=0.0)

    theirs = mlx_rules(
        mlx_fpgrowth(to_dense(basket), min_support=0.1, max_len=3, use_colnames=True),
        metric="confidence", min_threshold=0.0,
    )

    def key(frame):
        return {
            (frozenset(map(str, r["antecedents"])), frozenset(map(str, r["consequents"]))): r
            for _, r in frame.iterrows()
        }

    ours_by_key, theirs_by_key = key(ours), key(theirs)
    assert set(ours_by_key) == set(theirs_by_key)

    for rule_key, our_row in ours_by_key.items():
        their_row = theirs_by_key[rule_key]
        for metric in ("support", "confidence", "lift", "leverage"):
            assert our_row[metric] == pytest.approx(their_row[metric], rel=1e-9), (
                f"{metric} differs for {rule_key}"
            )
        # mlxtend reports inf for conviction at confidence 1.0, as we do.
        if pd.notna(their_row["conviction"]):
            assert our_row["conviction"] == pytest.approx(
                their_row["conviction"], rel=1e-9, nan_ok=True
            )
