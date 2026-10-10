"""Tests against the real UCI Online Retail dataset.

All marked ``slow`` and skipped when the file has not been downloaded, so a
fresh clone still gets a green suite from ``pytest``.

Run them with::

    pytest -m slow
"""

import pytest

from mba.data.basket import encode_baskets
from mba.data.loader import load_raw
from mba.data.preprocess import clean_transactions
from mba.mining.apriori import apriori
from mba.mining.fpgrowth import fpgrowth
from mba.mining.rules import association_rules
from mba.mining.spurious import flag_spurious_rules

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_basket(cfg, real_data_path):
    if real_data_path is None:
        pytest.skip("dataset not downloaded; run python scripts/download_data.py")
    raw = load_raw(cfg, path=real_data_path)
    clean, report = clean_transactions(raw, cfg)
    return encode_baskets(clean, cfg), clean, report


def test_dataset_has_the_expected_shape(real_basket):
    _, clean, report = real_basket
    # The UCI file is 541,909 rows; cleaning should retain most of it.
    assert report.rows_in > 500_000
    assert 0.6 < report.retention_rate < 1.0
    assert len(clean) == report.rows_out


def test_encoding_is_overwhelmingly_sparse(real_basket):
    basket, _, _ = real_basket
    stats = basket.describe()
    assert stats["n_transactions"] > 10_000
    assert stats["n_items"] > 1_000
    assert stats["density"] < 0.02, "a retail basket matrix should be ~99% empty"
    assert stats["compression_ratio"] > 10, "sparse storage should beat dense by 10x+"


def test_both_algorithms_agree_on_the_real_data(real_basket):
    """The headline correctness claim, on 18k real baskets."""
    basket, _, _ = real_basket
    a = apriori(basket, min_support=0.02, max_len=3, track_memory=False)
    f = fpgrowth(basket, min_support=0.02, max_len=3, track_memory=False)

    as_a = {row.itemsets: row.count for row in a.itemsets.itertuples()}
    as_f = {row.itemsets: row.count for row in f.itemsets.itertuples()}
    assert as_a == as_f
    assert len(as_a) > 100, "expected a few hundred itemsets at 2% support"


def test_out_of_core_matches_in_memory_on_real_data(real_basket):
    from mba.scaling.outofcore import mine_out_of_core

    basket, _, _ = real_basket
    in_memory = fpgrowth(basket, min_support=0.03, max_len=2, track_memory=False)
    out_of_core = mine_out_of_core(
        basket.as_item_lists(), min_support=0.03, max_len=2,
        chunk_size=4000, memory_budget_mb=None,
    )
    assert {r.itemsets: r.count for r in in_memory.itemsets.itertuples()} == {
        r.itemsets: r.count for r in out_of_core.itemsets.itertuples()
    }


def test_known_product_associations_are_found(real_basket, cfg):
    """Sanity check against domain knowledge of this catalogue.

    The green/pink/roses Regency teacups are the dataset's best-known
    association. If the pipeline cannot find them, something upstream broke.
    """
    basket, _, _ = real_basket
    itemsets = fpgrowth(basket, min_support=0.02, max_len=3, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.3, min_lift=1.2)

    assert not rules.empty
    text = " | ".join(
        f"{sorted(r.antecedents)} -> {sorted(r.consequents)}" for r in rules.itertuples()
    )
    assert "REGENCY TEACUP" in text, "the classic teacup association is missing"
    assert (rules["lift"] > 1.2).all()


def test_rule_triage_flags_colour_variants(real_basket, cfg):
    basket, _, _ = real_basket
    itemsets = fpgrowth(basket, min_support=0.02, max_len=3, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.3, min_lift=1.2)
    annotated, flags = flag_spurious_rules(rules, cfg, basket.n_transactions)

    assert flags.n_rules == len(rules)
    assert flags.n_high_value + flags.n_flagged == flags.n_rules
    # The teacup colour variants are the textbook self-referential case.
    assert flags.counts["self_referential"] > 0


def test_uk_dominates_so_regional_mining_is_justified(real_basket, cfg):
    _, clean, _ = real_basket
    share = clean["Country"].value_counts(normalize=True)
    assert share.iloc[0] > 0.8, "expected one dominant market in this dataset"
