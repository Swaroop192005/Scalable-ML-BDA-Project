"""Rule generation, the metric formulas, and spurious-rule triage."""

import numpy as np
import pytest

from mba.data.basket import baskets_from_transactions
from mba.mining.fpgrowth import fpgrowth
from mba.mining.rules import (
    association_rules,
    deduplicate_symmetric,
    format_rules,
    rule_metrics,
)
from mba.mining.spurious import chi2_independence, flag_spurious_rules, high_value_rules


# ------------------------------------------------------------ the formulas
def test_metrics_match_hand_calculation():
    # s(A)=0.5, s(C)=0.4, s(A∪C)=0.3
    out = rule_metrics(0.3, 0.5, 0.4)
    assert out["confidence"] == pytest.approx(0.6)          # 0.3 / 0.5
    assert out["lift"] == pytest.approx(1.5)                # 0.6 / 0.4
    assert out["leverage"] == pytest.approx(0.1)            # 0.3 - 0.5*0.4
    assert out["conviction"] == pytest.approx(1.5)          # (1-0.4)/(1-0.6)
    # Zhang: (0.6-0.4) / max(0.6*0.6, 0.4*0.4) = 0.2 / 0.36
    assert out["zhangs_metric"] == pytest.approx(0.2 / 0.36)


def test_independent_items_give_lift_one():
    out = rule_metrics(0.2, 0.5, 0.4)  # 0.5 * 0.4 == 0.2 exactly
    assert out["lift"] == pytest.approx(1.0)
    assert out["leverage"] == pytest.approx(0.0)
    assert out["zhangs_metric"] == pytest.approx(0.0)


def test_perfect_confidence_gives_infinite_conviction():
    out = rule_metrics(0.4, 0.4, 0.5)  # confidence == 1.0
    assert out["confidence"] == pytest.approx(1.0)
    assert np.isinf(out["conviction"])


def test_zero_antecedent_support_is_not_nan():
    out = rule_metrics(0.0, 0.0, 0.5)
    assert out["confidence"] == 0.0
    assert np.isfinite(out["lift"])


def test_universal_consequent_is_not_infinitely_convincing():
    out = rule_metrics(0.5, 0.5, 1.0)
    assert out["conviction"] == pytest.approx(1.0)


def test_metrics_vectorise():
    out = rule_metrics(np.array([0.3, 0.2]), np.array([0.5, 0.5]), np.array([0.4, 0.4]))
    assert out["lift"].shape == (2,)
    assert out["lift"][0] == pytest.approx(1.5)


# ------------------------------------------------------- rule generation
def test_rules_are_generated_from_the_toy_association(toy_transactions):
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.2, max_len=3, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.5, min_lift=1.0)

    pairs = {(frozenset(r.antecedents), frozenset(r.consequents)) for r in rules.itertuples()}
    assert (frozenset({"bread"}), frozenset({"butter"})) in pairs


def test_bread_butter_outranks_bread_milk_on_lift(toy_transactions):
    """Confidence alone would rank these equally; lift separates them.

    milk is in 7/10 baskets, so predicting it is nearly free. butter is the
    genuine association.
    """
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=2, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)

    def lift_of(a, c):
        row = rules[(rules["antecedents"] == frozenset({a})) &
                    (rules["consequents"] == frozenset({c}))]
        return float(row["lift"].iloc[0])

    assert lift_of("bread", "butter") > lift_of("bread", "milk")


def test_thresholds_filter(toy_transactions):
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=3, track_memory=False)

    loose = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    strict = association_rules(itemsets, min_confidence=0.8, min_lift=2.0)
    assert len(strict) <= len(loose)
    if not strict.empty:
        assert (strict["confidence"] >= 0.8).all()
        assert (strict["lift"] >= 2.0).all()


def test_antecedent_and_consequent_never_overlap(toy_transactions):
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=3, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    for row in rules.itertuples():
        assert not (row.antecedents & row.consequents)


def test_max_antecedent_len_caps_the_left_hand_side(toy_transactions):
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=3, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0, max_antecedent_len=1)
    assert (rules["antecedent_len"] == 1).all()


def test_no_itemsets_gives_empty_rules():
    basket = baskets_from_transactions([["a"], ["b"]])
    itemsets = fpgrowth(basket, min_support=0.9, max_len=3, track_memory=False)
    assert association_rules(itemsets).empty


def test_format_and_dedup(toy_transactions):
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.2, max_len=2, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)

    formatted = format_rules(rules)
    assert "rule" in formatted.columns
    assert formatted["rule"].str.contains("=>").all()

    deduped = deduplicate_symmetric(rules)
    assert len(deduped) <= len(rules)
    keys = [frozenset(r.antecedents | r.consequents) for r in deduped.itertuples()]
    assert len(keys) == len(set(keys)), "both directions of a pair survived"


# ----------------------------------------------------------- triage
def test_chi2_detects_independence_and_dependence():
    # Perfectly independent: s(A∪C) == s(A)*s(C).
    _, p_independent = chi2_independence(
        np.array([0.25]), np.array([0.5]), np.array([0.5]), 1000
    )
    # Strongly dependent.
    _, p_dependent = chi2_independence(
        np.array([0.45]), np.array([0.5]), np.array([0.5]), 1000
    )
    assert p_independent[0] > 0.05
    assert p_dependent[0] < 0.001


def test_rare_pairs_are_flagged(toy_transactions, cfg):
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=2, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    annotated, flags = flag_spurious_rules(rules, cfg, basket.n_transactions)

    # Only 10 toy baskets, so every rule is below the min_rule_count of 20.
    assert annotated["rare_pair"].all()
    assert flags.n_flagged == len(annotated)


def test_colour_variants_are_flagged_self_referential(cfg):
    transactions = [["RED SPOTTY MUG", "BLUE SPOTTY MUG"]] * 60 + [["KETTLE", "TEA"]] * 40
    basket = baskets_from_transactions(transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=2, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    annotated, _ = flag_spurious_rules(rules, cfg, basket.n_transactions)

    mug_rules = annotated[
        annotated["antecedents"].map(lambda s: "SPOTTY MUG" in " ".join(s))
    ]
    assert mug_rules["self_referential"].all()

    other = annotated[annotated["antecedents"] == frozenset({"KETTLE"})]
    if not other.empty:
        assert not other["self_referential"].any()


def test_redundant_rules_are_flagged(cfg):
    """{A,B} => C adds nothing when {A} => C is already as confident."""
    transactions = [["a", "b", "c"]] * 50 + [["a", "c"]] * 50 + [["x", "y"]] * 20
    basket = baskets_from_transactions(transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=3, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    annotated, _ = flag_spurious_rules(rules, cfg, basket.n_transactions)

    target = annotated[
        (annotated["antecedents"] == frozenset({"a", "b"}))
        & (annotated["consequents"] == frozenset({"c"}))
    ]
    assert not target.empty and bool(target["redundant"].iloc[0])


def test_quality_tiers_are_assigned(toy_transactions, cfg):
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=2, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    annotated, _ = flag_spurious_rules(rules, cfg, basket.n_transactions)

    assert set(annotated["quality_tier"]) <= {"high_value", "review", "spurious"}
    assert (annotated["n_flags"] >= 0).all()
    assert (annotated["is_spurious"] == (annotated["n_flags"] > 0)).all()


def test_high_value_helper_returns_only_unflagged(toy_transactions, cfg):
    basket = baskets_from_transactions(toy_transactions)
    itemsets = fpgrowth(basket, min_support=0.1, max_len=2, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    kept = high_value_rules(rules, cfg)
    assert kept.empty or (~kept["is_spurious"]).all()


def test_triage_on_empty_rules(cfg):
    from mba.mining.rules import empty_rules_frame

    annotated, flags = flag_spurious_rules(empty_rules_frame(), cfg)
    assert annotated.empty and flags.n_rules == 0


# --------------------------------------------- multiple-testing correction
def test_bh_matches_the_textbook_example():
    """Benjamini & Hochberg's own 1995 worked example."""
    from mba.mining.spurious import benjamini_hochberg

    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216])
    q = benjamini_hochberg(p)
    expected = np.array([0.010, 0.040, 0.084, 0.084, 0.084, 0.100, 0.1057,
                         0.216, 0.216, 0.216])
    assert q == pytest.approx(expected, abs=1e-3)


def test_bh_q_values_never_fall_below_p_values():
    from mba.mining.spurious import benjamini_hochberg

    rng = np.random.default_rng(0)
    p = rng.random(500)
    assert (benjamini_hochberg(p) >= p - 1e-12).all()


def test_bh_is_monotone_in_p():
    """A worse-ranked p-value must never get a better q-value."""
    from mba.mining.spurious import benjamini_hochberg

    rng = np.random.default_rng(1)
    p = np.sort(rng.random(200))
    q = benjamini_hochberg(p)
    assert (np.diff(q) >= -1e-12).all()


def test_bh_is_order_invariant():
    from mba.mining.spurious import benjamini_hochberg

    rng = np.random.default_rng(2)
    p = rng.random(100)
    order = rng.permutation(100)
    shuffled = benjamini_hochberg(p[order])
    assert benjamini_hochberg(p)[order] == pytest.approx(shuffled)


def test_bh_clips_to_one():
    from mba.mining.spurious import benjamini_hochberg

    assert (benjamini_hochberg(np.full(50, 0.99)) <= 1.0).all()


def test_bh_handles_empty_input():
    from mba.mining.spurious import benjamini_hochberg

    assert len(benjamini_hochberg(np.array([]))) == 0


def test_significance_flag_uses_the_corrected_q_value(toy_transactions, cfg):
    """Testing hundreds of rules at p<0.05 would otherwise produce false positives."""
    basket = baskets_from_transactions(toy_transactions * 30)
    itemsets = fpgrowth(basket, min_support=0.05, max_len=3, track_memory=False)
    rules = association_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    annotated, _ = flag_spurious_rules(rules, cfg, basket.n_transactions)

    assert "chi2_q_value" in annotated.columns
    assert (annotated["chi2_q_value"] >= annotated["chi2_p_value"] - 1e-12).all()
    # The flag must follow the q-value, not the raw p-value.
    threshold = cfg.get("spurious.min_chi2_p_value", 0.05)
    expected = annotated["chi2_q_value"] > threshold
    assert (annotated["not_significant"] == expected).all()
