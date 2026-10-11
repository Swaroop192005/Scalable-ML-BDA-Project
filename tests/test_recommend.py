"""Recommenders and the held-out evaluation protocol."""

import numpy as np
import pandas as pd
import pytest

from mba.recommend.engine import (
    AssociationRecommender,
    CooccurrenceRecommender,
    PopularityRecommender,
    RandomRecommender,
    build_recommender,
)
from mba.recommend.evaluate import (
    compare_recommenders,
    evaluate_at_ks,
    evaluate_recommender,
    lift_over_baseline,
    temporal_split,
)


@pytest.fixture
def learnable_transactions():
    """Data with a planted association a strong model must find.

    ``bread`` is followed by ``butter`` in 300 of 400 baskets. ``filler`` is
    in 500 baskets — more popular than butter — so a popularity model will
    recommend filler and miss, while an association model should not.
    """
    return (
        [["bread", "butter", "jam"]] * 200
        + [["bread", "butter"]] * 100
        + [["bread", "cheese"]] * 100
        + [["filler", "other"]] * 500
        + [["milk", "cereal"]] * 150
    )


@pytest.fixture
def dated_frame():
    """A transaction frame spanning 100 days, for the temporal split."""
    rows = []
    base = pd.Timestamp("2011-01-01")
    for day in range(100):
        for order in range(5):
            invoice = f"{day:03d}-{order}"
            for item in (["a", "b"] if day % 2 == 0 else ["c", "d"]):
                rows.append(
                    {
                        "InvoiceNo": invoice,
                        "Description": item,
                        "InvoiceDate": base + pd.Timedelta(days=day),
                    }
                )
    return pd.DataFrame(rows)


# -------------------------------------------------------------- engines
@pytest.mark.parametrize(
    "name", ["association", "cooccurrence", "popularity", "random"]
)
def test_every_model_returns_k_items(name, learnable_transactions):
    model = build_recommender(name)
    model.fit(learnable_transactions)
    for k in (1, 3, 5):
        picks = model.recommend(["bread"], k=k)
        assert len(picks) == k
        assert len(set(picks)) == k, "duplicate recommendations"


@pytest.mark.parametrize(
    "name", ["association", "cooccurrence", "popularity", "random"]
)
def test_never_recommends_what_is_already_in_the_basket(name, learnable_transactions):
    model = build_recommender(name)
    model.fit(learnable_transactions)
    basket = ["bread", "butter"]
    assert not set(model.recommend(basket, k=5)) & set(basket)


def test_association_finds_the_planted_rule(learnable_transactions):
    model = AssociationRecommender(mine_kwargs={"min_support": 0.05, "max_len": 2})
    model.fit(learnable_transactions)
    assert "butter" in model.recommend(["bread"], k=2)


def test_popularity_ranking_ignores_basket_content(learnable_transactions):
    """Popularity ranks identically regardless of basket.

    The two lists are not equal outright, because each excludes its own
    basket's items. The invariant is that the *relative order* of everything
    not in either basket is the same — the basket changes what is filtered,
    never what is preferred.
    """
    model = PopularityRecommender().fit(learnable_transactions)
    from_bread = model.recommend(["bread"], k=6)
    from_milk = model.recommend(["milk"], k=6)
    shared = {"bread", "milk"}
    assert [i for i in from_bread if i not in shared] == [
        i for i in from_milk if i not in shared
    ]


def test_association_beats_popularity_on_the_planted_rule(learnable_transactions):
    """The whole point: context should beat a bestseller list."""
    association = AssociationRecommender(
        mine_kwargs={"min_support": 0.05, "max_len": 2}
    ).fit(learnable_transactions)
    popularity = PopularityRecommender().fit(learnable_transactions)
    assert "butter" in association.recommend(["bread"], k=1)
    assert "butter" not in popularity.recommend(["bread"], k=1)


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError, match="must be fitted"):
        PopularityRecommender().recommend(["x"], k=3)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="unknown recommender"):
        build_recommender("nope")


def test_recommendations_are_deterministic(learnable_transactions):
    a = AssociationRecommender(mine_kwargs={"min_support": 0.05}).fit(learnable_transactions)
    b = AssociationRecommender(mine_kwargs={"min_support": 0.05}).fit(learnable_transactions)
    assert a.recommend(["bread"], k=5) == b.recommend(["bread"], k=5)


def test_top_k_is_a_prefix_of_top_K(learnable_transactions):
    """The property the one-pass evaluation optimisation relies on."""
    for name in ("association", "cooccurrence", "popularity"):
        model = build_recommender(name).fit(learnable_transactions)
        top10 = model.recommend(["bread"], k=10)
        for k in (1, 3, 5):
            assert model.recommend(["bread"], k=k) == top10[:k], name


@pytest.mark.parametrize("strategy", ["weighted", "confidence", "max_confidence"])
def test_all_aggregation_strategies_work(strategy, learnable_transactions):
    model = AssociationRecommender(
        strategy=strategy, mine_kwargs={"min_support": 0.05, "max_len": 2}
    ).fit(learnable_transactions)
    assert len(model.recommend(["bread"], k=3)) == 3


def test_unknown_strategy_raises(learnable_transactions):
    model = AssociationRecommender(
        strategy="nonsense", mine_kwargs={"min_support": 0.05}
    ).fit(learnable_transactions)
    with pytest.raises(ValueError, match="unknown strategy"):
        model.recommend(["bread"], k=3)


def test_explain_reports_scores(learnable_transactions):
    model = AssociationRecommender(mine_kwargs={"min_support": 0.05}).fit(
        learnable_transactions
    )
    frame = model.explain(["bread"], k=3)
    assert list(frame.columns) == ["rank", "item", "score", "popularity", "from_fallback"]
    assert len(frame) == 3


def test_empty_rules_falls_back_to_popularity(learnable_transactions):
    model = AssociationRecommender(rules=pd.DataFrame()).fit(learnable_transactions)
    assert len(model.recommend(["bread"], k=3)) == 3


# -------------------------------------------------------- temporal split
def test_split_is_chronological(dated_frame):
    train, test, meta = temporal_split(
        dated_frame, "InvoiceNo", "Description", "InvoiceDate", test_fraction=0.25
    )
    assert len(train) > 0 and len(test) > 0
    assert meta["n_train"] == len(train)
    assert 0.15 < meta["test_fraction_actual"] < 0.35


def test_split_honours_an_explicit_cut_date(dated_frame):
    train, test, meta = temporal_split(
        dated_frame, "InvoiceNo", "Description", "InvoiceDate",
        cut_date="2011-03-01",
    )
    assert meta["cut_date"] == "2011-03-01"
    assert len(train) + len(test) == dated_frame["InvoiceNo"].nunique()


def test_split_requires_a_date_column(dated_frame):
    with pytest.raises(KeyError):
        temporal_split(
            dated_frame.drop(columns=["InvoiceDate"]),
            "InvoiceNo", "Description", "InvoiceDate",
        )


# ------------------------------------------------------------ evaluation
def test_metrics_are_in_range(learnable_transactions):
    train = learnable_transactions[:800]
    test = learnable_transactions[800:]
    result = evaluate_recommender(build_recommender("association"), train, test, k=5)
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.coverage <= 1.0
    assert result.map_score <= result.hit_rate + 1e-9


def test_leave_one_out_identities(learnable_transactions):
    """With exactly one held-out item, recall == hit_rate and P@k == hit/k."""
    train, test = learnable_transactions[:800], learnable_transactions[800:]
    result = evaluate_recommender(build_recommender("cooccurrence"), train, test, k=5)
    assert result.recall == pytest.approx(result.hit_rate)
    assert result.precision == pytest.approx(result.hit_rate / 5)


def test_hit_rate_rises_with_k(learnable_transactions):
    train, test = learnable_transactions[:800], learnable_transactions[800:]
    model = build_recommender("cooccurrence").fit(train)
    results = evaluate_at_ks(model, train, test, k_values=(1, 5, 10), fit=False)
    rates = [r.hit_rate for r in results]
    assert rates == sorted(rates), "hit-rate must be monotone in k"


def test_one_pass_matches_separate_passes(learnable_transactions):
    """The optimisation must not change the numbers."""
    train, test = learnable_transactions[:800], learnable_transactions[800:]
    model = build_recommender("association").fit(train)
    batched = {r.k: r.hit_rate for r in evaluate_at_ks(model, train, test, (1, 5, 10), fit=False)}
    for k in (1, 5, 10):
        single = evaluate_recommender(model, train, test, k=k, fit=False)
        assert single.hit_rate == pytest.approx(batched[k]), f"k={k} diverged"


def test_association_outscores_popularity_on_held_out_data(learnable_transactions):
    train, test = learnable_transactions[:800], learnable_transactions[800:]
    frame = compare_recommenders(train, test, k_values=(5,))
    scores = frame.set_index("model")["hit_rate@k"]
    assert scores["association"] >= scores["popularity"]
    assert scores["association"] >= scores["random"]


def test_comparison_covers_every_model_and_k(learnable_transactions):
    train, test = learnable_transactions[:800], learnable_transactions[800:]
    frame = compare_recommenders(train, test, k_values=(1, 5))
    assert len(frame) == 8
    assert set(frame["model"]) == {"association", "cooccurrence", "popularity", "random"}


def test_lift_over_baseline_normalises_to_one(learnable_transactions):
    train, test = learnable_transactions[:800], learnable_transactions[800:]
    frame = lift_over_baseline(compare_recommenders(train, test, k_values=(5,)))
    popularity = frame.loc[frame["model"] == "popularity", "vs_popularity"]
    assert popularity.iloc[0] == pytest.approx(1.0)


def test_empty_test_set_is_handled(learnable_transactions):
    result = evaluate_recommender(
        build_recommender("popularity"), learnable_transactions, [], k=5
    )
    assert result.n_test_baskets == 0
    assert result.hit_rate == 0.0


def test_single_item_test_baskets_are_skipped(learnable_transactions):
    """A one-item basket has nothing to hold out."""
    result = evaluate_recommender(
        build_recommender("popularity"), learnable_transactions, [["bread"]] * 50, k=5
    )
    assert result.n_test_baskets == 0


def test_items_unseen_in_training_are_excluded(learnable_transactions):
    """No model can recover an item it never saw; counting it is noise."""
    test = [["brand-new-item", "another-new-one"]] * 20
    result = evaluate_recommender(
        build_recommender("popularity"), learnable_transactions, test, k=5
    )
    assert result.n_test_baskets == 0
