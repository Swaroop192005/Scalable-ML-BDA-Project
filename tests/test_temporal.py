"""Temporal rule drift."""

import numpy as np
import pandas as pd
import pytest

from mba.analytics.temporal import (
    CLASSES,
    _classify,
    rule_trajectory_matrix,
    seasonal_rules,
    temporal_analysis,
    trending_rules,
)


def _frame_from_spec(spec: dict[str, list[tuple[list[str], int]]]) -> pd.DataFrame:
    """Build a transaction frame from ``{month: [(basket, repeats), ...]}``."""
    rows = []
    counter = 0
    for month, baskets in spec.items():
        date = pd.Timestamp(f"{month}-15")
        for basket, repeats in baskets:
            for _ in range(repeats):
                counter += 1
                for item in basket:
                    rows.append(
                        {
                            "InvoiceNo": f"INV{counter:06d}",
                            "StockCode": item[:5],
                            "Description": item,
                            "Quantity": 1,
                            "InvoiceDate": date,
                            "UnitPrice": 5.0,
                            "CustomerID": float(counter % 50),
                            "Country": "United Kingdom",
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def seasonal_frame():
    """A pairing that only happens in December, over a 6-month span.

    Each month carries filler baskets so the period has enough volume to
    mine, plus noise so the December pair is not the only association.
    """
    spec = {}
    for month in ("2011-07", "2011-08", "2011-09", "2011-10", "2011-11", "2011-12"):
        baskets = [
            (["FILLER ONE", "FILLER TWO"], 150),
            (["NOISE A", "NOISE B"], 80),
        ]
        if month == "2011-12":
            baskets.append((["SANTA HAT", "REINDEER MUG"], 120))
        spec[month] = baskets
    return _frame_from_spec(spec)


# ------------------------------------------------------- classification
def test_classify_detects_a_rising_trend():
    lifts = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    label, slope, _ = _classify(lifts, 5, 5, 0.3, 0.6, 3)
    assert label == "emerging" and slope > 0


def test_classify_detects_a_falling_trend():
    lifts = np.array([6.0, 5.0, 4.0, 3.0, 2.0])
    label, slope, _ = _classify(lifts, 5, 5, 0.3, 0.6, 3)
    assert label == "decaying" and slope < 0


def test_classify_detects_stability():
    lifts = np.array([4.0, 4.1, 3.9, 4.05, 3.95])
    label, _, cv = _classify(lifts, 5, 5, 0.3, 0.6, 3)
    assert label == "stable" and cv < 0.1


def test_classify_detects_seasonality():
    """High variance with no net trend is seasonal."""
    lifts = np.array([1.0, 12.0, 1.0, 12.0, 1.0, 12.0])
    label, _, cv = _classify(lifts, 6, 6, 0.3, 0.6, 3)
    assert label == "seasonal" and cv >= 0.6


def test_low_presence_with_steady_lift_is_intermittent_not_seasonal():
    """Guards a real mislabelling: near-threshold rules are not seasonal."""
    lifts = np.array([5.0, 5.1, 4.9])
    label, _, cv = _classify(lifts, 3, 12, 0.3, 0.6, 3)
    assert label == "intermittent", "a steady, rarely-present rule is not seasonal"
    assert cv < 0.6


def test_too_few_periods_is_sporadic():
    label, _, _ = _classify(np.array([5.0, 6.0]), 2, 12, 0.3, 0.6, 3)
    assert label == "sporadic"


def test_classify_handles_degenerate_input():
    label, _, _ = _classify(np.array([0.0, 0.0, 0.0]), 3, 3, 0.3, 0.6, 3)
    assert label in CLASSES


# ---------------------------------------------------------- end to end
def test_periods_are_mined_independently(seasonal_frame, cfg):
    result = temporal_analysis(
        seasonal_frame, cfg, freq="M", min_support=0.05,
        min_confidence=0.3, min_basket_count=50,
    )
    assert result.n_periods == 6
    assert set(result.period_rules) == set(result.periods)
    assert result.periods == sorted(result.periods), "periods must be chronological"


def test_a_december_only_rule_is_not_called_stable(seasonal_frame, cfg):
    result = temporal_analysis(
        seasonal_frame, cfg, freq="M", min_support=0.05,
        min_confidence=0.3, min_basket_count=50,
    )
    santa = result.drift.loc[
        result.drift["antecedents"].str.contains("SANTA HAT", na=False)
    ]
    assert not santa.empty, "the planted December pairing was not found"
    row = santa.iloc[0]
    assert row["periods_present"] == 1
    assert row["peak_period"] == "2011-12"
    assert row["rule_class"] in {"sporadic", "seasonal", "intermittent"}
    assert row["rule_class"] != "stable"


def test_a_year_round_rule_is_stable(seasonal_frame, cfg):
    result = temporal_analysis(
        seasonal_frame, cfg, freq="M", min_support=0.05,
        min_confidence=0.3, min_basket_count=50,
    )
    filler = result.drift.loc[
        result.drift["antecedents"].str.contains("FILLER ONE", na=False)
    ]
    assert not filler.empty
    assert filler.iloc[0]["periods_present"] == 6


def test_drift_table_is_internally_consistent(seasonal_frame, cfg):
    result = temporal_analysis(
        seasonal_frame, cfg, freq="M", min_support=0.05,
        min_confidence=0.3, min_basket_count=50,
    )
    drift = result.drift
    assert (drift["periods_present"] <= drift["periods_total"]).all()
    assert (drift["min_lift"] <= drift["mean_lift"] + 1e-9).all()
    assert (drift["mean_lift"] <= drift["max_lift"] + 1e-9).all()
    assert (drift["lift_cv"] >= 0).all()
    assert drift["rule_class"].isin(CLASSES).all()
    assert (drift["presence"].between(0, 1)).all()


def test_partial_periods_are_flagged(cfg):
    """A truncated final period must not read as a collapse in demand."""
    rows = []
    for day in list(range(1, 29)) + [1, 2]:          # full Jan, 2 days of Feb
        month = 1 if day <= 28 and len(rows) < 28 * 60 else 2
        date = pd.Timestamp(f"2011-{month:02d}-{day:02d}")
        for order in range(60):
            invoice = f"{month:02d}{day:02d}{order:03d}"
            for item in ("ALPHA MUG", "BETA BOWL"):
                rows.append(
                    {
                        "InvoiceNo": invoice, "StockCode": item[:5], "Description": item,
                        "Quantity": 1, "InvoiceDate": date, "UnitPrice": 5.0,
                        "CustomerID": 1.0, "Country": "UK",
                    }
                )
    result = temporal_analysis(
        pd.DataFrame(rows), cfg, freq="M", min_support=0.1,
        min_confidence=0.1, min_basket_count=10,
    )
    assert "partial" in result.period_stats.columns
    assert "days_covered" in result.period_stats.columns


def test_helpers_return_frames(seasonal_frame, cfg):
    result = temporal_analysis(
        seasonal_frame, cfg, freq="M", min_support=0.05,
        min_confidence=0.3, min_basket_count=50,
    )
    assert isinstance(seasonal_rules(result, 5), pd.DataFrame)
    assert isinstance(trending_rules(result, 5), pd.DataFrame)
    matrix = rule_trajectory_matrix(result, top_n=5)
    if not matrix.empty:
        assert list(matrix.columns) == result.periods


def test_requires_a_date_column(seasonal_frame, cfg):
    with pytest.raises(KeyError):
        temporal_analysis(seasonal_frame.drop(columns=["InvoiceDate"]), cfg)


def test_sparse_data_yields_no_periods(cfg):
    frame = _frame_from_spec({"2011-01": [(["ALPHA MUG", "BETA BOWL"], 3)]})
    result = temporal_analysis(frame, cfg, freq="M", min_basket_count=1000)
    assert result.n_periods == 0
    assert result.drift.empty
