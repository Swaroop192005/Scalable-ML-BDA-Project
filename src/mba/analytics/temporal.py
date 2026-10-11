"""Temporal rule drift: which associations hold, and when.

A rules table mined over a whole year states that an association held *on
average*. That hides the two cases a merchandiser most needs to tell apart:

* A rule at 3x lift every month of the year — a stable product fact worth
  building a permanent planogram around.
* A rule at 20x lift in November and December and absent the rest of the
  year — a seasonal fact. Averaged over twelve months it may not even clear
  the global support threshold, and if it does, acting on it in March is
  wasted effort.

This dataset runs December 2010 to December 2011, so it contains exactly
one Christmas peak and the comparison is available.

:func:`temporal_analysis` mines each period independently, tracks every
rule's lift across periods, and classifies it:

``stable``        present in most periods with little variation
``emerging``      lift trending clearly upward — act on it now
``decaying``      lift trending clearly downward — stop acting on it
``seasonal``      high variance, concentrated in particular periods
``intermittent``  appears in only part of the span but at steady strength
                  when present — usually a rule hovering near the support
                  threshold rather than a genuine seasonal effect
``sporadic``      appears in too few periods to call

Classification uses the **coefficient of variation** (std / mean) for
seasonality and the **ordinary-least-squares slope normalised by mean lift**
for trend, so both are scale-free and comparable across rules of very
different strength.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mba.config import Config

log = logging.getLogger(__name__)

#: Rule classes, in reporting priority order.
CLASSES = ("stable", "emerging", "decaying", "seasonal", "intermittent", "sporadic")


@dataclass
class TemporalResult:
    """Per-period rule tables plus the drift classification."""

    periods: list[str] = field(default_factory=list)
    period_rules: dict[str, pd.DataFrame] = field(default_factory=dict)
    drift: pd.DataFrame = field(default_factory=pd.DataFrame)
    period_stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    def by_class(self, name: str) -> pd.DataFrame:
        if self.drift.empty:
            return self.drift
        return self.drift.loc[self.drift["rule_class"] == name]

    def to_dict(self) -> dict[str, Any]:
        counts = (
            self.drift["rule_class"].value_counts().to_dict()
            if not self.drift.empty else {}
        )
        return {
            "n_periods": self.n_periods,
            "periods": self.periods,
            "n_distinct_rules": len(self.drift),
            "by_class": {c: int(counts.get(c, 0)) for c in CLASSES},
        }

    def summary(self) -> str:
        if self.drift.empty:
            return f"{self.n_periods} periods, no rules found"
        counts = self.drift["rule_class"].value_counts()
        parts = ", ".join(f"{c}={int(counts.get(c, 0))}" for c in CLASSES if counts.get(c, 0))
        return (
            f"{self.n_periods} periods -> {len(self.drift):,} distinct rules [{parts}]"
        )


def _classify(
    lifts: np.ndarray,
    n_present: int,
    n_periods: int,
    trend_threshold: float,
    seasonal_cv: float,
    min_periods: int,
    trend_r2: float = 0.5,
) -> tuple[str, float, float]:
    """Return ``(class, normalised_slope, coefficient_of_variation)``.

    A trend label requires **two** things: a slope large enough to matter,
    *and* a linear fit that actually explains the variation (R-squared above
    *trend_r2*). The R-squared gate is not decoration. An alternating series
    like ``[1, 12, 1, 12, 1, 12]`` has a large OLS slope purely because it
    happens to end high, and without the gate it would be reported as
    "emerging" — the opposite of the truth, and exactly the kind of rule
    someone would build a campaign on.

    Check order: sporadic first (too few points for any statistic), then
    trend, then seasonality, then intermittency.
    """
    if n_present < min_periods:
        return "sporadic", 0.0, 0.0

    mean = float(lifts.mean())
    if mean <= 0:
        return "sporadic", 0.0, 0.0

    cv = float(lifts.std() / mean)

    # OLS slope over period index, normalised by the mean so it is scale-free.
    x = np.arange(len(lifts), dtype="float64")
    if len(lifts) > 1:
        slope, intercept = np.polyfit(x, lifts, 1)
        residual = lifts - (slope * x + intercept)
        total_variation = float(((lifts - mean) ** 2).sum())
        r_squared = (
            1.0 - float((residual ** 2).sum()) / total_variation
            if total_variation > 0 else 0.0
        )
    else:
        slope, r_squared = 0.0, 0.0

    # Slope across the full span, as a fraction of mean lift.
    normalised = (float(slope) * max(len(lifts) - 1, 1)) / mean

    if abs(normalised) >= trend_threshold and r_squared >= trend_r2:
        return ("emerging" if normalised > 0 else "decaying"), normalised, cv
    # Genuinely seasonal: lift swings widely without a consistent direction.
    if cv >= seasonal_cv:
        return "seasonal", normalised, cv
    # Present in only part of the span, but at steady strength when it is.
    # That is a rule sitting near the support threshold, not a seasonal
    # effect — calling it seasonal would invite a campaign built on nothing.
    if n_present < n_periods * 0.6:
        return "intermittent", normalised, cv
    return "stable", normalised, cv


def temporal_analysis(
    frame: pd.DataFrame,
    cfg: Config,
    freq: str | None = None,
    min_support: float | None = None,
    min_confidence: float | None = None,
    min_periods: int | None = None,
    min_basket_count: int = 100,
) -> TemporalResult:
    """Mine rules per period and classify how each one drifts.

    Each period is mined **independently** at the same relative support, so
    a rule's lift in March is computed only from March's baskets. Comparing
    lift across periods is then meaningful; comparing raw support would not
    be, because period sizes differ.
    """
    from mba.data.basket import encode_baskets
    from mba.mining.base import mine
    from mba.mining.rules import association_rules

    settings = cfg.get("analytics.temporal", {})
    freq = freq or settings.get("freq", "M")
    min_support = (
        min_support if min_support is not None else float(settings.get("min_support", 0.03))
    )
    min_confidence = (
        min_confidence if min_confidence is not None
        else float(settings.get("min_confidence", 0.3))
    )
    min_periods = (
        min_periods if min_periods is not None else int(settings.get("min_periods", 3))
    )
    trend_threshold = float(settings.get("trend_threshold", 0.30))
    seasonal_cv = float(settings.get("seasonal_cv", 0.60))
    trend_r2 = float(settings.get("trend_r2", 0.50))

    schema = cfg["schema"]
    date_col, invoice_col = schema["date_col"], schema["invoice_col"]
    if date_col not in frame.columns:
        raise KeyError(f"temporal analysis needs a {date_col!r} column")

    working = frame.loc[frame[date_col].notna()].copy()
    # pandas 3 renamed the month alias; 'M' still reads naturally in config.
    alias = {"M": "ME", "Q": "QE", "W": "W"}.get(freq, freq)
    try:
        working["_period"] = working[date_col].dt.to_period(freq).astype(str)
    except ValueError:
        working["_period"] = (
            working[date_col].dt.to_period(alias.rstrip("E")).astype(str)
        )

    result = TemporalResult()
    period_rows: list[dict[str, Any]] = []

    for period, subset in working.groupby("_period", observed=True, sort=True):
        basket = encode_baskets(subset, cfg, min_item_count=2)
        if basket.n_transactions < min_basket_count:
            log.info("period %s skipped (%d baskets)", period, basket.n_transactions)
            continue

        itemsets = mine(
            basket, algorithm=cfg.get("mining.algorithm", "fpgrowth"),
            min_support=min_support, max_len=int(cfg.get("mining.max_len", 3)),
            track_memory=False,
        )
        rules = association_rules(
            itemsets, min_confidence=min_confidence,
            min_lift=float(cfg.get("mining.min_lift", 1.0)), sort_by="lift",
        )
        result.periods.append(str(period))
        result.period_rules[str(period)] = rules
        span = subset[date_col]
        period_rows.append(
            {
                "period": str(period),
                "days_covered": int((span.max() - span.min()).days) + 1,
                "baskets": basket.n_transactions,
                "items": basket.n_items,
                "itemsets": len(itemsets),
                "rules": len(rules),
                "revenue": float(
                    (subset[schema["quantity_col"]] * subset[schema["price_col"]]).sum()
                ) if schema["price_col"] in subset.columns else 0.0,
            }
        )
        log.info("period %s: %5d baskets -> %4d rules", period, basket.n_transactions, len(rules))

    result.period_stats = pd.DataFrame(period_rows)
    if not result.period_stats.empty:
        # The first and last period of a span are usually truncated — this
        # dataset ends on 9 December, so December 2011 is a third of a
        # month. Flagging it stops a short final period being read as a
        # collapse in demand, and warns that it pulls on every trend fit.
        typical = result.period_stats["days_covered"].median()
        result.period_stats["partial"] = (
            result.period_stats["days_covered"] < 0.75 * typical
        )
        partial = result.period_stats.loc[result.period_stats["partial"], "period"].tolist()
        if partial:
            log.warning(
                "partial periods (fewer than 75%% of %d days): %s — these pull on "
                "trend estimates",
                int(typical), ", ".join(partial),
            )

    if not result.periods:
        log.warning("no period had enough baskets to mine")
        return result

    # ---- collect each rule's lift trajectory across periods -------------
    trajectories: dict[tuple[frozenset, frozenset], dict[str, float]] = {}
    for period, rules in result.period_rules.items():
        if rules is None or rules.empty:
            continue
        for row in rules.itertuples():
            key = (frozenset(row.antecedents), frozenset(row.consequents))
            trajectories.setdefault(key, {})[period] = float(row.lift)

    ordered = result.periods
    rows: list[dict[str, Any]] = []
    for (antecedents, consequents), by_period in trajectories.items():
        lifts = np.array([by_period[p] for p in ordered if p in by_period])
        n_present = len(lifts)
        rule_class, slope, cv = _classify(
            lifts, n_present, len(ordered), trend_threshold, seasonal_cv,
            min_periods, trend_r2,
        )
        peak_period = max(by_period, key=by_period.get)
        rows.append(
            {
                "antecedents": ", ".join(sorted(antecedents)),
                "consequents": ", ".join(sorted(consequents)),
                "rule_class": rule_class,
                "periods_present": n_present,
                "periods_total": len(ordered),
                "presence": round(n_present / len(ordered), 3),
                "mean_lift": round(float(lifts.mean()), 3),
                "min_lift": round(float(lifts.min()), 3),
                "max_lift": round(float(lifts.max()), 3),
                "lift_cv": round(cv, 3),
                "trend": round(slope, 3),
                "peak_period": peak_period,
                "peak_lift": round(float(by_period[peak_period]), 3),
                "trajectory": {p: round(by_period[p], 2) for p in ordered if p in by_period},
            }
        )

    drift = pd.DataFrame(rows)
    if not drift.empty:
        drift = drift.sort_values(
            ["rule_class", "mean_lift"], ascending=[True, False]
        ).reset_index(drop=True)
    result.drift = drift

    log.info("%s", result.summary())
    return result


def seasonal_rules(result: TemporalResult, top_n: int = 10) -> pd.DataFrame:
    """Rules concentrated in particular periods, strongest peak first."""
    seasonal = result.by_class("seasonal")
    if seasonal.empty:
        return seasonal
    return seasonal.nlargest(top_n, "peak_lift").reset_index(drop=True)


def trending_rules(result: TemporalResult, top_n: int = 10) -> pd.DataFrame:
    """Emerging and decaying rules, strongest trend magnitude first."""
    if result.drift.empty:
        return result.drift
    trending = result.drift.loc[result.drift["rule_class"].isin(["emerging", "decaying"])]
    if trending.empty:
        return trending
    return (
        trending.assign(_magnitude=trending["trend"].abs())
        .nlargest(top_n, "_magnitude")
        .drop(columns="_magnitude")
        .reset_index(drop=True)
    )


def rule_trajectory_matrix(result: TemporalResult, top_n: int = 20) -> pd.DataFrame:
    """Rules x periods matrix of lift, for a heatmap.

    Blank cells mean the rule did not clear the threshold that period, which
    is itself information — it is not the same as a lift of zero.
    """
    if result.drift.empty:
        return pd.DataFrame()

    top = result.drift.nlargest(top_n, "mean_lift")
    rows = {}
    for _, row in top.iterrows():
        label = f"{row['antecedents']} → {row['consequents']}"
        rows[label[:60]] = row["trajectory"]
    return pd.DataFrame(rows).T.reindex(columns=result.periods)
