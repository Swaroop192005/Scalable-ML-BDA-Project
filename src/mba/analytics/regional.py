"""Country-by-country market basket analysis.

Pooling every market into one model hides the thing a merchandiser actually
wants. The UK is ~90% of Online Retail's baskets, so a global rules table is
effectively a UK rules table — and the rules that matter for France or
Germany never clear a global support threshold at all.

:func:`regional_analysis` mines each country separately at a locally
calibrated threshold, then :func:`compare_regions` contrasts the results so
the genuinely *local* associations stand out from the ones that hold
everywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from mba.config import Config
from mba.data.basket import encode_baskets
from mba.mining.base import mine
from mba.mining.rules import association_rules
from mba.mining.spurious import flag_spurious_rules

log = logging.getLogger(__name__)


@dataclass
class RegionResult:
    """Mining output for a single country."""

    country: str
    n_transactions: int
    n_items: int
    n_itemsets: int
    rules: pd.DataFrame
    top_products: pd.DataFrame = field(default_factory=pd.DataFrame)
    revenue: float = 0.0
    runtime_sec: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.country:<20s} {self.n_transactions:>7,} baskets  "
            f"{self.n_items:>5,} items  {self.n_itemsets:>5,} itemsets  "
            f"{len(self.rules):>5,} rules"
        )


def regional_analysis(
    frame: pd.DataFrame,
    cfg: Config,
    countries: Iterable[str] | None = None,
    min_support: float | None = None,
    min_confidence: float | None = None,
    min_transactions: int | None = None,
    top_n_countries: int | None = None,
    algorithm: str | None = None,
) -> dict[str, RegionResult]:
    """Mine rules separately for each country with enough data.

    Countries below ``analytics.regional.min_transactions`` are skipped —
    below roughly 150 baskets, support estimates are too noisy to act on.
    """
    import time

    schema = cfg["schema"]
    country_col = schema["country_col"]
    invoice_col = schema["invoice_col"]
    item_col = schema["item_col"]
    qty_col, price_col = schema["quantity_col"], schema["price_col"]

    settings = cfg.get("analytics.regional", {})
    min_support = min_support if min_support is not None else float(settings.get("min_support", 0.03))
    min_confidence = (
        min_confidence if min_confidence is not None else float(settings.get("min_confidence", 0.3))
    )
    min_transactions = (
        min_transactions if min_transactions is not None else int(settings.get("min_transactions", 150))
    )
    top_n_countries = (
        top_n_countries if top_n_countries is not None else int(settings.get("top_n_countries", 8))
    )
    algorithm = algorithm or cfg.get("mining.algorithm", "fpgrowth")
    max_len = int(cfg.get("mining.max_len", 3))
    min_lift = float(cfg.get("mining.min_lift", 1.0))

    if country_col not in frame.columns:
        log.warning("no %r column; skipping regional analysis", country_col)
        return {}

    basket_counts = frame.groupby(country_col, observed=True)[invoice_col].nunique()
    eligible = basket_counts.loc[basket_counts >= min_transactions].sort_values(ascending=False)

    if countries is None:
        selected = list(eligible.head(top_n_countries).index)
    else:
        selected = [c for c in countries if c in basket_counts.index]

    skipped = sorted(set(basket_counts.index) - set(eligible.index))
    if skipped:
        log.info("skipping %d countries below %d baskets", len(skipped), min_transactions)

    results: dict[str, RegionResult] = {}
    for country in selected:
        t0 = time.perf_counter()
        subset = frame.loc[frame[country_col] == country]
        basket = encode_baskets(subset, cfg, min_item_count=2)
        if basket.n_transactions < min_transactions or basket.n_items == 0:
            continue

        mined = mine(
            basket,
            algorithm=algorithm,
            min_support=min_support,
            max_len=max_len,
            track_memory=False,
        )
        rules = association_rules(
            mined, min_confidence=min_confidence, min_lift=min_lift, sort_by="lift"
        )
        if not rules.empty:
            rules, _ = flag_spurious_rules(rules, cfg, n_transactions=basket.n_transactions)
            rules.insert(0, "country", country)

        revenue = 0.0
        top_products = pd.DataFrame()
        if qty_col in subset.columns and price_col in subset.columns:
            enriched = subset.assign(Revenue=subset[qty_col] * subset[price_col])
            revenue = float(enriched["Revenue"].sum())
            top_products = (
                enriched.groupby(item_col, observed=True)
                .agg(transactions=(invoice_col, "nunique"), revenue=("Revenue", "sum"))
                .sort_values("transactions", ascending=False)
                .head(10)
                .reset_index()
            )

        result = RegionResult(
            country=country,
            n_transactions=basket.n_transactions,
            n_items=basket.n_items,
            n_itemsets=len(mined),
            rules=rules,
            top_products=top_products,
            revenue=revenue,
            runtime_sec=time.perf_counter() - t0,
        )
        results[country] = result
        log.info("%s", result.summary())

    return results


def compare_regions(results: dict[str, RegionResult], top_n: int = 5) -> pd.DataFrame:
    """One row per country: scale, rule yield and its strongest association."""
    rows: list[dict[str, Any]] = []
    for country, result in results.items():
        rules = result.rules
        best = ""
        best_lift = float("nan")
        if not rules.empty:
            top = rules.sort_values("lift", ascending=False).iloc[0]
            best = f"{', '.join(sorted(top['antecedents']))} => {', '.join(sorted(top['consequents']))}"
            best_lift = float(top["lift"])

        high_value = (
            int((~rules["is_spurious"]).sum()) if "is_spurious" in rules.columns else len(rules)
        )
        rows.append(
            {
                "country": country,
                "transactions": result.n_transactions,
                "items": result.n_items,
                "revenue": result.revenue,
                "avg_basket_value": result.revenue / result.n_transactions
                if result.n_transactions
                else 0.0,
                "itemsets": result.n_itemsets,
                "rules": len(rules),
                "high_value_rules": high_value,
                "top_rule": best,
                "top_rule_lift": best_lift,
                "runtime_sec": round(result.runtime_sec, 3),
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values("transactions", ascending=False).reset_index(drop=True) if not frame.empty else frame


def distinctive_rules(
    results: dict[str, RegionResult],
    reference: str = "United Kingdom",
    min_lift_ratio: float = 1.5,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rules that are much stronger in one market than in the reference one.

    A rule present everywhere is a product fact; a rule with 3x the lift in
    France is a *local* fact, and that is what a regional campaign is for.
    """
    reference_lift: dict[tuple[frozenset, frozenset], float] = {}
    if reference in results and not results[reference].rules.empty:
        for _, row in results[reference].rules.iterrows():
            reference_lift[(row["antecedents"], row["consequents"])] = float(row["lift"])

    rows: list[dict[str, Any]] = []
    for country, result in results.items():
        if country == reference or result.rules.empty:
            continue
        for _, row in result.rules.iterrows():
            key = (row["antecedents"], row["consequents"])
            base = reference_lift.get(key)
            ratio = float(row["lift"]) / base if base else float("inf")
            if ratio >= min_lift_ratio:
                rows.append(
                    {
                        "country": country,
                        "antecedents": ", ".join(sorted(row["antecedents"])),
                        "consequents": ", ".join(sorted(row["consequents"])),
                        "local_lift": float(row["lift"]),
                        "reference_lift": base,
                        "lift_ratio": ratio,
                        "local_confidence": float(row["confidence"]),
                        "local_support": float(row["support"]),
                        "exists_in_reference": base is not None,
                    }
                )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).sort_values(
        ["country", "local_support"], ascending=[True, False]
    )
    return frame.groupby("country", observed=True).head(top_n).reset_index(drop=True)
