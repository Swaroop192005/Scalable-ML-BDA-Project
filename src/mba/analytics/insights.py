"""Turning rules into business recommendations.

A rules table is a statistical artefact. A merchandiser needs a *decision*:
what to bundle, what to place together, what to recommend at checkout, and
roughly what it is worth. This module applies an explicit rulebook to the
mined rules and emits typed :class:`Insight` objects, each carrying the
evidence it rests on so the recommendation can be challenged.

Insight types
-------------
``bundle``
    High lift *and* high confidence in both directions — the items behave
    like one product. Package them.
``cross_sell``
    Strong one-way implication: buying A reliably predicts C, but not the
    reverse. Recommend C once A is in the basket.
``placement``
    High confidence, moderate lift, both items individually popular.
    Physical or on-page adjacency, not a discount.
``anchor``
    A hub product appearing across many rules; it pulls whole baskets.
``untapped``
    Strong association whose *joint* support is low — the pairing works when
    it happens, but it rarely happens. The clearest promotion candidate.
``regional``
    An association materially stronger in one market than in the reference.

Revenue estimates are deliberately labelled *indicative*. They extrapolate
from observed basket counts and average item revenue under the stated
attach-rate assumption; they are a sizing aid, not a forecast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from mba.config import Config

log = logging.getLogger(__name__)


@dataclass
class Insight:
    """One actionable recommendation, with the evidence behind it."""

    kind: str
    title: str
    detail: str
    action: str
    priority: float
    evidence: dict[str, Any] = field(default_factory=dict)
    estimated_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "action": self.action,
            "priority": round(self.priority, 4),
            "estimated_value": (
                round(self.estimated_value, 2) if self.estimated_value is not None else None
            ),
            **{f"evidence_{k}": v for k, v in self.evidence.items()},
        }

    def as_markdown(self) -> str:
        value = (
            f"\n- *Indicative annual value:* £{self.estimated_value:,.0f}"
            if self.estimated_value
            else ""
        )
        return (
            f"### {self.title}\n"
            f"**Type:** `{self.kind}`  |  **Priority:** {self.priority:.2f}\n\n"
            f"{self.detail}\n\n"
            f"- **Recommended action:** {self.action}{value}\n"
        )


def _items(value: Iterable[str]) -> str:
    return ", ".join(sorted(value))


def _item_revenue_lookup(frame: pd.DataFrame | None, cfg: Config) -> dict[str, float]:
    """Mean revenue per basket containing each item, used for value sizing."""
    if frame is None or frame.empty:
        return {}
    schema = cfg["schema"]
    item, qty, price = schema["item_col"], schema["quantity_col"], schema["price_col"]
    invoice = schema["invoice_col"]
    if not {item, qty, price}.issubset(frame.columns):
        return {}
    enriched = frame.assign(_rev=frame[qty] * frame[price])
    grouped = enriched.groupby(item, observed=True).agg(
        total=("_rev", "sum"), baskets=(invoice, "nunique")
    )
    return (grouped["total"] / grouped["baskets"].clip(lower=1)).to_dict()


def _estimate_value(
    rule: pd.Series,
    revenue_lookup: dict[str, float],
    n_transactions: int,
    attach_uplift: float = 0.15,
) -> float | None:
    """Indicative uplift from acting on a rule.

    Model: the baskets that already contain the antecedent but *not* the
    consequent are the addressable population. Assume a modest *attach_uplift*
    conversion on them, valued at the consequent's mean per-basket revenue.
    Deliberately conservative — the point is relative sizing between rules.
    """
    if not revenue_lookup or not n_transactions:
        return None
    consequent_value = sum(revenue_lookup.get(i, 0.0) for i in rule["consequents"])
    if consequent_value <= 0:
        return None
    addressable = (float(rule["antecedent_support"]) - float(rule["support"])) * n_transactions
    return max(0.0, addressable * attach_uplift * consequent_value)


def generate_insights(
    rules: pd.DataFrame,
    cfg: Config,
    frame: pd.DataFrame | None = None,
    n_transactions: int | None = None,
    graph_summary: dict[str, Any] | None = None,
    regional: pd.DataFrame | None = None,
    top_n: int | None = None,
) -> list[Insight]:
    """Apply the rulebook and return insights, highest priority first."""
    settings = cfg.get("analytics.insights", {})
    top_n = top_n if top_n is not None else int(settings.get("top_n", 12))
    min_lift_bundle = float(settings.get("min_lift_for_bundle", 1.5))
    min_conf_placement = float(settings.get("min_confidence_for_placement", 0.5))

    if rules is None or rules.empty:
        return []

    # Act only on rules that survived triage, when triage has been run.
    candidates = rules.loc[~rules["is_spurious"]] if "is_spurious" in rules.columns else rules
    if candidates.empty:
        candidates = rules

    if n_transactions is None and "count" in candidates.columns:
        first = candidates.iloc[0]
        n_transactions = int(round(first["count"] / first["support"])) if first["support"] else 0

    revenue_lookup = _item_revenue_lookup(frame, cfg)
    insights: list[Insight] = []

    # Reverse-direction confidence, to tell a bundle from a one-way cross-sell.
    reverse_confidence: dict[tuple[frozenset, frozenset], float] = {
        (row["antecedents"], row["consequents"]): float(row["confidence"])
        for _, row in rules.iterrows()
    }

    for _, rule in candidates.iterrows():
        antecedents, consequents = rule["antecedents"], rule["consequents"]
        lift = float(rule["lift"])
        confidence = float(rule["confidence"])
        support = float(rule["support"])
        reverse = reverse_confidence.get((consequents, antecedents), 0.0)
        value = _estimate_value(rule, revenue_lookup, n_transactions or 0)
        baskets = int(rule.get("count", round(support * (n_transactions or 0))))

        evidence = {
            "antecedents": _items(antecedents),
            "consequents": _items(consequents),
            "support": round(support, 5),
            "confidence": round(confidence, 4),
            "lift": round(lift, 3),
            "reverse_confidence": round(reverse, 4),
            "baskets": baskets,
        }

        if lift >= min_lift_bundle and confidence >= 0.4 and reverse >= 0.4:
            insights.append(
                Insight(
                    kind="bundle",
                    title=f"Bundle: {_items(antecedents)} + {_items(consequents)}",
                    detail=(
                        f"These appear together in {baskets:,} baskets ({support:.1%} of all "
                        f"orders) at {lift:.1f}x the rate independence predicts. The implication "
                        f"runs both ways ({confidence:.0%} forward, {reverse:.0%} reverse), so "
                        "customers treat them as one purchase rather than one leading to the other."
                    ),
                    action=(
                        "Create a single bundled SKU at a small discount to the combined price. "
                        "Two-way implication means the bundle cannibalises little — most of these "
                        "customers were buying both anyway, and the bundle captures the ones who "
                        "were not."
                    ),
                    priority=lift * confidence * (support ** 0.5) * 10,
                    evidence=evidence,
                    estimated_value=value,
                )
            )
        elif lift >= min_lift_bundle and confidence >= 0.3 and reverse < 0.3:
            insights.append(
                Insight(
                    kind="cross_sell",
                    title=f"Cross-sell {_items(consequents)} to buyers of {_items(antecedents)}",
                    detail=(
                        f"{confidence:.0%} of baskets containing {_items(antecedents)} also contain "
                        f"{_items(consequents)} — {lift:.1f}x the base rate. The reverse holds only "
                        f"{reverse:.0%} of the time, so this is a genuine one-way implication: "
                        f"{_items(antecedents)} is the trigger product."
                    ),
                    action=(
                        f"Trigger a recommendation for {_items(consequents)} when "
                        f"{_items(antecedents)} enters the basket — product page, cart page and "
                        "post-purchase email. Do not run it in reverse; the evidence does not "
                        "support it."
                    ),
                    priority=lift * confidence * (support ** 0.5) * 8,
                    evidence=evidence,
                    estimated_value=value,
                )
            )
        elif confidence >= min_conf_placement and lift >= 1.2:
            insights.append(
                Insight(
                    kind="placement",
                    title=f"Co-locate {_items(antecedents)} with {_items(consequents)}",
                    detail=(
                        f"High confidence ({confidence:.0%}) with moderate lift ({lift:.1f}x): both "
                        "items are individually popular and frequently bought together. That is an "
                        "adjacency signal, not a discount signal."
                    ),
                    action=(
                        "Place these adjacently in-store and in the same on-site category block. "
                        "No price action needed — the association is already strong at full price."
                    ),
                    priority=confidence * lift * (support ** 0.5) * 5,
                    evidence=evidence,
                    estimated_value=value,
                )
            )

        # Strong association that rarely fires: the promotion opportunity.
        if lift >= 3.0 and support < 0.01 and confidence >= 0.3:
            insights.append(
                Insight(
                    kind="untapped",
                    title=f"Under-exploited pairing: {_items(antecedents)} -> {_items(consequents)}",
                    detail=(
                        f"Lift is {lift:.1f}x — when these are bought together the association is "
                        f"very strong — but the pairing occurs in only {baskets:,} baskets "
                        f"({support:.2%}). The affinity is real and almost entirely unexploited."
                    ),
                    action=(
                        "Run a targeted test: feature the pair in one campaign slot for a limited "
                        "period and measure attach rate. Low current support means high headroom "
                        "and low cannibalisation risk."
                    ),
                    priority=lift * confidence * 3,
                    evidence=evidence,
                    estimated_value=value,
                )
            )

    # -- hub products from the co-occurrence network ---------------------
    if graph_summary and graph_summary.get("top_hubs"):
        hubs = graph_summary["top_hubs"][:3]
        insights.append(
            Insight(
                kind="anchor",
                title=f"Anchor products: {', '.join(hubs)}",
                detail=(
                    f"In the co-occurrence network these sit at the centre of "
                    f"{graph_summary.get('n_communities', 0)} product communities across "
                    f"{graph_summary.get('n_nodes', 0)} products. They co-occur with more distinct "
                    "items than any other product, so they pull whole baskets rather than "
                    "single lines."
                ),
                action=(
                    "Guarantee availability and prime placement for these. A stockout on an anchor "
                    "costs more than its own margin — it suppresses the basket it would have "
                    "anchored. Use them as the entry product in acquisition offers."
                ),
                priority=50.0,
                evidence={
                    "hubs": ", ".join(hubs),
                    "bridges": ", ".join(graph_summary.get("top_bridges", [])[:3]),
                    "n_communities": graph_summary.get("n_communities", 0),
                },
            )
        )

    if graph_summary and graph_summary.get("top_bridges"):
        bridges = graph_summary["top_bridges"][:3]
        insights.append(
            Insight(
                kind="anchor",
                title=f"Category bridges: {', '.join(bridges)}",
                detail=(
                    "These products have the highest betweenness centrality — they sit on the "
                    "paths between otherwise separate product communities. Customers who buy them "
                    "are the ones who shop across categories."
                ),
                action=(
                    "Use these as the hook in cross-category campaigns, and target their buyers "
                    "for range-expansion offers. They are the cheapest route into a second "
                    "category."
                ),
                priority=45.0,
                evidence={"bridges": ", ".join(bridges)},
            )
        )

    # -- regional divergence ---------------------------------------------
    if regional is not None and not regional.empty:
        for _, row in regional.head(3).iterrows():
            insights.append(
                Insight(
                    kind="regional",
                    title=f"{row['country']}: {row['antecedents']} -> {row['consequents']}",
                    detail=(
                        f"This association reaches {row['local_lift']:.1f}x lift in "
                        f"{row['country']}"
                        + (
                            f", against {row['reference_lift']:.1f}x in the reference market"
                            if row.get("exists_in_reference")
                            else ", and does not clear the threshold in the reference market at all"
                        )
                        + f". Local confidence is {row['local_confidence']:.0%}."
                    ),
                    action=(
                        f"Localise merchandising for {row['country']} rather than inheriting the "
                        "global planogram. A single global rules table is dominated by the largest "
                        "market and would never surface this."
                    ),
                    priority=40.0 + float(row["local_lift"]),
                    evidence={
                        "country": row["country"],
                        "local_lift": round(float(row["local_lift"]), 3),
                        "local_confidence": round(float(row["local_confidence"]), 4),
                    },
                )
            )

    # Deduplicate (a rule can qualify as both cross-sell and untapped) and rank.
    seen: set[tuple[str, str]] = set()
    unique: list[Insight] = []
    for insight in sorted(insights, key=lambda i: i.priority, reverse=True):
        key = (insight.kind, insight.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(insight)

    log.info("generated %d insights (returning top %d)", len(unique), min(top_n, len(unique)))
    return unique[:top_n]


def insights_to_frame(insights: list[Insight]) -> pd.DataFrame:
    """Tabular form for CSV export and the dashboard."""
    if not insights:
        return pd.DataFrame(columns=["kind", "title", "detail", "action", "priority"])
    return pd.DataFrame([i.to_dict() for i in insights])


def insights_to_markdown(insights: list[Insight], title: str = "Business Insights") -> str:
    """Render the insight list as a Markdown report section."""
    if not insights:
        return f"# {title}\n\nNo insights cleared the configured thresholds.\n"

    by_kind: dict[str, list[Insight]] = {}
    for insight in insights:
        by_kind.setdefault(insight.kind, []).append(insight)

    parts = [
        f"# {title}",
        "",
        f"{len(insights)} recommendations, ranked by priority. Revenue figures are "
        "indicative sizing estimates under a 15% attach-rate assumption, not forecasts.",
        "",
    ]
    for kind in sorted(by_kind, key=lambda k: -max(i.priority for i in by_kind[k])):
        parts.append(f"## {kind.replace('_', ' ').title()}")
        parts.append("")
        parts.extend(i.as_markdown() for i in by_kind[kind])
    return "\n".join(parts)
