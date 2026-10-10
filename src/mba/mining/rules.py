"""Association-rule generation and interestingness metrics.

For a rule ``A => C`` over *N* transactions, with ``s(X)`` the support of
itemset *X*:

================  =====================================================  ==========================
Metric            Definition                                             Reads as
================  =====================================================  ==========================
support           ``s(A u C)``                                           how often the pair happens
confidence        ``s(A u C) / s(A)``                                    P(C | A)
lift              ``conf / s(C)``                                        lift over base rate; 1 = independent
leverage          ``s(A u C) - s(A)s(C)``                                excess co-occurrence, absolute
conviction        ``(1 - s(C)) / (1 - conf)``                            how often the rule would be wrong by chance
zhangs_metric     ``(conf - s(C)) / max(conf(1-s(C)), s(C)(1-conf))``    association in [-1, 1], handles rare items
================  =====================================================  ==========================

Why more than lift: lift is unbounded and unstable for rare consequents, so a
pair seen four times can top the chart. Leverage keeps the *absolute* size of
the effect in view, conviction is sensitive to the direction of implication,
and Zhang's metric is bounded and stays meaningful when supports are skewed.
Ranking on several of them is what separates a real bundle from an artefact —
see :mod:`mba.mining.spurious`.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from mba.mining.base import MiningResult

log = logging.getLogger(__name__)

#: Every metric the generator knows how to compute.
ALL_METRICS = (
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "zhangs_metric",
)

_EMPTY_COLUMNS = [
    "antecedents",
    "consequents",
    "antecedent_support",
    "consequent_support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "zhangs_metric",
    "count",
    "antecedent_len",
    "consequent_len",
]


def empty_rules_frame() -> pd.DataFrame:
    """A correctly-typed empty rules table."""
    frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in _EMPTY_COLUMNS})
    frame["antecedents"] = pd.Series(dtype="object")
    frame["consequents"] = pd.Series(dtype="object")
    frame["count"] = pd.Series(dtype="int64")
    frame["antecedent_len"] = pd.Series(dtype="int64")
    frame["consequent_len"] = pd.Series(dtype="int64")
    return frame


def rule_metrics(
    support_ac: np.ndarray | float,
    support_a: np.ndarray | float,
    support_c: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Compute every interestingness metric from the three supports.

    Vectorised over numpy arrays; division-by-zero edge cases are resolved to
    the mathematically meaningful limits rather than ``inf``/``nan`` where
    one exists:

    * ``confidence`` is 0 when the antecedent never occurs.
    * ``conviction`` is ``inf`` for a perfectly-confident rule (conf == 1),
      which is the correct limit — the rule has no counter-examples.
    * ``zhangs_metric`` is 0 when both terms of the denominator vanish.
    """
    s_ac = np.asarray(support_ac, dtype="float64")
    s_a = np.asarray(support_a, dtype="float64")
    s_c = np.asarray(support_c, dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        confidence = np.where(s_a > 0, s_ac / s_a, 0.0)
        lift = np.where(s_c > 0, confidence / s_c, 0.0)
        leverage = s_ac - (s_a * s_c)

        denom = 1.0 - confidence
        conviction = np.where(
            denom > 0,
            np.divide(1.0 - s_c, denom, out=np.full_like(s_ac, np.inf), where=denom > 0),
            np.inf,
        )
        # A rule whose consequent is universal (s_c == 1) is vacuous, not
        # infinitely convincing.
        conviction = np.where(s_c >= 1.0, 1.0, conviction)

        numerator = confidence - s_c
        zhang_denom = np.maximum(confidence * (1.0 - s_c), s_c * (1.0 - confidence))
        zhangs = np.where(zhang_denom > 0, numerator / zhang_denom, 0.0)

    return {
        "support": s_ac,
        "confidence": confidence,
        "lift": lift,
        "leverage": leverage,
        "conviction": conviction,
        "zhangs_metric": zhangs,
    }


def association_rules(
    itemsets: MiningResult | pd.DataFrame,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    metrics: Sequence[str] | None = None,
    max_antecedent_len: int | None = None,
    n_transactions: int | None = None,
    sort_by: str = "lift",
) -> pd.DataFrame:
    """Generate association rules from frequent itemsets.

    Every frequent itemset of size >= 2 is split into each non-empty proper
    antecedent/consequent partition; a partition survives when it clears both
    *min_confidence* and *min_lift*.

    Parameters
    ----------
    itemsets:
        A :class:`~mba.mining.base.MiningResult` or its ``.itemsets`` frame.
    min_confidence, min_lift:
        Rule thresholds.
    metrics:
        Subset of :data:`ALL_METRICS` to keep. ``None`` keeps all of them.
    max_antecedent_len:
        Cap on ``|A|``; a 1-item antecedent is the most actionable form for
        merchandising ("customer put X in the basket, recommend Y").
    n_transactions:
        Needed only when *itemsets* is a bare frame without ``count``.
    sort_by:
        Column to sort descending by.
    """
    frame = itemsets.itemsets if isinstance(itemsets, MiningResult) else itemsets
    if n_transactions is None and isinstance(itemsets, MiningResult):
        n_transactions = itemsets.stats.n_transactions

    if frame is None or frame.empty:
        return empty_rules_frame()

    required = {"support", "itemsets"}
    if not required.issubset(frame.columns):
        raise KeyError(f"itemsets frame must contain {sorted(required)}")

    # support lookup for every mined itemset (antecedents and consequents of
    # a frequent set are themselves frequent, so they are all present).
    support_of: dict[frozenset, float] = dict(zip(frame["itemsets"], frame["support"]))

    rows_a: list[frozenset] = []
    rows_c: list[frozenset] = []
    s_ac: list[float] = []
    s_a: list[float] = []
    s_c: list[float] = []

    for itemset, support in zip(frame["itemsets"], frame["support"]):
        k = len(itemset)
        if k < 2:
            continue
        ordered = tuple(sorted(itemset))
        max_a = k - 1 if max_antecedent_len is None else min(max_antecedent_len, k - 1)
        for r in range(1, max_a + 1):
            for antecedent in combinations(ordered, r):
                a = frozenset(antecedent)
                c = frozenset(itemset - a)
                sa = support_of.get(a)
                sc = support_of.get(c)
                if sa is None or sc is None:
                    # Possible only when max_len truncated the lattice.
                    continue
                rows_a.append(a)
                rows_c.append(c)
                s_ac.append(support)
                s_a.append(sa)
                s_c.append(sc)

    if not rows_a:
        return empty_rules_frame()

    computed = rule_metrics(np.array(s_ac), np.array(s_a), np.array(s_c))

    rules = pd.DataFrame(
        {
            "antecedents": rows_a,
            "consequents": rows_c,
            "antecedent_support": s_a,
            "consequent_support": s_c,
            **computed,
        }
    )

    keep = (rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift)
    rules = rules.loc[keep].copy()
    if rules.empty:
        return empty_rules_frame()

    if n_transactions:
        rules["count"] = (rules["support"] * n_transactions).round().astype("int64")
    else:
        rules["count"] = -1

    rules["antecedent_len"] = rules["antecedents"].map(len).astype("int64")
    rules["consequent_len"] = rules["consequents"].map(len).astype("int64")

    if metrics is not None:
        wanted = set(metrics)
        drop = [m for m in ALL_METRICS if m not in wanted and m != "support"]
        rules = rules.drop(columns=[c for c in drop if c in rules.columns])

    if sort_by in rules.columns:
        rules = rules.sort_values(sort_by, ascending=False, kind="mergesort")

    rules = rules.reset_index(drop=True)
    log.info(
        "generated %s rules (min_conf=%.2f, min_lift=%.2f) from %s itemsets",
        f"{len(rules):,}",
        min_confidence,
        min_lift,
        f"{len(frame):,}",
    )
    return rules


def format_rules(rules: pd.DataFrame, max_items: int = 4) -> pd.DataFrame:
    """Render frozenset columns as readable strings for CSV/report output."""
    if rules.empty:
        return rules.copy()

    def _fmt(items: Iterable[str]) -> str:
        ordered = sorted(items)
        if len(ordered) > max_items:
            return ", ".join(ordered[:max_items]) + f" (+{len(ordered) - max_items})"
        return ", ".join(ordered)

    out = rules.copy()
    out["antecedents"] = out["antecedents"].map(_fmt)
    out["consequents"] = out["consequents"].map(_fmt)
    out.insert(0, "rule", out["antecedents"] + "  =>  " + out["consequents"])
    return out


def deduplicate_symmetric(rules: pd.DataFrame) -> pd.DataFrame:
    """Keep one direction of each ``A => C`` / ``C => A`` pair (highest confidence).

    Support, lift and leverage are symmetric, so both directions of a
    two-item rule carry the same evidence. For reporting, the higher-
    confidence direction is the actionable one.
    """
    if rules.empty:
        return rules.copy()
    key = rules.apply(
        lambda r: frozenset(r["antecedents"] | r["consequents"]), axis=1
    )
    out = rules.assign(_key=key).sort_values("confidence", ascending=False)
    return out.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
