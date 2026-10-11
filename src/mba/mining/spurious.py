"""Triage of high-value versus misleading association rules.

Thresholding on support/confidence/lift alone reliably surfaces rules that
are statistically true and commercially useless. This module flags the
recurring failure modes so a reader can tell the two apart:

``rare_pair``
    The rule rests on a handful of baskets. Lift is a ratio of small numbers
    and swings wildly; one wholesale order can create it.

``ubiquitous_consequent``
    The consequent is in most baskets anyway, so predicting it is nearly
    free. Confidence looks excellent while the rule adds almost nothing over
    the base rate.

``negligible_leverage``
    Lift is high but the *absolute* excess co-occurrence is tiny. Acting on
    it moves no revenue.

``weak_association``
    Zhang's metric is near zero: once the skew in the marginals is accounted
    for, the items are close to independent.

``not_significant``
    A chi-squared test of independence on the 2x2 contingency table fails to
    reject independence — the pattern is compatible with chance. The test is
    **Benjamini-Hochberg corrected across every rule in the table**, because
    testing hundreds of rules at p < 0.05 would otherwise produce a
    predictable crop of false positives.

``redundant``
    A more general rule (subset antecedent, same consequent) already achieves
    the same or better confidence, so the extra conditions earn nothing.

``self_referential``
    Antecedent and consequent are near-duplicate product variants (``RED
    SPOTTY MUG`` => ``BLUE SPOTTY MUG``). Statistically strong, but it
    describes the product catalogue rather than customer behaviour.

A rule flagged by nothing is reported as **high value**.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mba.config import Config

log = logging.getLogger(__name__)

#: Colour/size/variant words stripped when testing for product variants.
_VARIANT_TOKENS = {
    "RED", "BLUE", "GREEN", "PINK", "WHITE", "BLACK", "IVORY", "CREAM",
    "PURPLE", "ORANGE", "YELLOW", "SILVER", "GOLD", "BROWN", "GREY", "GRAY",
    "SMALL", "LARGE", "MEDIUM", "MINI", "JUMBO", "SET", "OF", "3", "6", "12",
    "DESIGN", "VINTAGE", "RETROSPOT", "ASSORTED", "COLOUR", "COLOR",
}
_TOKEN_RE = re.compile(r"[^A-Z0-9]+")

__all__ = [
    "FLAG_COLUMNS",
    "SpuriousFlags",
    "benjamini_hochberg",
    "chi2_independence",
    "flag_spurious_rules",
    "high_value_rules",
]

FLAG_COLUMNS = [
    "rare_pair",
    "ubiquitous_consequent",
    "negligible_leverage",
    "weak_association",
    "not_significant",
    "redundant",
    "self_referential",
]


@dataclass
class SpuriousFlags:
    """Summary of a triage pass."""

    n_rules: int
    n_flagged: int
    n_high_value: int
    counts: dict[str, int]

    @property
    def flagged_rate(self) -> float:
        return self.n_flagged / self.n_rules if self.n_rules else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rules": self.n_rules,
            "n_flagged": self.n_flagged,
            "n_high_value": self.n_high_value,
            "flagged_rate": round(self.flagged_rate, 4),
            "by_reason": dict(self.counts),
        }

    def summary(self) -> str:
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()) if v)
        return (
            f"{self.n_high_value:,}/{self.n_rules:,} rules are high value; "
            f"{self.n_flagged:,} flagged ({self.flagged_rate:.1%})"
            + (f" [{reasons}]" if reasons else "")
        )


def chi2_independence(
    support_ac: np.ndarray,
    support_a: np.ndarray,
    support_c: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Chi-squared statistic and p-value for each rule's 2x2 table.

    The table counts baskets by presence/absence of the antecedent and the
    consequent. With one degree of freedom the survival function has the
    closed form ``p = erfc(sqrt(chi2 / 2))``, so no per-row scipy call is
    needed and the whole column is computed at once.
    """
    from scipy.special import erfc

    n = float(n)
    n11 = support_ac * n
    n10 = (support_a - support_ac) * n
    n01 = (support_c - support_ac) * n
    n00 = n - n11 - n10 - n01

    row1, row0 = n11 + n10, n01 + n00
    col1, col0 = n11 + n01, n10 + n00

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = (n11 * n00 - n10 * n01) ** 2 * n
        denominator = row1 * row0 * col1 * col0
        chi2 = np.where(denominator > 0, numerator / denominator, 0.0)
    chi2 = np.nan_to_num(chi2, nan=0.0, posinf=0.0)
    p_value = erfc(np.sqrt(np.maximum(chi2, 0.0) / 2.0))
    return chi2, p_value


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), controlling the FDR.

    Rule mining is mass hypothesis testing. Testing 500 rules at p < 0.05
    means ~25 of them clear the bar by chance alone, so an uncorrected
    per-rule test does not mean what it appears to. Bonferroni would fix
    that but is far too conservative here — the tests are positively
    dependent (rules share items) and it would discard most true findings.

    Benjamini-Hochberg instead controls the *false discovery rate*: the
    expected share of false positives **among the rules declared
    significant**. For p-values sorted ascending with ranks i = 1..m::

        q(i) = min over k >= i of  ( p(k) * m / k )

    The running minimum from the largest rank downward enforces
    monotonicity, so a rule is never reported as more significant than a
    worse-ranked one. Returned q-values are clipped to 1.
    """
    p = np.asarray(p_values, dtype="float64")
    m = p.size
    if m == 0:
        return p.copy()

    order = np.argsort(p)
    ranked = p[order]
    # p * m / i, then the running minimum from the tail back to the head.
    scaled = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]

    q = np.empty_like(q_sorted)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def _normalised_tokens(items: frozenset) -> frozenset:
    """Content words of a product name, with colour/size variants removed."""
    tokens: set[str] = set()
    for item in items:
        for tok in _TOKEN_RE.split(str(item).upper()):
            if tok and tok not in _VARIANT_TOKENS and len(tok) > 2:
                tokens.add(tok)
    return frozenset(tokens)


def _variant_overlap(antecedents: pd.Series, consequents: pd.Series) -> np.ndarray:
    """Jaccard overlap of the de-variant-ed token sets, per rule."""
    scores = np.zeros(len(antecedents), dtype="float64")
    for i, (a, c) in enumerate(zip(antecedents, consequents)):
        ta, tc = _normalised_tokens(a), _normalised_tokens(c)
        union = ta | tc
        scores[i] = len(ta & tc) / len(union) if union else 0.0
    return scores


def _redundancy_mask(rules: pd.DataFrame) -> np.ndarray:
    """Flag rules whose antecedent is a superset of a simpler, equally good rule.

    ``{A, B} => C`` is redundant when ``{A} => C`` exists with confidence at
    least as high: the extra condition narrows the audience for no gain.
    """
    mask = np.zeros(len(rules), dtype=bool)
    by_consequent: dict[frozenset, list[tuple[frozenset, float]]] = {}
    for ante, cons, conf in zip(rules["antecedents"], rules["consequents"], rules["confidence"]):
        by_consequent.setdefault(cons, []).append((ante, conf))

    for i, (ante, cons, conf) in enumerate(
        zip(rules["antecedents"], rules["consequents"], rules["confidence"])
    ):
        if len(ante) < 2:
            continue
        for other_ante, other_conf in by_consequent.get(cons, ()):
            if other_ante < ante and other_conf >= conf - 1e-12:
                mask[i] = True
                break
    return mask


def flag_spurious_rules(
    rules: pd.DataFrame,
    cfg: Config | None = None,
    n_transactions: int | None = None,
    **overrides: Any,
) -> tuple[pd.DataFrame, SpuriousFlags]:
    """Annotate *rules* with quality flags and a ``quality_tier`` column.

    Returns the annotated frame (a copy) and a :class:`SpuriousFlags`
    summary. Added columns: the seven booleans in :data:`FLAG_COLUMNS`, plus
    ``chi2``, ``chi2_p_value``, ``variant_overlap``, ``n_flags``,
    ``is_spurious``, ``quality_tier`` and ``flag_reasons``.
    """
    params: dict[str, Any] = {
        "min_antecedent_support": 0.01,
        "max_consequent_support": 0.60,
        "min_leverage": 0.0005,
        "min_zhangs_metric": 0.10,
        "min_chi2_p_value": 0.05,
        "min_rule_count": 20,
        "max_variant_overlap": 0.60,
    }
    if cfg is not None:
        section = cfg.get("spurious", {})
        params.update(dict(section) if section else {})
    params.update({k: v for k, v in overrides.items() if v is not None})

    out = rules.copy()
    if out.empty:
        for col in FLAG_COLUMNS:
            out[col] = pd.Series(dtype="bool")
        out["n_flags"] = pd.Series(dtype="int64")
        out["is_spurious"] = pd.Series(dtype="bool")
        out["quality_tier"] = pd.Series(dtype="object")
        out["flag_reasons"] = pd.Series(dtype="object")
        return out, SpuriousFlags(0, 0, 0, {c: 0 for c in FLAG_COLUMNS})

    if n_transactions is None:
        counts = out.get("count")
        n_transactions = (
            int(round(counts.iloc[0] / out["support"].iloc[0]))
            if counts is not None and out["support"].iloc[0] > 0
            else 0
        )

    count = out["count"] if "count" in out.columns else (out["support"] * n_transactions)

    # -- statistical significance ---------------------------------------
    chi2, p_value = chi2_independence(
        out["support"].to_numpy(),
        out["antecedent_support"].to_numpy(),
        out["consequent_support"].to_numpy(),
        n_transactions or 1,
    )
    out["chi2"] = chi2
    out["chi2_p_value"] = p_value
    # Adjust for the fact that every rule in this table was tested.
    out["chi2_q_value"] = benjamini_hochberg(p_value)
    out["variant_overlap"] = _variant_overlap(out["antecedents"], out["consequents"])

    # -- individual flags ------------------------------------------------
    out["rare_pair"] = (count < params["min_rule_count"]) | (
        out["antecedent_support"] < params["min_antecedent_support"]
    )
    out["ubiquitous_consequent"] = out["consequent_support"] > params["max_consequent_support"]
    out["negligible_leverage"] = (
        out["leverage"].abs() < params["min_leverage"]
        if "leverage" in out.columns
        else False
    )
    out["weak_association"] = (
        out["zhangs_metric"] < params["min_zhangs_metric"]
        if "zhangs_metric" in out.columns
        else False
    )
    # Flag on the FDR-adjusted q-value, not the raw p-value: with hundreds
    # of rules tested at once, the raw p-value overstates significance.
    out["not_significant"] = out["chi2_q_value"] > params["min_chi2_p_value"]
    out["redundant"] = _redundancy_mask(out)
    out["self_referential"] = out["variant_overlap"] >= params["max_variant_overlap"]

    flags = out[FLAG_COLUMNS].astype(bool)
    out[FLAG_COLUMNS] = flags
    out["n_flags"] = flags.sum(axis=1).astype("int64")
    out["is_spurious"] = out["n_flags"] > 0
    out["flag_reasons"] = [
        ", ".join(c for c in FLAG_COLUMNS if row[c]) or "none"
        for _, row in flags.iterrows()
    ]
    out["quality_tier"] = np.select(
        [out["n_flags"] == 0, out["n_flags"] == 1],
        ["high_value", "review"],
        default="spurious",
    )

    summary = SpuriousFlags(
        n_rules=len(out),
        n_flagged=int(out["is_spurious"].sum()),
        n_high_value=int((~out["is_spurious"]).sum()),
        counts={c: int(flags[c].sum()) for c in FLAG_COLUMNS},
    )
    log.info("%s", summary.summary())
    return out, summary


def high_value_rules(
    rules: pd.DataFrame,
    cfg: Config | None = None,
    top_n: int | None = None,
    sort_by: str = "leverage",
) -> pd.DataFrame:
    """Shortcut: triage, keep the unflagged rules, sort and truncate."""
    annotated, _ = flag_spurious_rules(rules, cfg)
    keep = annotated.loc[~annotated["is_spurious"]]
    if sort_by in keep.columns:
        keep = keep.sort_values(sort_by, ascending=False)
    return keep.head(top_n) if top_n else keep
