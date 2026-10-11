"""RFM customer segmentation, and association rules mined per segment.

Pooling every customer into one model assumes they all shop the same way.
They do not. A wholesale buyer placing £15,000 orders monthly and a
one-time gift buyer are both "customers", and the rules that describe them
average into something that describes neither.

**RFM** is the standard retail framing:

``Recency``   days since the customer's last purchase (lower is better)
``Frequency`` number of distinct invoices
``Monetary``  total spend

Clustering on those three separates behaviour patterns without needing
labels. :func:`segment_customers` then mines association rules per segment,
and the differences are the point: what a champion buys alongside X is not
what a one-time buyer does.

On k-means
----------
Implemented here rather than imported. The project implements its
algorithms from scratch where they are part of what is being studied, and
adding scikit-learn — a large dependency — for one k-means would be out of
proportion. The implementation uses **k-means++** initialisation, which
picks seeds spread across the data rather than uniformly at random; plain
random seeding on skewed RFM data routinely collapses two centroids into
the same dense region and produces a degenerate clustering.

Features are **log-transformed then z-scored** before clustering. Frequency
and monetary are heavily right-skewed — a handful of wholesale accounts
spend four orders of magnitude more than the median — and on raw values
Euclidean distance is dominated entirely by those outliers, yielding one
cluster of three customers and one of everybody else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mba.config import Config

log = logging.getLogger(__name__)


# --------------------------------------------------------------- k-means
def kmeans_plusplus_init(
    points: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii, 2007).

    The first centre is uniform; each subsequent centre is drawn with
    probability proportional to its squared distance from the nearest centre
    already chosen. That spreads the seeds out and gives the well-known
    O(log k) approximation guarantee in expectation, where uniform seeding
    has none.
    """
    n = len(points)
    centres = np.empty((k, points.shape[1]), dtype=points.dtype)
    centres[0] = points[rng.integers(n)]

    closest_sq = ((points - centres[0]) ** 2).sum(axis=1)
    for index in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            # All remaining points coincide with a centre; fill arbitrarily.
            centres[index] = points[rng.integers(n)]
        else:
            probabilities = closest_sq / total
            centres[index] = points[rng.choice(n, p=probabilities)]
        new_sq = ((points - centres[index]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)

    return centres


def kmeans(
    points: np.ndarray,
    k: int = 4,
    max_iter: int = 300,
    tol: float = 1e-6,
    n_init: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's algorithm with k-means++ seeding, best of *n_init* restarts.

    Returns ``(labels, centroids, inertia)``, where inertia is the summed
    squared distance to the assigned centroid. Restarts matter: k-means
    converges to a local optimum, and the best of ten runs is meaningfully
    better than one.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype="float64")
    n = len(points)
    if n == 0:
        return np.array([], dtype=int), np.empty((0, points.shape[1])), 0.0
    k = min(k, n)

    best: tuple[np.ndarray, np.ndarray, float] | None = None

    for _ in range(n_init):
        centroids = kmeans_plusplus_init(points, k, rng)
        labels = np.zeros(n, dtype=int)

        for _iteration in range(max_iter):
            # Assign: squared distance to every centroid, then argmin.
            distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)

            # Update: mean of each cluster; re-seed any cluster that emptied.
            new_centroids = np.empty_like(centroids)
            for index in range(k):
                members = points[new_labels == index]
                new_centroids[index] = (
                    members.mean(axis=0) if len(members) else points[rng.integers(n)]
                )

            shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
            centroids, labels = new_centroids, new_labels
            if shift < tol:
                break

        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        inertia = float(distances[np.arange(n), labels].sum())
        if best is None or inertia < best[2]:
            best = (labels, centroids, inertia)

    assert best is not None
    return best


def silhouette_score(points: np.ndarray, labels: np.ndarray, sample: int = 2000,
                     seed: int = 42) -> float:
    """Mean silhouette coefficient, in [-1, 1]; higher is better separated.

    For each point: ``(b - a) / max(a, b)`` where *a* is the mean distance to
    its own cluster and *b* the mean distance to the nearest other cluster.
    Subsampled above *sample* points because the computation is O(n^2).
    """
    points = np.asarray(points, dtype="float64")
    labels = np.asarray(labels)
    n = len(points)
    if n < 2 or len(set(labels.tolist())) < 2:
        return 0.0

    if n > sample:
        rng = np.random.default_rng(seed)
        index = rng.choice(n, size=sample, replace=False)
        points, labels = points[index], labels[index]
        n = sample

    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    unique = np.unique(labels)
    scores = np.zeros(n)

    for i in range(n):
        own = labels == labels[i]
        own_count = own.sum() - 1
        if own_count <= 0:
            scores[i] = 0.0
            continue
        a = distances[i, own].sum() / own_count
        b = min(
            distances[i, labels == other].mean()
            for other in unique
            if other != labels[i]
        )
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(scores.mean())


# ------------------------------------------------------------------- RFM
@dataclass
class SegmentationResult:
    """Per-customer RFM table, segment profiles, and per-segment rules."""

    customers: pd.DataFrame
    profiles: pd.DataFrame
    rules: dict[str, pd.DataFrame] = field(default_factory=dict)
    inertia: float = 0.0
    silhouette: float = 0.0
    k: int = 0

    @property
    def n_customers(self) -> int:
        return len(self.customers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_customers": self.n_customers,
            "k": self.k,
            "inertia": round(self.inertia, 2),
            "silhouette": round(self.silhouette, 4),
            "segments": self.profiles["segment"].tolist() if not self.profiles.empty else [],
        }

    def summary(self) -> str:
        return (
            f"{self.n_customers:,} customers -> {self.k} segments "
            f"(silhouette {self.silhouette:.3f}): "
            + ", ".join(
                f"{r['segment']} n={int(r['n_customers']):,}"
                for _, r in self.profiles.iterrows()
            )
        )


def compute_rfm(
    frame: pd.DataFrame,
    cfg: Config,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-customer recency, frequency and monetary value.

    *reference_date* defaults to the day after the last transaction, so the
    most recent customer has recency 1 rather than 0 — which keeps the log
    transform defined.
    """
    schema = cfg["schema"]
    customer, invoice = schema["customer_col"], schema["invoice_col"]
    date, qty, price = schema["date_col"], schema["quantity_col"], schema["price_col"]

    if customer not in frame.columns:
        raise KeyError(f"RFM needs a {customer!r} column")

    known = frame.loc[frame[customer].notna()].copy()
    if known.empty:
        return pd.DataFrame(
            columns=["customer", "recency", "frequency", "monetary", "avg_basket_value"]
        )

    known["_revenue"] = known[qty] * known[price]
    if reference_date is None:
        reference_date = known[date].max() + pd.Timedelta(days=1)

    rfm = known.groupby(customer, observed=True).agg(
        last_purchase=(date, "max"),
        frequency=(invoice, "nunique"),
        monetary=("_revenue", "sum"),
        n_items=(schema["item_col"], "nunique"),
    )
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days.clip(lower=1)
    rfm["avg_basket_value"] = rfm["monetary"] / rfm["frequency"].clip(lower=1)
    rfm = rfm.reset_index().rename(columns={customer: "customer"})

    log.info(
        "RFM for %s customers (reference %s)",
        f"{len(rfm):,}", reference_date.date(),
    )
    return rfm[
        ["customer", "recency", "frequency", "monetary", "avg_basket_value",
         "n_items", "last_purchase"]
    ]


def _prepare_features(rfm: pd.DataFrame) -> np.ndarray:
    """Log-transform then z-score the three RFM dimensions.

    Monetary and frequency span four orders of magnitude; without the log,
    Euclidean distance is decided entirely by a handful of wholesale
    accounts and k-means returns one tiny cluster plus everyone else.
    """
    features = np.column_stack(
        [
            np.log1p(rfm["recency"].to_numpy(dtype="float64")),
            np.log1p(rfm["frequency"].to_numpy(dtype="float64")),
            np.log1p(rfm["monetary"].clip(lower=0).to_numpy(dtype="float64")),
        ]
    )
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - mean) / std


def _name_segments(profiles: pd.DataFrame) -> list[str]:
    """Give each cluster a business-readable name from its RFM profile.

    Names are assigned by rank within this run — relative to the other
    clusters found — not by absolute thresholds, so they stay meaningful on
    a different dataset or a different *k*.
    """
    ranked = profiles.copy()
    # Low recency = recent = good, so invert it.
    ranked["r_score"] = ranked["recency"].rank(ascending=False)
    ranked["f_score"] = ranked["frequency"].rank(ascending=True)
    ranked["m_score"] = ranked["monetary"].rank(ascending=True)
    ranked["value"] = ranked["f_score"] + ranked["m_score"]

    n = len(ranked)
    names: dict[int, str] = {}
    by_value = ranked.sort_values(["value", "r_score"], ascending=False)

    for position, (index, row) in enumerate(by_value.iterrows()):
        recent = row["r_score"] > n / 2
        if position == 0:
            names[index] = "champions" if recent else "big spenders at risk"
        elif position == n - 1:
            names[index] = "one-time buyers" if not recent else "new / low value"
        elif recent:
            names[index] = "loyal"
        else:
            names[index] = "at risk"

    # Guarantee uniqueness even when two clusters land on the same label.
    seen: dict[str, int] = {}
    out = []
    for index in profiles.index:
        base = names.get(index, "segment")
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base} {seen[base]}")
    return out


def segment_customers(
    frame: pd.DataFrame,
    cfg: Config,
    k: int | None = None,
    seed: int | None = None,
    mine_rules: bool = True,
    min_support: float | None = None,
    min_confidence: float | None = None,
    min_segment_size: int | None = None,
) -> SegmentationResult:
    """Cluster customers on RFM, then mine association rules per segment.

    Defaults come from the ``analytics.segmentation`` config section. The
    per-segment support threshold is lower than the global one by design:
    a segment holds a fraction of the baskets, so the global threshold
    would return almost nothing for the smaller ones.
    """
    settings = cfg.get("analytics.segmentation", {})
    k = k if k is not None else int(settings.get("k", 4))
    min_support = (
        min_support if min_support is not None else float(settings.get("min_support", 0.02))
    )
    min_confidence = (
        min_confidence if min_confidence is not None
        else float(settings.get("min_confidence", 0.3))
    )
    min_segment_size = (
        min_segment_size if min_segment_size is not None
        else int(settings.get("min_segment_size", 50))
    )
    from mba.data.basket import encode_baskets
    from mba.mining.base import mine
    from mba.mining.rules import association_rules
    from mba.mining.spurious import flag_spurious_rules

    seed = seed if seed is not None else int(cfg.get("project.random_seed", 42))
    schema = cfg["schema"]
    customer_col = schema["customer_col"]

    rfm = compute_rfm(frame, cfg)
    if rfm.empty:
        return SegmentationResult(rfm, pd.DataFrame())

    features = _prepare_features(rfm)
    labels, centroids, inertia = kmeans(features, k=k, seed=seed)
    rfm = rfm.assign(cluster=labels)

    profiles = (
        rfm.groupby("cluster", observed=True)
        .agg(
            n_customers=("customer", "size"),
            recency=("recency", "median"),
            frequency=("frequency", "median"),
            monetary=("monetary", "median"),
            avg_basket_value=("avg_basket_value", "median"),
            total_revenue=("monetary", "sum"),
        )
    )
    profiles["revenue_share"] = profiles["total_revenue"] / profiles["total_revenue"].sum()
    profiles["customer_share"] = profiles["n_customers"] / profiles["n_customers"].sum()
    profiles["segment"] = _name_segments(profiles)
    profiles = profiles.reset_index()

    naming = dict(zip(profiles["cluster"], profiles["segment"]))
    rfm["segment"] = rfm["cluster"].map(naming)

    silhouette = silhouette_score(features, labels, seed=seed)
    result = SegmentationResult(
        customers=rfm, profiles=profiles, inertia=inertia,
        silhouette=silhouette, k=int(len(profiles)),
    )

    if mine_rules:
        membership = dict(zip(rfm["customer"], rfm["segment"]))
        tagged = frame.loc[frame[customer_col].notna()].copy()
        tagged["_segment"] = tagged[customer_col].map(membership)

        for segment, subset in tagged.groupby("_segment", observed=True):
            basket = encode_baskets(subset, cfg, min_item_count=2)
            if basket.n_transactions < min_segment_size or basket.n_items == 0:
                log.info("segment %r too small to mine (%d baskets)",
                         segment, basket.n_transactions)
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
            if not rules.empty:
                rules, _ = flag_spurious_rules(rules, cfg, basket.n_transactions)
                rules.insert(0, "segment", segment)
            result.rules[str(segment)] = rules
            log.info("segment %-22s %5d baskets -> %4d rules",
                     segment, basket.n_transactions, len(rules))

    log.info("%s", result.summary())
    return result


def choose_k(
    frame: pd.DataFrame,
    cfg: Config,
    k_values: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    seed: int = 42,
) -> pd.DataFrame:
    """Inertia and silhouette across *k*, for the elbow / silhouette choice.

    Inertia always falls as *k* rises, so it alone cannot pick *k* — the
    silhouette column is what to read.
    """
    rfm = compute_rfm(frame, cfg)
    if rfm.empty:
        return pd.DataFrame()
    features = _prepare_features(rfm)

    rows = []
    for k in k_values:
        labels, _, inertia = kmeans(features, k=k, seed=seed)
        rows.append(
            {
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette": round(silhouette_score(features, labels, seed=seed), 4),
                "smallest_cluster": int(np.bincount(labels).min()),
            }
        )
        log.info("k=%d inertia=%.1f silhouette=%.4f", k, inertia, rows[-1]["silhouette"])
    return pd.DataFrame(rows)


def compare_segment_rules(result: SegmentationResult, top_n: int = 5) -> pd.DataFrame:
    """Strongest rules per segment, side by side."""
    rows = []
    for segment, rules in result.rules.items():
        if rules is None or rules.empty:
            continue
        for _, row in rules.nlargest(top_n, "lift").iterrows():
            rows.append(
                {
                    "segment": segment,
                    "antecedents": ", ".join(sorted(row["antecedents"])),
                    "consequents": ", ".join(sorted(row["consequents"])),
                    "support": round(float(row["support"]), 4),
                    "confidence": round(float(row["confidence"]), 4),
                    "lift": round(float(row["lift"]), 3),
                }
            )
    return pd.DataFrame(rows)
