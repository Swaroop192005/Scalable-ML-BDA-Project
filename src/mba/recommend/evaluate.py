"""Held-out evaluation of recommenders.

Mining rules proves a pattern existed in the data. It does not prove the
pattern *predicts* anything. This module closes that gap with the standard
protocol: train on the past, predict the future, score against what actually
happened.

The protocol
------------
**Temporal split, not random.** Transactions are split by date: everything
before the cut trains, everything after tests. A random split would let the
model learn from December while being tested on June, which is leakage — in
production you only ever have the past. Temporal splitting is also harsher,
because tastes and stock drift, and that is the point.

**Leave-one-out within each test basket.** For a test basket of *n* items,
one item is held out and the other *n−1* are shown to the model. A hit is
the held-out item appearing in the top-*k*. Baskets with fewer than two
items are skipped; they carry no signal.

**Rules are mined on the training split only.** `AssociationRecommender`
re-mines inside `fit()`, so test-period co-occurrences never reach the
model. Passing a pre-mined rules table that was mined on everything would
silently inflate every number here.

The metrics
-----------
``precision@k`` / ``recall@k``
    With exactly one held-out item, recall@k equals hit-rate, and
    precision@k is hit-rate / k. Both are reported because the relationship
    is only that simple under leave-one-out, and it is worth seeing.
``hit_rate@k``
    Share of test baskets where the held-out item was recovered. The most
    directly interpretable figure.
``MAP@k``
    Mean average precision — rewards ranking the right item *high*, not just
    including it. Separates models that are right at rank 1 from ones that
    are right at rank 10.
``MRR``
    Mean reciprocal rank of the first hit.
``coverage``
    Fraction of the catalogue the model ever recommends. A model with high
    accuracy and 0.4% coverage has learned "recommend the best-sellers" and
    will never surface the long tail.
``novelty``
    Mean ``-log2(popularity)`` of recommended items. High accuracy with low
    novelty means the model is recommending what the customer would have
    bought anyway — accurate and commercially worthless.

Coverage and novelty are not decoration. A recommender is judged on accuracy
*and* on whether it does anything a bestseller list would not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from mba.recommend.engine import Recommender, build_recommender

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Scores for one model at one *k*."""

    model: str
    k: int
    n_test_baskets: int
    n_train_baskets: int
    precision: float = 0.0
    recall: float = 0.0
    hit_rate: float = 0.0
    map_score: float = 0.0
    mrr: float = 0.0
    coverage: float = 0.0
    novelty: float = 0.0
    runtime_sec: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "k": self.k,
            "n_train_baskets": self.n_train_baskets,
            "n_test_baskets": self.n_test_baskets,
            "precision@k": round(self.precision, 5),
            "recall@k": round(self.recall, 5),
            "hit_rate@k": round(self.hit_rate, 5),
            "MAP@k": round(self.map_score, 5),
            "MRR": round(self.mrr, 5),
            "coverage": round(self.coverage, 5),
            "novelty": round(self.novelty, 4),
            "runtime_sec": round(self.runtime_sec, 2),
            **self.extra,
        }

    def summary(self) -> str:
        return (
            f"{self.model:<14s} k={self.k}  "
            f"P@k={self.precision:.4f}  R@k={self.recall:.4f}  "
            f"hit={self.hit_rate:.4f}  MAP={self.map_score:.4f}  "
            f"cov={self.coverage:.1%}  nov={self.novelty:.2f}"
        )


def temporal_split(
    frame: pd.DataFrame,
    invoice_col: str,
    item_col: str,
    date_col: str,
    test_fraction: float = 0.25,
    cut_date: str | pd.Timestamp | None = None,
) -> tuple[list[list[str]], list[list[str]], dict[str, Any]]:
    """Split transactions into train/test by date.

    The cut is placed at the quantile of *invoice* dates (not row dates), so
    *test_fraction* refers to the share of baskets, not of line items — a
    distinction that matters because basket sizes vary by an order of
    magnitude.

    Returns ``(train_baskets, test_baskets, metadata)``.
    """
    if date_col not in frame.columns:
        raise KeyError(f"temporal split needs a {date_col!r} column")

    # One timestamp per invoice: its first line.
    invoice_dates = (
        frame.groupby(invoice_col, observed=True)[date_col].min().sort_values()
    )
    if invoice_dates.empty:
        return [], [], {"cut_date": None, "n_train": 0, "n_test": 0}

    if cut_date is not None:
        cut = pd.Timestamp(cut_date)
    else:
        cut = invoice_dates.quantile(1.0 - test_fraction)

    train_ids = set(invoice_dates.loc[invoice_dates < cut].index)
    test_ids = set(invoice_dates.loc[invoice_dates >= cut].index)

    grouped = frame.groupby(invoice_col, observed=True)[item_col]
    train: list[list[str]] = []
    test: list[list[str]] = []
    for invoice, items in grouped:
        basket = sorted({str(v) for v in items if pd.notna(v)})
        if invoice in train_ids:
            train.append(basket)
        elif invoice in test_ids:
            test.append(basket)

    meta = {
        "cut_date": str(pd.Timestamp(cut).date()),
        "train_start": str(invoice_dates.min().date()),
        "train_end": str(invoice_dates.loc[invoice_dates < cut].max().date())
        if train_ids else None,
        "test_end": str(invoice_dates.max().date()),
        "n_train": len(train),
        "n_test": len(test),
        "test_fraction_actual": len(test) / (len(train) + len(test))
        if (train or test) else 0.0,
    }
    log.info(
        "temporal split at %s: %s train / %s test baskets (%.1f%% test)",
        meta["cut_date"], f"{len(train):,}", f"{len(test):,}",
        100 * meta["test_fraction_actual"],
    )
    return train, test, meta


def _leave_one_out_cases(
    test_baskets: Sequence[Sequence[str]],
    known_items: set[str],
    seed: int = 42,
    max_cases: int | None = None,
) -> list[tuple[list[str], str]]:
    """Build ``(visible_items, held_out_item)`` cases from the test baskets.

    The held-out item must be one the model could possibly know about — an
    item that never appeared in training is unrecoverable by *any* model, so
    including it would just add a constant penalty that tells us nothing and
    depresses every score equally.
    """
    rng = np.random.default_rng(seed)
    cases: list[tuple[list[str], str]] = []

    for basket in test_baskets:
        items = sorted(set(basket))
        if len(items) < 2:
            continue
        recoverable = [i for i in items if i in known_items]
        if not recoverable:
            continue
        held_out = recoverable[int(rng.integers(len(recoverable)))]
        visible = [i for i in items if i != held_out]
        if visible:
            cases.append((visible, held_out))

    if max_cases is not None and len(cases) > max_cases:
        picked = rng.choice(len(cases), size=max_cases, replace=False)
        cases = [cases[i] for i in sorted(picked)]
    return cases


def evaluate_at_ks(
    model: Recommender,
    train: Sequence[Sequence[str]],
    test: Sequence[Sequence[str]],
    k_values: Sequence[int] = (1, 5, 10),
    seed: int = 42,
    max_cases: int | None = None,
    fit: bool = True,
) -> list[EvaluationResult]:
    """Score *model* at several *k* in a single pass over the test cases.

    Recommendations are generated once at ``max(k_values)`` and sliced. That
    is exact rather than an approximation: :meth:`Recommender.recommend`
    ranks deterministically and truncates, so the top-k list is always a
    prefix of the top-K list for K >= k. Evaluating three values of k this
    way costs one pass instead of three.
    """
    import time

    t0 = time.perf_counter()
    if fit:
        model.fit(list(train))

    k_values = sorted(set(int(k) for k in k_values))
    k_max = max(k_values)
    known_items = set(model.item_counts)
    cases = _leave_one_out_cases(test, known_items, seed=seed, max_cases=max_cases)

    if not cases:
        return [
            EvaluationResult(
                model=model.name, k=k, n_test_baskets=0, n_train_baskets=len(train)
            )
            for k in k_values
        ]

    n_train = max(len(train), 1)
    popularity = {item: count / n_train for item, count in model.item_counts.items()}

    hits = {k: 0 for k in k_values}
    average_precisions = {k: [] for k in k_values}
    reciprocal_ranks = {k: [] for k in k_values}
    recommended: dict[int, set[str]] = {k: set() for k in k_values}
    novelty: dict[int, list[float]] = {k: [] for k in k_values}

    for visible, held_out in cases:
        picks = model.recommend(visible, k=k_max)
        # Rank of the held-out item within the full top-K list, or None.
        rank = picks.index(held_out) + 1 if held_out in picks else None

        for k in k_values:
            top_k = picks[:k]
            recommended[k].update(top_k)
            for item in top_k:
                p = popularity.get(item, 1.0 / n_train)
                novelty[k].append(-np.log2(max(p, 1e-12)))

            if rank is not None and rank <= k:
                hits[k] += 1
                # One relevant item, so average precision is simply 1/rank.
                average_precisions[k].append(1.0 / rank)
                reciprocal_ranks[k].append(1.0 / rank)
            else:
                average_precisions[k].append(0.0)
                reciprocal_ranks[k].append(0.0)

    n_cases = len(cases)
    catalogue = max(len(model.item_counts), 1)
    elapsed = time.perf_counter() - t0

    results = []
    for k in k_values:
        hit_rate = hits[k] / n_cases
        results.append(
            EvaluationResult(
                model=model.name,
                k=k,
                n_test_baskets=n_cases,
                n_train_baskets=len(train),
                # Exactly one relevant item per case, so these collapse as
                # noted in the module docstring.
                precision=hit_rate / k,
                recall=hit_rate,
                hit_rate=hit_rate,
                map_score=float(np.mean(average_precisions[k])),
                mrr=float(np.mean(reciprocal_ranks[k])),
                coverage=len(recommended[k]) / catalogue,
                novelty=float(np.mean(novelty[k])) if novelty[k] else 0.0,
                runtime_sec=elapsed,
            )
        )
        log.info("%s", results[-1].summary())
    return results


def evaluate_recommender(
    model: Recommender,
    train: Sequence[Sequence[str]],
    test: Sequence[Sequence[str]],
    k: int = 5,
    seed: int = 42,
    max_cases: int | None = None,
    fit: bool = True,
) -> EvaluationResult:
    """Fit on *train*, score leave-one-out predictions over *test* at one *k*."""
    return evaluate_at_ks(
        model, train, test, k_values=(k,), seed=seed, max_cases=max_cases, fit=fit
    )[0]


def compare_recommenders(
    train: Sequence[Sequence[str]],
    test: Sequence[Sequence[str]],
    models: dict[str, Recommender] | None = None,
    k_values: Sequence[int] = (1, 5, 10),
    seed: int = 42,
    max_cases: int | None = None,
) -> pd.DataFrame:
    """Fit every model once and score it at each *k*.

    Each model is fitted a single time and reused across *k*, so the
    comparison is not polluted by refit variance and the runtime column
    stays meaningful.
    """
    if models is None:
        models = {
            "association": build_recommender("association"),
            "cooccurrence": build_recommender("cooccurrence"),
            "popularity": build_recommender("popularity"),
            "random": build_recommender("random", seed=seed),
        }

    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        log.info("fitting %s...", name)
        model.fit(list(train))
        for result in evaluate_at_ks(
            model, train, test, k_values=k_values, seed=seed,
            max_cases=max_cases, fit=False,
        ):
            record = result.to_dict()
            record["model"] = name
            if hasattr(model, "n_rules"):
                record["n_rules"] = model.n_rules
            rows.append(record)

    frame = pd.DataFrame(rows)
    return frame.sort_values(["k", "hit_rate@k"], ascending=[True, False]).reset_index(drop=True)


def lift_over_baseline(frame: pd.DataFrame, baseline: str = "popularity") -> pd.DataFrame:
    """Express each model's hit-rate as a multiple of a baseline's.

    The number that actually matters. An absolute hit-rate of 0.19 means
    nothing until you know popularity scores 0.09 on the same split.
    """
    if frame.empty or "hit_rate@k" not in frame.columns:
        return frame

    out = frame.copy()
    reference = (
        out.loc[out["model"] == baseline].set_index("k")["hit_rate@k"].to_dict()
    )
    out[f"vs_{baseline}"] = [
        (row["hit_rate@k"] / reference[row["k"]]) if reference.get(row["k"]) else np.nan
        for _, row in out.iterrows()
    ]
    return out
