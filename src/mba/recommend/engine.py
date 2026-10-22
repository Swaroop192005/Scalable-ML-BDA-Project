"""Recommenders built from mined rules, plus the baselines that keep them honest.

A rules table is not a recommender. Turning one into a recommender means
deciding how to combine the rules that fire for a given basket, and that
choice matters more than the mining did.

Every model here implements the same interface::

    model.fit(transactions)
    model.recommend(basket, k=5) -> [item, ...]

so the evaluation harness can score any of them identically.

The baselines exist because a recommender is only meaningful relative to
what it beats:

``PopularityRecommender``
    Always recommends the globally most popular items not already in the
    basket. Embarrassingly hard to beat on hit-rate, and the reason a raw
    accuracy number means nothing on its own.
``CooccurrenceRecommender``
    Raw co-occurrence counts — no support threshold, no lift, no rule
    mining. This isolates what the *association rule machinery* adds over
    simply counting pairs.
``RandomRecommender``
    The floor.

If the association model cannot beat co-occurrence, the mining added
nothing but cost.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class Recommender(ABC):
    """Common interface for every model the evaluator can score."""

    name: str = "recommender"

    def __init__(self) -> None:
        self.item_counts: Counter = Counter()
        self.n_transactions: int = 0
        self._fitted = False

    @abstractmethod
    def _fit(self, transactions: Sequence[Sequence[str]]) -> None:
        """Model-specific training."""

    @abstractmethod
    def _score(self, basket: set[str]) -> dict[str, float]:
        """Return ``{candidate_item: score}`` for items not already in *basket*."""

    def fit(self, transactions: Sequence[Sequence[str]]) -> "Recommender":
        """Train on a list of baskets."""
        self.n_transactions = len(transactions)
        self.item_counts = Counter()
        for transaction in transactions:
            self.item_counts.update(set(transaction))
        self._fit(transactions)
        self._fitted = True
        log.info(
            "%s fitted on %s transactions, %s distinct items",
            self.name, f"{self.n_transactions:,}", f"{len(self.item_counts):,}",
        )
        return self

    def recommend(self, basket: Iterable[str], k: int = 5) -> list[str]:
        """Top-*k* items to recommend given the items already in *basket*.

        Ties are broken by global popularity, then alphabetically — a fixed
        order, so results are reproducible and two models are never
        accidentally compared on tie-break luck.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fitted before recommending")

        basket_set = set(basket)
        scores = self._score(basket_set)
        if not scores:
            return self._popularity_fallback(basket_set, k)

        ranked = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], -self.item_counts.get(kv[0], 0), kv[0]),
        )
        picks = [item for item, _ in ranked[:k]]

        # Top up from popularity when the model has too few candidates, so
        # every model is scored on a full k-length list.
        if len(picks) < k:
            picks += [
                item
                for item in self._popularity_fallback(basket_set | set(picks), k - len(picks))
            ]
        return picks[:k]

    def _popularity_fallback(self, exclude: set[str], k: int) -> list[str]:
        return [
            item
            for item, _ in self.item_counts.most_common()
            if item not in exclude
        ][:k]

    def recommend_batch(
        self, baskets: Sequence[Iterable[str]], k: int = 5
    ) -> list[list[str]]:
        return [self.recommend(b, k) for b in baskets]

    def explain(self, basket: Iterable[str], k: int = 5) -> pd.DataFrame:
        """Recommendations with their scores, for inspection."""
        basket_set = set(basket)
        scores = self._score(basket_set)
        picks = self.recommend(basket_set, k)
        return pd.DataFrame(
            [
                {
                    "rank": i + 1,
                    "item": item,
                    "score": scores.get(item, 0.0),
                    "popularity": self.item_counts.get(item, 0) / max(self.n_transactions, 1),
                    "from_fallback": item not in scores,
                }
                for i, item in enumerate(picks)
            ]
        )


class RandomRecommender(Recommender):
    """Uniformly random items. The floor any model must clear."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._items: list[str] = []

    def _fit(self, transactions: Sequence[Sequence[str]]) -> None:
        self._items = sorted(self.item_counts)
        self._rng = np.random.default_rng(self.seed)

    def _score(self, basket: set[str]) -> dict[str, float]:
        candidates = [i for i in self._items if i not in basket]
        if not candidates:
            return {}
        return {item: float(self._rng.random()) for item in candidates}

    def _popularity_fallback(self, exclude: set[str], k: int) -> list[str]:
        # A random model must not quietly fall back to popularity — that
        # would make the floor look much better than it is.
        candidates = [i for i in self._items if i not in exclude]
        if not candidates:
            return []
        size = min(k, len(candidates))
        return list(self._rng.choice(candidates, size=size, replace=False))


class PopularityRecommender(Recommender):
    """Always the globally most popular items the basket does not have.

    Ignores basket content entirely beyond exclusion. This is the baseline
    that embarrasses most recommenders, and the reason a bare precision
    figure says nothing.
    """

    name = "popularity"

    def _fit(self, transactions: Sequence[Sequence[str]]) -> None:
        pass  # global counts, computed in fit()

    def _score(self, basket: set[str]) -> dict[str, float]:
        return {
            item: count / max(self.n_transactions, 1)
            for item, count in self.item_counts.items()
            if item not in basket
        }


class CooccurrenceRecommender(Recommender):
    """Raw item-to-item co-occurrence, with no rule mining at all.

    Scores a candidate by how often it appears with the basket's items.
    ``normalise="lift"`` divides by the candidate's own popularity, which
    stops ubiquitous items winning every slot; ``"count"`` leaves it raw.

    This is the model the association recommender has to beat to justify
    the mining pipeline's existence.
    """

    name = "cooccurrence"

    def __init__(self, normalise: str = "lift", min_cooccurrence: int = 2) -> None:
        super().__init__()
        self.normalise = normalise
        self.min_cooccurrence = min_cooccurrence
        self.pair_counts: dict[str, Counter] = defaultdict(Counter)

    def _fit(self, transactions: Sequence[Sequence[str]]) -> None:
        self.pair_counts = defaultdict(Counter)
        for transaction in transactions:
            items = sorted(set(transaction))
            for i, a in enumerate(items):
                for b in items[i + 1 :]:
                    self.pair_counts[a][b] += 1
                    self.pair_counts[b][a] += 1

    def _score(self, basket: set[str]) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        n = max(self.n_transactions, 1)
        for item in basket:
            for candidate, pair_count in self.pair_counts.get(item, {}).items():
                if candidate in basket or pair_count < self.min_cooccurrence:
                    continue
                if self.normalise == "lift":
                    s_a = self.item_counts[item] / n
                    s_c = self.item_counts[candidate] / n
                    scores[candidate] += (pair_count / n) / (s_a * s_c) if s_a and s_c else 0.0
                else:
                    scores[candidate] += pair_count
        return dict(scores)


class AssociationRecommender(Recommender):
    """Recommender driven by mined association rules.

    For a basket *B*, every rule whose antecedent is a subset of *B*
    "fires", and each fired rule votes for its consequent items. The
    aggregation strategy decides how votes combine:

    ``weighted`` (default)
        Sum of ``confidence * log(lift)`` over fired rules. Confidence is the
        probability the rule asserts; ``log(lift)`` discounts rules that are
        confident only because the consequent is popular anyway. Summing
        rewards a consequent that several independent rules agree on.
    ``max_confidence``
        The single most confident fired rule wins. Conservative — one strong
        rule beats many weak ones.
    ``confidence``
        Plain sum of confidences. Ignores lift, so popular consequents
        dominate; included to show what lift is buying.

    Longer antecedents are weighted up by ``specificity_weight``: a rule
    matching three of the basket's items is more targeted evidence than one
    matching a single item.
    """

    name = "association"

    def __init__(
        self,
        rules: pd.DataFrame | None = None,
        strategy: str = "weighted",
        specificity_weight: float = 0.5,
        min_confidence: float = 0.0,
        min_lift: float = 1.0,
        exclude_spurious: bool = True,
        mine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.specificity_weight = specificity_weight
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.exclude_spurious = exclude_spurious
        self.mine_kwargs = mine_kwargs or {}
        self.rules = rules
        #: item -> list of (antecedent, consequents, confidence, lift)
        self._index: dict[str, list[tuple[frozenset, frozenset, float, float]]] = defaultdict(list)
        self.n_rules = 0

    def _mine_rules(self, transactions: Sequence[Sequence[str]]) -> pd.DataFrame:
        """Mine rules from the training transactions.

        Critically, this runs on the **training split only**. Mining on the
        whole dataset and then evaluating on part of it would leak the test
        period's co-occurrences into the model and inflate every metric.
        """
        from mba.data.basket import baskets_from_transactions
        from mba.mining.base import mine
        from mba.mining.rules import association_rules

        params = {
            "algorithm": "fpgrowth",
            "min_support": 0.005,
            "max_len": 3,
            **self.mine_kwargs,
        }
        basket = baskets_from_transactions(transactions)
        itemsets = mine(
            basket,
            algorithm=params["algorithm"],
            min_support=params["min_support"],
            max_len=params["max_len"],
            track_memory=False,
        )
        return association_rules(
            itemsets,
            min_confidence=self.min_confidence,
            min_lift=self.min_lift,
            sort_by="lift",
        )

    def _fit(self, transactions: Sequence[Sequence[str]]) -> None:
        rules = self.rules if self.rules is not None else self._mine_rules(transactions)

        if rules is None or rules.empty:
            log.warning("no rules available; %s will fall back to popularity", self.name)
            self._index = defaultdict(list)
            self.n_rules = 0
            return

        if self.exclude_spurious and "is_spurious" in rules.columns:
            rules = rules.loc[~rules["is_spurious"]]

        # Index by every antecedent item so lookup is proportional to basket
        # size, not to the size of the rules table.
        self._index = defaultdict(list)
        for row in rules.itertuples():
            entry = (
                frozenset(row.antecedents),
                frozenset(row.consequents),
                float(row.confidence),
                float(row.lift),
            )
            for item in entry[0]:
                self._index[item].append(entry)

        self.n_rules = len(rules)
        log.info("%s indexed %s rules over %s items",
                 self.name, f"{self.n_rules:,}", f"{len(self._index):,}")

    def top_rule_for(
        self, basket: Iterable[str], item: str
    ) -> tuple[frozenset[str], float, float] | None:
        """The strongest rule that recommends *item* for *basket*.

        :meth:`recommend` aggregates every rule that fires, which is the right
        way to rank but leaves the result unexplainable. This returns the
        single highest-lift contributing rule as
        ``(antecedent, confidence, lift)``, so a caller can say *why* an item
        was suggested. ``None`` when no rule fired — the caller is then looking
        at a popularity fallback rather than evidence.
        """
        basket_set = frozenset(basket)
        best: tuple[frozenset[str], float, float] | None = None
        best_contribution = -1.0
        seen: set[tuple[frozenset, frozenset]] = set()

        for member in basket_set:
            for antecedent, consequents, confidence, lift in self._index.get(member, ()):
                if item not in consequents or not antecedent <= basket_set:
                    continue
                key = (antecedent, consequents)
                if key in seen:
                    continue
                seen.add(key)

                # Rank by what the rule actually contributed to the score, not
                # by lift alone. A rare single-item rule can out-lift the
                # specific multi-item rule that did the real work, and
                # explaining the wrong one is worse than not explaining.
                contribution = self._contribution(confidence, lift, len(antecedent))
                if contribution > best_contribution:
                    best_contribution = contribution
                    best = (antecedent, confidence, lift)
        return best

    def _contribution(self, confidence: float, lift: float, antecedent_len: int) -> float:
        """One rule's share of a consequent's score, per the active strategy."""
        if self.strategy == "max_confidence":
            return confidence
        weight = antecedent_len ** self.specificity_weight
        if self.strategy == "confidence":
            return confidence * weight
        if self.strategy != "weighted":
            raise ValueError(f"unknown strategy: {self.strategy!r}")
        return confidence * float(np.log(max(lift, 1.0 + 1e-9))) * weight


    def _score(self, basket: set[str]) -> dict[str, float]:
        if not self._index:
            return {}

        scores: dict[str, float] = defaultdict(float)
        seen: set[tuple[frozenset, frozenset]] = set()

        for item in basket:
            for antecedent, consequents, confidence, lift in self._index.get(item, ()):
                if not antecedent <= basket:
                    continue  # rule does not fire for this basket
                key = (antecedent, consequents)
                if key in seen:
                    continue  # a multi-item antecedent is reached once per item
                seen.add(key)

                # Longer antecedents are more specific evidence; the weighting
                # lives in _contribution so scoring and explanation cannot drift.
                value = self._contribution(confidence, lift, len(antecedent))

                for consequent in consequents:
                    if consequent in basket:
                        continue
                    if self.strategy == "max_confidence":
                        scores[consequent] = max(scores[consequent], value)
                    else:
                        scores[consequent] += value

        return dict(scores)


#: Name -> constructor, for the CLI and the evaluation harness.
REGISTRY: dict[str, type[Recommender]] = {
    "association": AssociationRecommender,
    "cooccurrence": CooccurrenceRecommender,
    "popularity": PopularityRecommender,
    "random": RandomRecommender,
}


def build_recommender(name: str, **kwargs) -> Recommender:
    if name not in REGISTRY:
        raise ValueError(f"unknown recommender {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
