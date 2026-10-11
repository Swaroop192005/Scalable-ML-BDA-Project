"""Lossy Counting: frequent items over an unbounded stream.

Every miner elsewhere in this project assumes the dataset is *finite and
re-readable* — the in-memory path holds it all, SON streams it twice. Neither
works for a genuinely unbounded stream: a live checkout feed has no end and
no second pass, and an exact count needs one counter per distinct item,
which is unbounded memory.

Lossy Counting (Manku & Motwani, VLDB 2002) trades a *bounded, known* amount
of accuracy for bounded memory in a single pass.

The algorithm
-------------
Fix an error tolerance ``epsilon``. The stream is cut into buckets of
``w = ceil(1/epsilon)`` *observations* -- one observation is one transaction
in the itemset setting, or one element for a plain item stream. Bucket *b* is
the one currently filling.

For each item in an arriving observation:

* if it has an entry, increment its count;
* otherwise insert it with count 1 and **delta = b − 1**, where delta is the
  maximum number of occurrences it could already have had before it was
  first tracked.

At each bucket boundary, delete every entry with ``count + delta <= b``.
Anything that infrequent cannot be a false negative later.

The guarantees
--------------
With *N* observations seen, for a support threshold *s* and error *epsilon < s*,
reporting entries with ``count >= (s − epsilon) * N`` gives:

1. **No false negatives.** Every item with true frequency ``>= s*N`` is
   reported.
2. **Bounded underestimate.** A reported count is at most ``epsilon*N``
   below the truth — never above it.
3. **Bounded memory.** At most ``(1/epsilon) * log(epsilon * N)`` entries for
   a single-item stream, independent of how many distinct items pass through.
   In the transaction variant an observation inserts a whole basket between
   prunes, so the table can exceed that figure by roughly the mean basket
   size -- and on real data it does. :attr:`StreamingStats.space_bound`
   reports the scaled bound that actually applies; the unscaled classical one
   is kept alongside it as ``classical_space_bound`` rather than quietly
   presented as a guarantee it does not provide here.

All three are asserted in ``tests/test_streaming.py`` against exact counts.

The trade-off is explicit and tunable: ``epsilon = 0.001`` bounds the count
error at 0.1% of the stream length and the table at a few thousand entries,
whatever passes through.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Sequence

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class StreamingStats:
    """What a streaming pass cost and how accurate it is guaranteed to be."""

    n_observations: int = 0
    n_buckets: int = 0
    bucket_width: int = 0
    epsilon: float = 0.0
    entries_tracked: int = 0
    peak_entries: int = 0
    entries_evicted: int = 0
    distinct_seen: int = 0
    runtime_sec: float = 0.0

    @property
    def max_count_error(self) -> float:
        """Guaranteed bound on how far any count can undershoot, in observations."""
        return self.epsilon * self.n_observations

    #: Mean distinct items per observation. 1.0 for a plain element stream.
    mean_items_per_observation: float = 1.0

    @property
    def classical_space_bound(self) -> float:
        """Manku & Motwani's space bound, ``(1/eps) * log(eps * N)``.

        **This bound assumes one item per observation.** It is exact for a
        plain element stream and does *not* bind for the transaction variant:
        a basket inserts all of its items between two prunes, so the table
        can hold up to roughly ``mean_items_per_observation`` times more. The
        measured value genuinely exceeds this bound on real basket data, and
        reporting it as a guarantee there would be wrong.

        Use :attr:`space_bound` for the bound that actually applies.
        """
        if self.epsilon <= 0 or self.n_observations <= 0:
            return 0.0
        value = self.epsilon * self.n_observations
        return (1.0 / self.epsilon) * math.log(value) if value > 1 else 1.0 / self.epsilon

    @property
    def space_bound(self) -> float:
        """The space bound that applies to this stream.

        The classical bound scaled by the mean number of distinct items an
        observation contributes. For a single-item stream the factor is 1 and
        this equals :attr:`classical_space_bound`.
        """
        return self.classical_space_bound * max(self.mean_items_per_observation, 1.0)

    @property
    def within_space_bound(self) -> bool:
        return self.peak_entries <= self.space_bound

    @property
    def memory_saving(self) -> float:
        """Distinct items seen, divided by entries actually held."""
        return self.distinct_seen / self.peak_entries if self.peak_entries else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "n_buckets": self.n_buckets,
            "bucket_width": self.bucket_width,
            "epsilon": self.epsilon,
            "entries_tracked": self.entries_tracked,
            "peak_entries": self.peak_entries,
            "entries_evicted": self.entries_evicted,
            "distinct_seen": self.distinct_seen,
            "max_count_error": round(self.max_count_error, 2),
            "mean_items_per_observation": round(self.mean_items_per_observation, 2),
            "classical_space_bound": round(self.classical_space_bound, 1),
            "space_bound": round(self.space_bound, 1),
            "within_space_bound": self.within_space_bound,
            "memory_saving": round(self.memory_saving, 2),
            "runtime_sec": round(self.runtime_sec, 3),
        }

    def summary(self) -> str:
        return (
            f"Lossy Counting: {self.n_observations:,} observations, "
            f"{self.distinct_seen:,} distinct items -> {self.peak_entries:,} entries held "
            f"({self.memory_saving:.1f}x fewer), eps={self.epsilon:g}, "
            f"count error <= {self.max_count_error:,.0f}, {self.runtime_sec:.2f}s"
        )


@dataclass
class LossyCounter:
    """Single-pass approximate frequency counter with bounded memory.

    The bucketing unit is the **observation**, and an observation is one
    *transaction* — not one item occurrence. That is Manku & Motwani's own
    adaptation for the itemset setting (§4.2), and getting it wrong is subtle:
    a basket contributes ~24 item occurrences, so bucketing on occurrences
    would express both the support threshold and the error bound in units 24x
    larger than the counts they are compared against, and nothing would ever
    be reported.

    With transaction bucketing, an item's tracked count *is* the number of
    transactions containing it, so support, threshold and error bound are all
    in the same unit.

    For a plain stream of single items, :meth:`update` treats each element as
    its own one-item transaction, which recovers classical Lossy Counting
    exactly.

    Usable incrementally — feed it batches as they arrive and query
    :meth:`frequent` at any point::

        counter = LossyCounter(epsilon=0.001)
        for basket in live_feed():
            counter.update_transaction(basket)
        print(counter.frequent(min_support=0.01))
    """

    epsilon: float = 0.001
    #: item -> (count, delta)
    entries: dict[Hashable, tuple[int, int]] = field(default_factory=dict)
    n_observations: int = 0
    _distinct: set = field(default_factory=set, repr=False)
    peak_entries: int = 0
    entries_evicted: int = 0
    _items_seen: int = 0
    _track_distinct: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.epsilon < 1:
            raise ValueError(f"epsilon must be in (0, 1), got {self.epsilon}")
        self.bucket_width = max(1, int(math.ceil(1.0 / self.epsilon)))

    @property
    def current_bucket(self) -> int:
        """1-based index of the bucket currently filling."""
        return int(math.ceil(self.n_observations / self.bucket_width)) or 1

    def update_transaction(self, transaction: Sequence[Hashable]) -> None:
        """Observe one basket, counting each distinct item once."""
        items = set(transaction)
        previous_bucket = self.current_bucket
        self.n_observations += 1
        self._items_seen += len(items)
        if self._track_distinct:
            self._distinct |= items

        for item in items:
            existing = self.entries.get(item)
            if existing is None:
                # delta = b - 1: the most it could have occurred before we
                # started tracking it.
                self.entries[item] = (1, previous_bucket - 1)
            else:
                count, delta = existing
                self.entries[item] = (count + 1, delta)

        self.peak_entries = max(self.peak_entries, len(self.entries))

        # Prune at every bucket boundary.
        if self.n_observations % self.bucket_width == 0:
            self._prune(self.n_observations // self.bucket_width)

    def update(self, item: Hashable) -> None:
        """Observe a single element, as its own one-item transaction."""
        self.update_transaction((item,))

    def update_many(self, items: Iterable[Hashable]) -> None:
        """Observe each element as its own observation (classical form)."""
        for item in items:
            self.update(item)

    def _prune(self, bucket: int) -> None:
        """Drop entries that cannot become frequent."""
        before = len(self.entries)
        self.entries = {
            item: (count, delta)
            for item, (count, delta) in self.entries.items()
            if count + delta > bucket
        }
        self.entries_evicted += before - len(self.entries)

    def frequent(self, min_support: float = 0.01) -> pd.DataFrame:
        """Items whose count clears ``(min_support - epsilon) * N``.

        Using the *relaxed* threshold is what delivers the no-false-negative
        guarantee: a truly frequent item's tracked count may undershoot by up
        to ``epsilon * N``, so the bar has to be lowered by exactly that much.
        """
        if min_support <= self.epsilon:
            log.warning(
                "min_support %.4g <= epsilon %.4g: the error bound is as large as "
                "the threshold, so results are not meaningful. Lower epsilon.",
                min_support, self.epsilon,
            )
        threshold = (min_support - self.epsilon) * self.n_observations
        rows = [
            {
                "item": item,
                "count": count,
                "support": count / self.n_observations if self.n_observations else 0.0,
                "max_true_count": count + delta,
                "delta": delta,
            }
            for item, (count, delta) in self.entries.items()
            if count >= threshold
        ]
        if not rows:
            return pd.DataFrame(
                columns=["item", "count", "support", "max_true_count", "delta"]
            )
        return (
            pd.DataFrame(rows)
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )

    def stats(self, runtime_sec: float = 0.0) -> StreamingStats:
        mean_items = (
            self._items_seen / self.n_observations if self.n_observations else 1.0
        )
        return StreamingStats(
            mean_items_per_observation=mean_items,
            n_observations=self.n_observations,
            n_buckets=self.current_bucket,
            bucket_width=self.bucket_width,
            epsilon=self.epsilon,
            entries_tracked=len(self.entries),
            peak_entries=self.peak_entries,
            entries_evicted=self.entries_evicted,
            distinct_seen=len(self._distinct) if self._track_distinct else 0,
            runtime_sec=runtime_sec,
        )


def stream_frequent_items(
    transactions: Iterable[Sequence[Hashable]],
    min_support: float = 0.01,
    epsilon: float | None = None,
    progress_every: int = 0,
) -> tuple[pd.DataFrame, StreamingStats]:
    """Mine frequent items from a transaction stream in one bounded-memory pass.

    Parameters
    ----------
    transactions:
        Any iterable of baskets. Consumed **once**; never materialised.
    min_support:
        Reporting threshold, as a fraction of transactions.
    epsilon:
        Error tolerance. Defaults to ``min_support / 10``, the usual choice —
        it bounds the count error at a tenth of the threshold while keeping
        the table small.
    """
    import time

    epsilon = epsilon if epsilon is not None else min_support / 10.0
    counter = LossyCounter(epsilon=epsilon)

    t0 = time.perf_counter()
    for index, transaction in enumerate(transactions, start=1):
        counter.update_transaction(transaction)
        if progress_every and index % progress_every == 0:
            log.info(
                "  %s transactions, %s entries held",
                f"{index:,}", f"{len(counter.entries):,}",
            )
    runtime = time.perf_counter() - t0

    # Counts are already per transaction, so support needs no rescaling.
    frame = counter.frequent(min_support=min_support)
    stats = counter.stats(runtime)
    log.info("%s", stats.summary())
    return frame, stats


def compare_with_exact(
    transactions: Sequence[Sequence[Hashable]],
    min_support: float = 0.01,
    epsilon: float | None = None,
) -> pd.DataFrame:
    """Run Lossy Counting beside an exact count and report the error.

    Used by the tests and the CLI demo to show the guarantees holding rather
    than merely claiming them: no false negatives, and every approximate
    count at or below the true one by no more than ``epsilon * N``.
    """
    from collections import Counter

    exact = Counter()
    for transaction in transactions:
        exact.update(set(transaction))

    n = len(transactions)
    approximate, stats = stream_frequent_items(
        transactions, min_support=min_support, epsilon=epsilon
    )
    approx_counts = dict(zip(approximate["item"], approximate["count"]))

    truly_frequent = {i: c for i, c in exact.items() if c >= min_support * n}
    rows = []
    for item in set(approx_counts) | set(truly_frequent):
        true_count = exact.get(item, 0)
        approx_count = approx_counts.get(item)
        rows.append(
            {
                "item": item,
                "true_count": true_count,
                "approx_count": approx_count,
                "error": (true_count - approx_count) if approx_count is not None else None,
                "truly_frequent": item in truly_frequent,
                "reported": approx_count is not None,
                "false_negative": item in truly_frequent and approx_count is None,
                "within_error_bound": (
                    approx_count is not None
                    and 0 <= true_count - approx_count <= stats.max_count_error
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("true_count", ascending=False).reset_index(drop=True)
