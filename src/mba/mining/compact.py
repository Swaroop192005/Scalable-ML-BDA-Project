"""Compact representations: maximal and closed frequent itemsets.

The set of frequent itemsets is massively redundant. If `{a,b,c}` is
frequent then all seven of its non-empty subsets are too, and in a dense
dataset the count explodes combinatorially. Two lossless summaries fix this:

**Closed itemsets.** *X* is closed when no proper superset has the *same
support*. Dropping a non-closed itemset loses nothing, because its support
is recoverable: it equals the maximum support among its closed supersets.
So the closed set is **lossless for support** — you can reconstruct every
frequent itemset *and* its exact count.

**Maximal itemsets.** *X* is maximal when no proper superset is frequent at
all. The maximal set is far smaller still, but it only preserves the
*boundary* between frequent and infrequent: you can recover which itemsets
are frequent, not how frequent. Every maximal itemset is closed; the
converse does not hold.

The containment chain is therefore::

    maximal  ⊆  closed  ⊆  frequent

Why it matters here: a rule generator fed the closed set produces exactly
the same rules as one fed every frequent itemset, from a fraction of the
input. And the ratio ``|frequent| / |closed|`` is a direct measure of how
redundant the pattern set is — a useful diagnostic that a raw count hides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mba.mining.base import MiningResult, MiningStats

log = logging.getLogger(__name__)


@dataclass
class CompactResult:
    """Maximal and closed itemsets derived from a full frequent set."""

    frequent: pd.DataFrame
    closed: pd.DataFrame
    maximal: pd.DataFrame

    @property
    def n_frequent(self) -> int:
        return len(self.frequent)

    @property
    def n_closed(self) -> int:
        return len(self.closed)

    @property
    def n_maximal(self) -> int:
        return len(self.maximal)

    @property
    def closed_compression(self) -> float:
        return self.n_frequent / self.n_closed if self.n_closed else 0.0

    @property
    def maximal_compression(self) -> float:
        return self.n_frequent / self.n_maximal if self.n_maximal else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_frequent": self.n_frequent,
            "n_closed": self.n_closed,
            "n_maximal": self.n_maximal,
            "closed_compression": round(self.closed_compression, 2),
            "maximal_compression": round(self.maximal_compression, 2),
        }

    def summary(self) -> str:
        return (
            f"{self.n_frequent:,} frequent -> {self.n_closed:,} closed "
            f"({self.closed_compression:.1f}x, lossless for support) "
            f"-> {self.n_maximal:,} maximal "
            f"({self.maximal_compression:.1f}x, boundary only)"
        )


def _by_length(frame: pd.DataFrame) -> dict[int, list[tuple[frozenset, int]]]:
    """Group ``(itemset, count)`` pairs by cardinality."""
    buckets: dict[int, list[tuple[frozenset, int]]] = {}
    for itemset, count in zip(frame["itemsets"], frame["count"]):
        buckets.setdefault(len(itemset), []).append((itemset, int(count)))
    return buckets


def closed_itemsets(itemsets: MiningResult | pd.DataFrame) -> pd.DataFrame:
    """Return the closed frequent itemsets.

    *X* is closed when no immediate superset shares its support. Checking
    only supersets **one item larger** is sufficient: if some larger
    superset had equal support, then by monotonicity every intermediate
    itemset — including one of size |X|+1 — would have that support too.
    That reduces the check from all supersets to one length bucket.
    """
    frame = itemsets.itemsets if isinstance(itemsets, MiningResult) else itemsets
    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()

    buckets = _by_length(frame)
    keep: list[bool] = []

    for itemset, count in zip(frame["itemsets"], frame["count"]):
        size = len(itemset)
        is_closed = True
        for superset, super_count in buckets.get(size + 1, ()):
            if int(super_count) == int(count) and itemset < superset:
                is_closed = False
                break
        keep.append(is_closed)

    out = frame.loc[pd.Series(keep, index=frame.index)].copy()
    log.info("closed itemsets: %d of %d frequent", len(out), len(frame))
    return out.reset_index(drop=True)


def maximal_itemsets(itemsets: MiningResult | pd.DataFrame) -> pd.DataFrame:
    """Return the maximal frequent itemsets.

    *X* is maximal when no frequent superset exists. As with closure, only
    the next length bucket needs checking — a frequent superset of any size
    implies a frequent superset of size |X|+1 by downward closure.

    Note that `max_len` truncation makes the longest mined length
    artificially maximal: with ``max_len=3`` every frequent 3-itemset is
    "maximal" only because 4-itemsets were never mined. The returned frame
    carries a ``maximal_by_truncation`` column marking those, so the
    distinction is not silently lost.
    """
    frame = itemsets.itemsets if isinstance(itemsets, MiningResult) else itemsets
    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()

    buckets = _by_length(frame)
    longest = max(buckets) if buckets else 0
    keep: list[bool] = []

    for itemset in frame["itemsets"]:
        size = len(itemset)
        has_frequent_superset = any(
            itemset < superset for superset, _ in buckets.get(size + 1, ())
        )
        keep.append(not has_frequent_superset)

    out = frame.loc[pd.Series(keep, index=frame.index)].copy()
    out["maximal_by_truncation"] = out["length"] >= longest
    log.info(
        "maximal itemsets: %d of %d frequent (%d at the max_len boundary)",
        len(out), len(frame), int(out["maximal_by_truncation"].sum()),
    )
    return out.reset_index(drop=True)


def compact_representations(itemsets: MiningResult | pd.DataFrame) -> CompactResult:
    """Compute both summaries and return them with the original set."""
    frame = itemsets.itemsets if isinstance(itemsets, MiningResult) else itemsets
    result = CompactResult(
        frequent=frame,
        closed=closed_itemsets(frame),
        maximal=maximal_itemsets(frame),
    )
    log.info("%s", result.summary())
    return result


def recover_support(
    itemset: frozenset,
    closed: pd.DataFrame,
) -> int | None:
    """Recover any frequent itemset's exact support from the closed set alone.

    This is the proof that closure is lossless: the support of *X* equals the
    maximum support among closed itemsets containing *X*. Returns ``None``
    when *X* is not frequent.

    ``tests/test_compact.py`` runs this over every frequent itemset and
    asserts the recovered counts match the mined ones exactly.
    """
    best: int | None = None
    for candidate, count in zip(closed["itemsets"], closed["count"]):
        if itemset <= candidate:
            value = int(count)
            if best is None or value > best:
                best = value
    return best
