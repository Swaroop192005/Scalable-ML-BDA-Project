"""Apriori frequent-itemset mining.

Classical level-wise algorithm (Agrawal & Srikant, VLDB 1994):

1. Count 1-itemsets, keep those at or above ``min_support``: that is *L1*.
2. From *L(k-1)* generate candidate *k*-itemsets by the **F(k-1) x F(k-1)
   join**: merge two frequent (k-1)-itemsets that share their first k-2 items.
3. **Prune** any candidate with an infrequent (k-1)-subset — the downward
   closure property, and the whole reason Apriori is cheaper than brute force.
4. Count the survivors against the database; repeat until no candidate
   survives or ``max_len`` is reached.

The *structure* above is textbook Apriori. The *support counting* in step 4
uses vertical tid-sets — each itemset carries the sorted array of transaction
ids containing it, and a candidate's count is the size of the intersection of
two parents' tid-sets. This is mathematically identical to re-scanning the
database per level, but it replaces a Python-level pass over every basket
with a single vectorised ``numpy.intersect1d``, which is what makes a pure
Python Apriori tractable on half a million basket lines.

That design choice is also exactly why Apriori still loses to FP-Growth at
low support: no counting trick removes the combinatorial explosion of
*candidate generation* itself, which is the cost FP-Growth avoids entirely.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from mba.data.basket import BasketMatrix
from mba.mining.base import (
    MiningResult,
    MiningStats,
    _Timer,
    build_itemsets_frame,
    register_miner,
)

log = logging.getLogger(__name__)


def _tidsets_from_basket(basket: BasketMatrix) -> list[np.ndarray]:
    """Vertical layout: for each item column, the sorted transaction ids.

    Built once by transposing the CSR matrix to CSC, which is a single
    linear-time pass inside scipy rather than a Python loop.
    """
    csc = basket.matrix.tocsc()
    return [
        csc.indices[csc.indptr[j] : csc.indptr[j + 1]].astype(np.int32, copy=False)
        for j in range(basket.n_items)
    ]


def _join_candidates(
    prev_level: Sequence[tuple[int, ...]],
) -> list[tuple[tuple[int, ...], int, int]]:
    """F(k-1) x F(k-1) join.

    Itemsets are kept as *sorted integer tuples*, so two of them can be merged
    iff they agree on their first k-2 positions. Sorting the previous level
    means those pairs are adjacent, giving the classic blocked join rather
    than an O(n^2) sweep.

    Returns triples of ``(candidate, left_index, right_index)``; the parent
    indices let the caller intersect the two parents' tid-sets directly.
    """
    candidates: list[tuple[tuple[int, ...], int, int]] = []
    n = len(prev_level)
    i = 0
    while i < n:
        prefix = prev_level[i][:-1]
        j = i + 1
        while j < n and prev_level[j][:-1] == prefix:
            j += 1
        # prev_level[i:j] all share `prefix`; merge every pair within the block.
        for a in range(i, j):
            for b in range(a + 1, j):
                merged = prev_level[a] + (prev_level[b][-1],)
                candidates.append((merged, a, b))
        i = j
    return candidates


def _has_infrequent_subset(
    candidate: tuple[int, ...],
    prev_frequent: set[tuple[int, ...]],
) -> bool:
    """Downward-closure prune: is any (k-1)-subset of *candidate* infrequent?

    Only the subsets not covered by the join need checking, but testing all
    k of them is cheap and keeps the implementation obviously correct.
    """
    for subset in combinations(candidate, len(candidate) - 1):
        if subset not in prev_frequent:
            return True
    return False


def apriori(
    basket: BasketMatrix,
    min_support: float = 0.02,
    max_len: int | None = 3,
    min_count: int | None = None,
    track_memory: bool = True,
    verbose: bool = False,
) -> MiningResult:
    """Mine frequent itemsets with Apriori.

    Parameters
    ----------
    basket:
        Sparse transaction x item matrix.
    min_support:
        Minimum fraction of transactions an itemset must appear in.
    max_len:
        Largest itemset cardinality to mine; ``None`` runs to exhaustion.
    min_count:
        Absolute support floor, overriding *min_support* when given.
    track_memory:
        Record peak traced allocation (adds ~10% runtime; the benchmark
        harness turns it off for timing-only passes).

    Returns
    -------
    MiningResult
        ``.itemsets`` has columns ``support, itemsets, count, length``;
        ``.stats`` carries runtime, peak memory and candidates per level.
    """
    n_tx = basket.n_transactions
    stats = MiningStats(
        algorithm="apriori",
        min_support=min_support,
        max_len=max_len or -1,
        n_transactions=n_tx,
        n_items=basket.n_items,
    )
    if n_tx == 0 or basket.n_items == 0:
        return MiningResult(build_itemsets_frame([], 1), stats)

    threshold = int(min_count) if min_count is not None else int(np.ceil(min_support * n_tx))
    threshold = max(threshold, 1)
    limit = max_len if max_len and max_len > 0 else basket.n_items
    labels = basket.items

    results: list[tuple[frozenset, int]] = []

    with _Timer(track_memory) as timer:
        # ---- L1: single items ------------------------------------------
        counts = basket.support_counts()
        frequent_ids = np.flatnonzero(counts >= threshold)
        stats.candidates_per_level[1] = int(basket.n_items)

        if frequent_ids.size:
            all_tidsets = _tidsets_from_basket(basket)
            level: list[tuple[int, ...]] = [(int(i),) for i in frequent_ids]
            tidsets: list[np.ndarray] = [all_tidsets[i] for i in frequent_ids]
            del all_tidsets
            for ids, tids in zip(level, tidsets):
                results.append((frozenset((labels[ids[0]],)), int(tids.size)))
            stats.itemsets_per_level[1] = len(level)
        else:
            level, tidsets = [], []
            stats.itemsets_per_level[1] = 0

        if verbose:
            log.info("L1: %d/%d items frequent (count >= %d)", len(level), basket.n_items, threshold)

        # ---- Lk: level-wise generate / prune / count ---------------------
        k = 2
        while level and k <= limit:
            prev_frequent = set(level)
            raw_candidates = _join_candidates(level)
            stats.candidates_per_level[k] = len(raw_candidates)

            next_level: list[tuple[int, ...]] = []
            next_tidsets: list[np.ndarray] = []

            for candidate, a, b in raw_candidates:
                if k > 2 and _has_infrequent_subset(candidate, prev_frequent):
                    continue  # pruned by downward closure, never counted
                tids = np.intersect1d(tidsets[a], tidsets[b], assume_unique=True)
                if tids.size >= threshold:
                    next_level.append(candidate)
                    next_tidsets.append(tids)
                    results.append(
                        (frozenset(labels[i] for i in candidate), int(tids.size))
                    )

            stats.itemsets_per_level[k] = len(next_level)
            if verbose:
                log.info(
                    "L%d: %d candidates -> %d frequent",
                    k,
                    len(raw_candidates),
                    len(next_level),
                )
            level, tidsets = next_level, next_tidsets
            k += 1

    stats.runtime_sec = timer.elapsed
    stats.peak_memory_mb = timer.peak_mb
    stats.n_itemsets = len(results)
    stats.extra["min_count_threshold"] = threshold

    log.info("%s", stats.summary())
    return MiningResult(build_itemsets_frame(results, n_tx), stats)


def apriori_from_transactions(
    transactions: Iterable[Sequence[str]],
    min_support: float = 0.02,
    max_len: int | None = 3,
    **kwargs,
) -> MiningResult:
    """Convenience wrapper for list-of-lists input (used by the tests)."""
    from mba.data.basket import baskets_from_transactions

    return apriori(
        baskets_from_transactions(transactions),
        min_support=min_support,
        max_len=max_len,
        **kwargs,
    )


register_miner("apriori", apriori)
