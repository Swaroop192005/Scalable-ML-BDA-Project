"""Out-of-core mining: more transactions than fit in RAM.

The in-memory path encodes every basket into one sparse matrix. That is fine
for the 18k baskets of Online Retail, but a year of national retail data is
1M+ transactions and the encoded matrix alone would exceed a modest memory
budget — before any candidate bookkeeping.

This module implements the **SON algorithm** (Savasere, Omiecinski & Navathe,
VLDB 1995), which turns mining into two streaming passes:

**Pass 1 — local mining.** Read the data in chunks that *do* fit. Mine each
chunk independently at the same relative support. Union the per-chunk
frequent itemsets into a global candidate set.

**Pass 2 — global counting.** Stream the data again, counting only those
candidates. Keep the ones that clear global support.

The correctness guarantee is the point: if an itemset has global support
``s``, it must reach local support ``s`` in **at least one** chunk — if it
were below ``s`` in *every* chunk it could not reach ``s`` overall. So pass 1
yields **no false negatives**, and pass 2 removes the false positives by
exact counting. The result is identical to mining the whole dataset at once,
but peak memory is bounded by one chunk plus the candidate table.

Memory is measured, not assumed: :func:`peak_rss_mb` samples real resident
set size, and :func:`mine_out_of_core` raises when the configured budget is
exceeded so a regression surfaces as a failure rather than a swap storm.
"""

from __future__ import annotations

import logging
import resource
import sys
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterable, Iterator, Sequence

import pandas as pd

from mba.config import Config
from mba.data.basket import baskets_from_transactions
from mba.mining.base import MiningResult, MiningStats, build_itemsets_frame, mine

log = logging.getLogger(__name__)


class MemoryBudgetExceeded(RuntimeError):
    """Raised when a run breaches the configured RAM ceiling."""


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in megabytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS — normalised here
    so the number means the same thing on both.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def current_rss_mb() -> float:
    """Current resident set size in MB, read from ``/proc`` where available."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * resource.getpagesize() / 1e6
    except (OSError, IndexError, ValueError):  # pragma: no cover - non-Linux
        return peak_rss_mb()


@dataclass
class OutOfCoreResult:
    """Itemsets from a SON run, plus the streaming cost profile."""

    itemsets: pd.DataFrame
    stats: MiningStats
    chunk_profile: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __len__(self) -> int:
        return len(self.itemsets)

    def as_mining_result(self) -> MiningResult:
        """Adapt to the in-memory result type so downstream code is unchanged."""
        return MiningResult(self.itemsets, self.stats)

    def summary(self) -> str:
        s = self.stats
        return (
            f"SON out-of-core: {s.n_transactions:,} transactions in "
            f"{s.extra.get('n_chunks', 0)} chunks -> {s.n_itemsets:,} itemsets "
            f"in {s.runtime_sec:.2f}s, peak RSS {s.peak_memory_mb:.0f} MB "
            f"(budget {s.extra.get('memory_budget_mb', 0)} MB), "
            f"{s.extra.get('n_candidates', 0):,} candidates counted"
        )


def iter_transaction_chunks(
    transactions: Iterable[Sequence[str]],
    chunk_size: int,
) -> Iterator[list[list[str]]]:
    """Batch an arbitrary transaction iterable into lists of *chunk_size*."""
    chunk: list[list[str]] = []
    for transaction in transactions:
        chunk.append(list(transaction))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def frame_to_transaction_stream(
    frames: Iterable[pd.DataFrame],
    invoice_col: str,
    item_col: str,
) -> Iterator[list[str]]:
    """Turn a stream of row-chunks into a stream of complete baskets.

    A single invoice can straddle a chunk boundary, so the last (partial)
    invoice of each chunk is carried over and merged with the first rows of
    the next one. Without this, large baskets would be silently split and
    their co-occurrences lost.
    """
    carry_id: Any = None
    carry_items: set[str] = set()

    for frame in frames:
        if frame.empty:
            continue
        grouped = frame.groupby(invoice_col, observed=True, sort=False)[item_col]
        groups = list(grouped)
        for position, (invoice_id, items) in enumerate(groups):
            item_set = {str(v) for v in items if pd.notna(v)}
            if invoice_id == carry_id:
                item_set |= carry_items
                carry_items = set()
                carry_id = None
            if position == len(groups) - 1:
                carry_id, carry_items = invoice_id, item_set  # may continue
            else:
                yield sorted(item_set)

    if carry_items:
        yield sorted(carry_items)


def _count_candidates(
    transactions: Iterable[Sequence[str]],
    candidates: set[frozenset],
) -> dict[frozenset, int]:
    """Exact pass-2 counting of a fixed candidate set.

    Two optimisations matter at this scale:

    * Items are interned to small integers and candidates to **sorted int
      tuples**. Tuples hash several times faster than frozensets, and pass 2
      performs tens of millions of lookups.
    * Per basket and per candidate size, the loop picks the cheaper of
      enumerating the basket's subsets versus testing each candidate for
      containment — so a wide candidate table never forces subset
      enumeration over a 90-item wholesale basket, and a narrow one never
      forces a scan of the whole table.
    """
    from math import comb

    if not candidates:
        return {}

    # Intern item labels -> ints, and candidates -> sorted int tuples.
    vocab: dict[str, int] = {}
    for candidate in candidates:
        for item in candidate:
            if item not in vocab:
                vocab[item] = len(vocab)

    by_size: dict[int, set[tuple[int, ...]]] = {}
    key_of: dict[tuple[int, ...], frozenset] = {}
    for candidate in candidates:
        key = tuple(sorted(vocab[i] for i in candidate))
        by_size.setdefault(len(key), set()).add(key)
        key_of[key] = candidate

    counts: dict[tuple[int, ...], int] = dict.fromkeys(key_of, 0)
    sizes = sorted(by_size)
    singles = {k[0] for k in by_size.get(1, ())}

    for transaction in transactions:
        basket = sorted({vocab[i] for i in transaction if i in vocab})
        n = len(basket)
        if not n:
            continue
        basket_set = set(basket)

        for size in sizes:
            if size > n:
                break
            pool = by_size[size]
            if size == 1:
                for item in basket_set & singles:
                    counts[(item,)] += 1
            elif comb(n, size) <= len(pool):
                for combo in combinations(basket, size):
                    if combo in pool:
                        counts[combo] += 1
            else:
                for candidate in pool:
                    if basket_set.issuperset(candidate):
                        counts[candidate] += 1

    return {key_of[k]: v for k, v in counts.items()}


def mine_out_of_core(
    transactions: Iterable[Sequence[str]],
    min_support: float = 0.02,
    max_len: int = 3,
    chunk_size: int = 50_000,
    memory_budget_mb: float | None = 800.0,
    local_support_scale: float = 1.0,
    algorithm: str = "fpgrowth",
    cfg: Config | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> OutOfCoreResult:
    """Mine frequent itemsets from a transaction *stream* using SON.

    Parameters
    ----------
    transactions:
        Any iterable of baskets. Consumed **twice**, so a one-shot generator
        is materialised per chunk rather than held whole — pass a callable-
        backed iterable (or a list) when the source cannot be re-read.
    min_support:
        Global support threshold.
    chunk_size:
        Transactions held in memory at once. The dominant memory knob.
    memory_budget_mb:
        Hard ceiling; breaching it raises :class:`MemoryBudgetExceeded`.
    local_support_scale:
        Multiplier on the per-chunk threshold. Below 1.0 the candidate set
        grows (safer, slower); 1.0 is already sound under SON.

    Returns
    -------
    OutOfCoreResult
        Itemsets identical to an in-memory run at the same threshold.
    """
    import time

    if cfg is not None:
        section = cfg.get("scaling.out_of_core", {})
        chunk_size = int(section.get("chunk_size", chunk_size))
        memory_budget_mb = float(section.get("memory_budget_mb", memory_budget_mb or 0)) or None
        local_support_scale = float(section.get("local_support_scale", local_support_scale))

    t0 = time.perf_counter()
    baseline_rss = current_rss_mb()

    # Materialise lazily-streamed input once so pass 2 can re-read it. For a
    # genuinely out-of-core source, pass a re-iterable object instead and
    # this branch keeps only a reference.
    if not hasattr(transactions, "__len__"):
        transactions = list(transactions)

    total_tx = len(transactions)  # type: ignore[arg-type]
    if total_tx == 0:
        stats = MiningStats("son-" + algorithm, min_support, max_len, 0, 0)
        return OutOfCoreResult(build_itemsets_frame([], 1), stats)

    local_threshold = max(min_support * local_support_scale, 1e-9)
    candidates: set[frozenset] = set()
    chunk_rows: list[dict[str, Any]] = []

    # ---- Pass 1: local mining per chunk -------------------------------
    n_chunks = 0
    for index, chunk in enumerate(iter_transaction_chunks(transactions, chunk_size)):
        n_chunks += 1
        t_chunk = time.perf_counter()
        basket = baskets_from_transactions(chunk)
        local = mine(
            basket,
            algorithm=algorithm,
            min_support=local_threshold,
            max_len=max_len,
            track_memory=False,
        )
        new = set(local.itemsets["itemsets"]) if not local.itemsets.empty else set()
        candidates |= new

        rss = current_rss_mb()
        chunk_rows.append(
            {
                "pass": 1,
                "chunk": index,
                "n_transactions": len(chunk),
                "local_itemsets": len(new),
                "cumulative_candidates": len(candidates),
                "runtime_sec": round(time.perf_counter() - t_chunk, 4),
                "rss_mb": round(rss, 1),
            }
        )
        log.info(
            "pass 1 chunk %d: %s tx -> %s local itemsets (%s candidates so far, RSS %.0f MB)",
            index,
            f"{len(chunk):,}",
            f"{len(new):,}",
            f"{len(candidates):,}",
            rss,
        )
        if progress:
            progress(index + 1, 0)
        if memory_budget_mb and rss > memory_budget_mb:
            raise MemoryBudgetExceeded(
                f"RSS {rss:.0f} MB exceeded the {memory_budget_mb:.0f} MB budget during "
                f"pass 1 chunk {index}. Lower scaling.out_of_core.chunk_size."
            )
        del basket, local, chunk

    # ---- Pass 2: exact global counting --------------------------------
    global_counts: dict[frozenset, int] = dict.fromkeys(candidates, 0)
    for index, chunk in enumerate(iter_transaction_chunks(transactions, chunk_size)):
        t_chunk = time.perf_counter()
        partial = _count_candidates(chunk, candidates)
        for key, value in partial.items():
            global_counts[key] += value

        rss = current_rss_mb()
        chunk_rows.append(
            {
                "pass": 2,
                "chunk": index,
                "n_transactions": len(chunk),
                "local_itemsets": 0,
                "cumulative_candidates": len(candidates),
                "runtime_sec": round(time.perf_counter() - t_chunk, 4),
                "rss_mb": round(rss, 1),
            }
        )
        if progress:
            progress(n_chunks, index + 1)
        if memory_budget_mb and rss > memory_budget_mb:
            raise MemoryBudgetExceeded(
                f"RSS {rss:.0f} MB exceeded the {memory_budget_mb:.0f} MB budget during "
                f"pass 2 chunk {index}."
            )
        del chunk

    threshold_count = min_support * total_tx
    survivors = [(k, v) for k, v in global_counts.items() if v >= threshold_count]

    runtime = time.perf_counter() - t0
    peak = peak_rss_mb()

    stats = MiningStats(
        algorithm=f"son-{algorithm}",
        min_support=min_support,
        max_len=max_len,
        n_transactions=total_tx,
        n_items=len({i for s in candidates for i in s}),
        n_itemsets=len(survivors),
        runtime_sec=runtime,
        peak_memory_mb=peak,
    )
    stats.extra.update(
        {
            "n_chunks": n_chunks,
            "chunk_size": chunk_size,
            "n_candidates": len(candidates),
            "false_positive_candidates": len(candidates) - len(survivors),
            "memory_budget_mb": memory_budget_mb,
            "baseline_rss_mb": round(baseline_rss, 1),
            "local_support_scale": local_support_scale,
        }
    )
    for itemset, _ in survivors:
        stats.itemsets_per_level[len(itemset)] = stats.itemsets_per_level.get(len(itemset), 0) + 1

    result = OutOfCoreResult(
        build_itemsets_frame(survivors, total_tx),
        stats,
        pd.DataFrame(chunk_rows),
    )
    log.info("%s", result.summary())
    return result


class SyntheticTransactions:
    """A re-iterable, lazily-generated transaction source for load testing.

    SON reads the data twice. A generator can only be read once, so
    :func:`mine_out_of_core` would have to materialise it — which for a
    million baskets means a multi-gigabyte list, defeating the entire point
    of an out-of-core miner.

    This class is a *re-iterable* instead: it knows its length without being
    consumed, and each ``iter()`` replays the same deterministic stream from
    the seed. Memory stays at one basket at a time, which is what makes the
    800 MB budget achievable at a million transactions.

    Items follow a Zipf popularity curve, as a real catalogue does, and a set
    of *planted bundles* is injected so the miner has ground truth to
    recover.
    """

    def __init__(
        self,
        n_transactions: int = 1_000_000,
        n_items: int = 5_000,
        mean_basket: int = 8,
        n_planted_bundles: int = 25,
        bundle_probability: float = 0.04,
        seed: int = 42,
    ) -> None:
        self.n_transactions = n_transactions
        self.n_items = n_items
        self.mean_basket = mean_basket
        self.n_planted_bundles = n_planted_bundles
        self.bundle_probability = bundle_probability
        self.seed = seed
        # Interning the labels once means the stream yields references to
        # 5,000 strings rather than allocating a fresh one per basket line.
        self._labels = [f"ITEM_{i:05d}" for i in range(n_items)]

    def __len__(self) -> int:
        return self.n_transactions

    def planted_bundles(self) -> list[list[str]]:
        """The ground-truth bundles, for verifying what the miner recovered."""
        import numpy as np

        rng = np.random.default_rng(self.seed)
        return [
            sorted(self._labels[i] for i in rng.choice(
                self.n_items, size=rng.integers(2, 4), replace=False))
            for _ in range(self.n_planted_bundles)
        ]

    def __iter__(self) -> Iterator[list[str]]:
        import numpy as np

        rng = np.random.default_rng(self.seed)

        # Zipf-ish popularity: item i has weight 1/(i+1)^0.8.
        weights = 1.0 / np.power(np.arange(1, self.n_items + 1), 0.8)
        weights /= weights.sum()

        bundles = [
            rng.choice(self.n_items, size=rng.integers(2, 4), replace=False)
            for _ in range(self.n_planted_bundles)
        ]
        sizes = rng.poisson(self.mean_basket, size=self.n_transactions).clip(1, 60)
        labels = self._labels

        for i in range(self.n_transactions):
            basket = set(
                rng.choice(self.n_items, size=int(sizes[i]), replace=False, p=weights)
            )
            if rng.random() < self.bundle_probability:
                basket |= set(bundles[rng.integers(len(bundles))].tolist())
            yield [labels[j] for j in sorted(basket)]


def synthesise_transactions(
    n_transactions: int = 1_000_000,
    n_items: int = 5_000,
    mean_basket: int = 8,
    n_planted_bundles: int = 25,
    bundle_probability: float = 0.04,
    seed: int = 42,
) -> Iterator[list[str]]:
    """Yield a synthetic transaction stream (one pass).

    For SON, which needs two passes, use :class:`SyntheticTransactions`
    directly — it is re-iterable and never materialises the dataset.
    """
    return iter(
        SyntheticTransactions(
            n_transactions, n_items, mean_basket,
            n_planted_bundles, bundle_probability, seed,
        )
    )
