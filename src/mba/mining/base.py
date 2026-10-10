"""Shared types for the frequent-itemset miners.

Both :func:`mba.mining.apriori.apriori` and
:func:`mba.mining.fpgrowth.fpgrowth` return a :class:`MiningResult`, so the
rule generator, the benchmark harness and the dashboard never need to know
which algorithm produced the itemsets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MiningStats:
    """Instrumentation captured during a mining run."""

    algorithm: str
    min_support: float
    max_len: int
    n_transactions: int
    n_items: int
    n_itemsets: int = 0
    runtime_sec: float = 0.0
    peak_memory_mb: float = 0.0
    #: Candidates generated per level (Apriori) or conditional trees built
    #: (FP-Growth) — the headline cost driver for each algorithm.
    candidates_per_level: dict[int, int] = field(default_factory=dict)
    itemsets_per_level: dict[int, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "min_support": self.min_support,
            "max_len": self.max_len,
            "n_transactions": self.n_transactions,
            "n_items": self.n_items,
            "n_itemsets": self.n_itemsets,
            "runtime_sec": round(self.runtime_sec, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "total_candidates": sum(self.candidates_per_level.values()),
            "candidates_per_level": dict(self.candidates_per_level),
            "itemsets_per_level": dict(self.itemsets_per_level),
            **self.extra,
        }

    def summary(self) -> str:
        return (
            f"{self.algorithm:9s} min_support={self.min_support:<6.4g} "
            f"-> {self.n_itemsets:,} itemsets in {self.runtime_sec:.3f}s "
            f"(peak {self.peak_memory_mb:.1f} MB)"
        )


@dataclass
class MiningResult:
    """Frequent itemsets plus the statistics of the run that produced them."""

    itemsets: pd.DataFrame
    stats: MiningStats

    def __len__(self) -> int:
        return len(self.itemsets)

    @property
    def empty(self) -> bool:
        return self.itemsets.empty

    def by_length(self) -> pd.Series:
        """Count of frequent itemsets per cardinality."""
        if self.itemsets.empty:
            return pd.Series(dtype="int64", name="n_itemsets")
        return (
            self.itemsets.groupby("length").size().rename("n_itemsets").sort_index()
        )

    def top(self, n: int = 20) -> pd.DataFrame:
        """Highest-support itemsets, most frequent first."""
        return self.itemsets.sort_values("support", ascending=False).head(n)

    def multi_item(self) -> pd.DataFrame:
        """Only itemsets of size >= 2 — the ones that can form a rule."""
        return self.itemsets.loc[self.itemsets["length"] >= 2]

    def to_display_frame(self) -> pd.DataFrame:
        """Human/CSV-friendly copy with frozensets rendered as strings."""
        out = self.itemsets.copy()
        out["itemsets"] = out["itemsets"].map(lambda s: ", ".join(sorted(s)))
        return out


def empty_itemsets_frame() -> pd.DataFrame:
    """The canonical empty result, with the right columns and dtypes."""
    return pd.DataFrame(
        {
            "support": pd.Series(dtype="float64"),
            "itemsets": pd.Series(dtype="object"),
            "count": pd.Series(dtype="int64"),
            "length": pd.Series(dtype="int64"),
        }
    )


def build_itemsets_frame(
    records: list[tuple[frozenset, int]],
    n_transactions: int,
) -> pd.DataFrame:
    """Assemble the standard itemset frame from ``(itemset, count)`` pairs."""
    if not records:
        return empty_itemsets_frame()

    itemsets, counts = zip(*records)
    frame = pd.DataFrame(
        {
            "support": [c / n_transactions for c in counts],
            "itemsets": list(itemsets),
            "count": list(counts),
            "length": [len(s) for s in itemsets],
        }
    )
    return frame.sort_values(["length", "support"], ascending=[True, False]).reset_index(drop=True)


#: Registry populated by the two miner modules at import time.
_MINERS: dict[str, Callable[..., MiningResult]] = {}


def register_miner(name: str, fn: Callable[..., MiningResult]) -> None:
    _MINERS[name.lower()] = fn


def available_miners() -> list[str]:
    return sorted(_MINERS)


def mine(basket, algorithm: str = "fpgrowth", **kwargs) -> MiningResult:
    """Dispatch to a registered miner by name.

    Lets the CLI, the benchmark harness and the dashboard switch algorithms
    from configuration without importing either implementation directly.
    """
    key = (algorithm or "").lower()
    if key not in _MINERS:
        # Import for side-effect registration, then retry once.
        from mba.mining import apriori as _a, fpgrowth as _f  # noqa: F401

        if key not in _MINERS:
            raise ValueError(
                f"unknown mining algorithm {algorithm!r}; available: {available_miners()}"
            )
    return _MINERS[key](basket, **kwargs)


class _Timer:
    """Context manager recording wall-clock seconds and peak traced memory."""

    def __init__(self, track_memory: bool = True) -> None:
        self.track_memory = track_memory
        self.elapsed = 0.0
        self.peak_mb = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "_Timer":
        if self.track_memory:
            import tracemalloc

            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._started_tracing = True
            else:
                self._started_tracing = False
            tracemalloc.reset_peak()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
        if self.track_memory:
            import tracemalloc

            _, peak = tracemalloc.get_traced_memory()
            self.peak_mb = peak / 1e6
            if getattr(self, "_started_tracing", False):
                tracemalloc.stop()
