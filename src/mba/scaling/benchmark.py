"""Benchmark harness: Apriori versus FP-Growth.

The interesting question is not "which is faster" but *where the crossover
sits*. Both algorithms return identical itemsets, so the comparison is purely
about cost, and the cost curves have different shapes:

* **Apriori** pays for *candidate generation*. The number of candidate
  2-itemsets is quadratic in the number of frequent items, and the frequent
  item count grows sharply as the support threshold falls. Its curve bends
  upward steeply.
* **FP-Growth** pays a fixed price to build the FP-tree — a real cost at high
  support, where there is little to mine — and then grows roughly with the
  number of patterns that actually exist.

So at a permissive threshold Apriori can win outright (few frequent items,
cheap candidate sets, no tree to build), while at a strict one FP-Growth
pulls away. :func:`benchmark_algorithms` sweeps the support grid and records
both curves plus the candidate counts that explain them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from mba.config import Config
from mba.data.basket import BasketMatrix
from mba.mining.base import mine

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Tabular benchmark output plus derived commentary."""

    frame: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def pivot(self, value: str = "runtime_sec") -> pd.DataFrame:
        """Support threshold as the index, one column per algorithm."""
        if self.frame.empty:
            return pd.DataFrame()
        return self.frame.pivot_table(
            index="min_support", columns="algorithm", values=value, aggfunc="mean"
        ).sort_index(ascending=False)

    def speedup(self) -> pd.DataFrame:
        """Apriori runtime divided by FP-Growth runtime, per threshold.

        Values above 1 mean FP-Growth is the faster of the two.
        """
        wide = self.pivot("runtime_sec")
        if not {"apriori", "fpgrowth"}.issubset(wide.columns):
            return pd.DataFrame()
        out = wide.copy()
        out["speedup_fpgrowth_vs_apriori"] = wide["apriori"] / wide["fpgrowth"]
        return out

    def crossover(self) -> float | None:
        """Highest support threshold at which FP-Growth becomes the faster one."""
        sp = self.speedup()
        if sp.empty:
            return None
        faster = sp.loc[sp["speedup_fpgrowth_vs_apriori"] > 1.0]
        return float(faster.index.max()) if not faster.empty else None

    def verify_agreement(self) -> bool:
        """True when both algorithms found the same itemset count everywhere."""
        if self.frame.empty:
            return True
        counts = self.frame.pivot_table(
            index="min_support", columns="algorithm", values="n_itemsets", aggfunc="max"
        )
        return bool(counts.nunique(axis=1).le(1).all())

    def summary(self) -> str:
        if self.frame.empty:
            return "no benchmark rows"
        lines = ["Apriori vs FP-Growth", "-" * 60]
        wide = self.speedup()
        for support, row in wide.iterrows():
            a, f = row.get("apriori", float("nan")), row.get("fpgrowth", float("nan"))
            winner = "FP-Growth" if row.get("speedup_fpgrowth_vs_apriori", 0) > 1 else "Apriori"
            lines.append(
                f"  support={support:<7.4g} apriori={a:7.3f}s  fpgrowth={f:7.3f}s  "
                f"-> {winner} by {max(a / f, f / a):.2f}x"
            )
        x = self.crossover()
        lines.append("-" * 60)
        lines.append(
            f"FP-Growth overtakes Apriori at min_support <= {x:.4g}"
            if x is not None
            else "Apriori stayed ahead across the whole grid"
        )
        lines.append(
            "itemsets identical across algorithms: "
            + ("yes" if self.verify_agreement() else "NO — investigate")
        )
        return "\n".join(lines)


def benchmark_algorithms(
    basket: BasketMatrix,
    support_grid: Sequence[float] | None = None,
    algorithms: Sequence[str] = ("apriori", "fpgrowth"),
    max_len: int = 3,
    repeats: int = 1,
    cfg: Config | None = None,
    timeout_guard_sec: float = 300.0,
) -> BenchmarkResult:
    """Sweep *support_grid* for each algorithm and record cost metrics.

    An algorithm that blows past *timeout_guard_sec* at one threshold is
    skipped for every stricter threshold below it — without the guard a
    sub-1% Apriori run can take longer than the rest of the sweep combined.
    """
    if cfg is not None and support_grid is None:
        support_grid = cfg.get("scaling.benchmark.support_grid", [0.05, 0.03, 0.02, 0.01])
    support_grid = sorted(support_grid or [0.05, 0.03, 0.02, 0.01], reverse=True)

    rows: list[dict[str, Any]] = []
    abandoned: set[str] = set()

    for support in support_grid:
        for algo in algorithms:
            if algo in abandoned:
                log.info("skipping %s at support=%.4g (exceeded time guard earlier)", algo, support)
                rows.append(
                    {
                        "algorithm": algo,
                        "min_support": support,
                        "runtime_sec": float("nan"),
                        "status": "skipped_timeout_guard",
                    }
                )
                continue

            best: dict[str, Any] | None = None
            for rep in range(max(1, repeats)):
                result = mine(
                    basket,
                    algorithm=algo,
                    min_support=support,
                    max_len=max_len,
                    track_memory=True,
                )
                stats = result.stats.to_dict()
                stats["repeat"] = rep
                stats["status"] = "ok"
                # Keep the fastest repeat — the usual convention for timing.
                if best is None or stats["runtime_sec"] < best["runtime_sec"]:
                    best = stats

            assert best is not None
            rows.append(best)
            log.info(
                "%-9s support=%-7.4g %7.3fs  %6s itemsets  %6.1f MB peak",
                algo,
                support,
                best["runtime_sec"],
                f"{best['n_itemsets']:,}",
                best["peak_memory_mb"],
            )
            if best["runtime_sec"] > timeout_guard_sec:
                log.warning(
                    "%s took %.1fs at support=%.4g; skipping lower thresholds",
                    algo,
                    best["runtime_sec"],
                    support,
                )
                abandoned.add(algo)

    frame = pd.DataFrame(rows)
    # Nested dicts do not belong in a CSV; keep the aggregates only.
    for col in ("candidates_per_level", "itemsets_per_level"):
        if col in frame.columns:
            frame[col] = frame[col].map(lambda d: str(d) if isinstance(d, dict) else d)

    meta = {
        "n_transactions": basket.n_transactions,
        "n_items": basket.n_items,
        "nnz": int(basket.matrix.nnz),
        "max_len": max_len,
        "repeats": repeats,
    }
    result = BenchmarkResult(frame, meta)
    log.info("\n%s", result.summary())
    return result


def scaling_curve(
    basket: BasketMatrix,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    algorithm: str = "fpgrowth",
    min_support: float = 0.02,
    max_len: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Runtime as a function of transaction volume, at fixed *relative* support.

    Supports stay proportional, so the pattern set is roughly stable and the
    curve isolates the algorithm's scaling in *N* rather than conflating it
    with a growing result set.
    """
    import numpy as np

    from mba.data.basket import BasketMatrix as BM

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for fraction in fractions:
        n = max(1, int(round(basket.n_transactions * fraction)))
        idx = rng.choice(basket.n_transactions, size=n, replace=False)
        idx.sort()
        sub = BM(basket.matrix[idx].tocsr(), basket.items, basket.transaction_ids[idx])
        result = mine(sub, algorithm=algorithm, min_support=min_support, max_len=max_len)
        rows.append(
            {
                "fraction": fraction,
                "n_transactions": n,
                "nnz": int(sub.matrix.nnz),
                "algorithm": algorithm,
                "runtime_sec": result.stats.runtime_sec,
                "peak_memory_mb": result.stats.peak_memory_mb,
                "n_itemsets": result.stats.n_itemsets,
            }
        )
        log.info(
            "scaling: %6.0f%% of data (%s tx) -> %.3fs",
            fraction * 100,
            f"{n:,}",
            result.stats.runtime_sec,
        )

    return pd.DataFrame(rows)
