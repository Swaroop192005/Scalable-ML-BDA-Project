"""Scalability: benchmarking, out-of-core mining and the Spark reference."""

from mba.scaling.benchmark import BenchmarkResult, benchmark_algorithms, scaling_curve
from mba.scaling.outofcore import (
    OutOfCoreResult,
    SyntheticTransactions,
    mine_out_of_core,
    peak_rss_mb,
)

__all__ = [
    "BenchmarkResult",
    "OutOfCoreResult",
    "SyntheticTransactions",
    "benchmark_algorithms",
    "mine_out_of_core",
    "peak_rss_mb",
    "scaling_curve",
]
