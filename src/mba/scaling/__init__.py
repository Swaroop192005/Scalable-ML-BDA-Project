"""Scalability: benchmarking, out-of-core mining and the Spark reference."""

from mba.scaling.benchmark import BenchmarkResult, benchmark_algorithms, scaling_curve
from mba.scaling.streaming import (
    LossyCounter,
    StreamingStats,
    compare_with_exact,
    stream_frequent_items,
)
from mba.scaling.outofcore import (
    OutOfCoreResult,
    SyntheticTransactions,
    mine_out_of_core,
    peak_rss_mb,
)

__all__ = [
    "BenchmarkResult",
    "LossyCounter",
    "OutOfCoreResult",
    "StreamingStats",
    "SyntheticTransactions",
    "benchmark_algorithms",
    "compare_with_exact",
    "mine_out_of_core",
    "peak_rss_mb",
    "scaling_curve",
    "stream_frequent_items",
]
