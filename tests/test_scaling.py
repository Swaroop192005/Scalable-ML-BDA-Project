"""Out-of-core mining, benchmarking and the Spark bridge."""

import random

import pytest

from conftest import brute_force_itemsets
from mba.data.basket import baskets_from_transactions
from mba.mining.fpgrowth import fpgrowth
from mba.scaling.benchmark import benchmark_algorithms, scaling_curve
from mba.scaling.outofcore import (
    MemoryBudgetExceeded,
    OutOfCoreResult,
    frame_to_transaction_stream,
    iter_transaction_chunks,
    mine_out_of_core,
    peak_rss_mb,
    synthesise_transactions,
)


def as_dict(frame) -> dict:
    return {row.itemsets: row.count for row in frame.itertuples()}


# -------------------------------------------------------------- SON
@pytest.mark.parametrize("chunk_size", [2, 3, 7, 1000])
def test_son_matches_brute_force_at_every_chunk_size(toy_transactions, chunk_size):
    """The whole point of SON: chunking must not change the answer."""
    result = mine_out_of_core(
        toy_transactions, min_support=0.2, max_len=3,
        chunk_size=chunk_size, memory_budget_mb=None,
    )
    assert as_dict(result.itemsets) == brute_force_itemsets(toy_transactions, 0.2, 3)


def test_son_matches_in_memory_on_random_data():
    rng = random.Random(777)
    for _ in range(12):
        transactions = [
            [f"i{x}" for x in rng.sample(range(8), rng.randint(1, 5))]
            for _ in range(rng.randint(20, 60))
        ]
        min_support = rng.choice([0.1, 0.2, 0.3])
        in_memory = fpgrowth(
            baskets_from_transactions(transactions), min_support, 3, track_memory=False
        )
        out_of_core = mine_out_of_core(
            transactions, min_support=min_support, max_len=3,
            chunk_size=rng.randint(3, 15), memory_budget_mb=None,
        )
        assert as_dict(out_of_core.itemsets) == as_dict(in_memory.itemsets)


def test_son_pass_one_has_no_false_negatives(toy_transactions):
    """Candidates must be a superset of the true answer — the SON guarantee."""
    result = mine_out_of_core(
        toy_transactions, min_support=0.2, max_len=3, chunk_size=3, memory_budget_mb=None
    )
    truth = set(brute_force_itemsets(toy_transactions, 0.2, 3))
    assert truth <= set(result.itemsets["itemsets"])
    assert result.stats.extra["n_candidates"] >= len(truth)


def test_chunk_profile_covers_both_passes(toy_transactions):
    result = mine_out_of_core(
        toy_transactions, min_support=0.2, max_len=2, chunk_size=3, memory_budget_mb=None
    )
    assert set(result.chunk_profile["pass"]) == {1, 2}
    assert result.chunk_profile["n_transactions"].sum() == 2 * len(toy_transactions)


def test_memory_budget_is_enforced(toy_transactions):
    with pytest.raises(MemoryBudgetExceeded):
        mine_out_of_core(
            toy_transactions, min_support=0.2, chunk_size=2, memory_budget_mb=0.001
        )


def test_empty_stream():
    result = mine_out_of_core([], min_support=0.2)
    assert len(result) == 0


def test_result_adapts_to_the_in_memory_type(toy_transactions):
    result = mine_out_of_core(toy_transactions, min_support=0.2, memory_budget_mb=None)
    adapted = result.as_mining_result()
    assert adapted.stats.algorithm.startswith("son-")
    assert len(adapted) == len(result)


def test_chunking_helper_preserves_every_transaction(toy_transactions):
    chunks = list(iter_transaction_chunks(toy_transactions, 3))
    assert sum(len(c) for c in chunks) == len(toy_transactions)
    assert [t for c in chunks for t in c] == [list(t) for t in toy_transactions]


def test_invoice_spanning_a_chunk_boundary_is_rejoined():
    """A basket split across two row-chunks must come out whole."""
    import pandas as pd

    rows = pd.DataFrame(
        {
            "InvoiceNo": ["1", "1", "1", "2", "2"],
            "Description": ["a", "b", "c", "d", "e"],
        }
    )
    # Split mid-invoice: rows 0-1 then 2-4.
    chunks = [rows.iloc[:2], rows.iloc[2:]]
    baskets = list(frame_to_transaction_stream(chunks, "InvoiceNo", "Description"))
    assert sorted(baskets) == [["a", "b", "c"], ["d", "e"]]


def test_peak_rss_is_a_positive_number():
    assert peak_rss_mb() > 0


# ------------------------------------------------------- synthetic load
def test_synthetic_stream_is_lazy_and_well_formed():
    stream = synthesise_transactions(n_transactions=500, n_items=80, seed=1)
    baskets = [next(stream) for _ in range(20)]
    assert all(len(set(b)) == len(b) for b in baskets), "items repeat within a basket"
    assert all(b == sorted(b) for b in baskets)


def test_son_recovers_planted_bundles():
    transactions = list(
        synthesise_transactions(
            n_transactions=4000, n_items=200, mean_basket=5,
            n_planted_bundles=3, bundle_probability=0.25, seed=7,
        )
    )
    result = mine_out_of_core(
        transactions, min_support=0.03, max_len=3, chunk_size=1000, memory_budget_mb=None
    )
    multi = result.itemsets[result.itemsets["length"] >= 2]
    assert not multi.empty, "planted bundles were not recovered"


# ---------------------------------------------------------- benchmarking
def test_benchmark_runs_both_algorithms(toy_basket):
    result = benchmark_algorithms(toy_basket, support_grid=[0.3, 0.2], max_len=2)
    assert set(result.frame["algorithm"]) == {"apriori", "fpgrowth"}
    assert len(result.frame) == 4


def test_benchmark_confirms_identical_itemsets(toy_basket):
    """If this ever fails, one of the two miners has a bug."""
    result = benchmark_algorithms(toy_basket, support_grid=[0.3, 0.2, 0.1], max_len=3)
    assert result.verify_agreement()


def test_benchmark_summary_and_pivot(toy_basket):
    result = benchmark_algorithms(toy_basket, support_grid=[0.3, 0.2], max_len=2)
    assert "apriori" in result.pivot("runtime_sec").columns
    assert "Apriori vs FP-Growth" in result.summary()


def test_scaling_curve_grows_with_volume(toy_basket):
    curve = scaling_curve(toy_basket, fractions=(0.5, 1.0), min_support=0.2, max_len=2)
    assert len(curve) == 2
    assert curve["n_transactions"].is_monotonic_increasing


# ---------------------------------------------------------------- Spark
def test_spark_module_imports_without_pyspark():
    """The Spark bridge must never break an install that has no JVM."""
    from mba.scaling import spark_fpgrowth

    assert isinstance(spark_fpgrowth.spark_available(), bool)
    assert "PFP" in spark_fpgrowth.ARCHITECTURE_NOTES


def test_spark_raises_a_helpful_error_when_absent():
    from mba.scaling.spark_fpgrowth import run_spark_fpgrowth, spark_available

    if spark_available():
        pytest.skip("pyspark is installed; the fallback path cannot be exercised")
    with pytest.raises(ImportError, match="PySpark is not installed"):
        run_spark_fpgrowth(transactions=[["a", "b"]])
