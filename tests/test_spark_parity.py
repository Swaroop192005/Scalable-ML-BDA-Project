"""Cross-validation against Spark's distributed FP-Growth (PFP).

``test_scaling.py`` proves the Spark bridge degrades gracefully when PySpark is
absent. This file proves the opposite half: that when a JVM *is* available, the
distributed implementation and our single-node one agree on what they find.

That is a stronger claim than it sounds. Spark's ``FPGrowth`` is an independent
implementation of a different algorithm (PFP partitions conditional pattern
bases across executors), written by other people in Scala. Agreement means our
support counting, our ``max_len`` semantics and our rule generation match a
production system, not just our own brute-force checker.

Two differences are expected and asserted rather than papered over:

* **Transaction count.** Spark mines the raw transaction stream; the in-memory
  encoder first drops items appearing in fewer than ``min_item_support_count``
  baskets, which leaves a few baskets with a single item and removes those too.
  A stream cannot do that pruning — it has not seen the end of the data — so
  the two paths legitimately see slightly different denominators.
* **Consequent arity.** ``pyspark.ml.fpm.FPGrowth.associationRules`` emits
  single-item consequents only. Ours generates every non-empty proper subset,
  so it finds strictly more rules. The comparison is therefore restricted to
  the single-item subset.
* **Itemset length.** Spark's ``FPGrowth`` has no length cap, while ours
  defaults to ``max_len=3``. These tests mine uncapped on both sides. The cap
  does not change the production comparison — no four-item set reaches the
  configured 2% support on the real dataset — but this fixture has one.

Skipped entirely when PySpark is absent, so it is a bonus check rather than a
dependency. Note that Spark 4 requires Java 17 or 21.
"""

import pytest

pytest.importorskip("pyspark", reason="pyspark is an optional extra")

from mba.data.basket import baskets_from_transactions
from mba.mining.fpgrowth import fpgrowth
from mba.mining.rules import association_rules
from mba.scaling.spark_fpgrowth import run_spark_fpgrowth, spark_available

pytestmark = pytest.mark.skipif(
    not spark_available(), reason="pyspark imports but no working JVM"
)

MIN_SUPPORT = 0.04
MIN_CONFIDENCE = 0.30


@pytest.fixture(scope="module")
def spark_session():
    """One session for the whole module — JVM startup dominates the runtime."""
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("mba-parity")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def transactions():
    """The conftest toy baskets, repeated.

    Defined here rather than reused from ``conftest`` because that fixture is
    function-scoped and these must be module-scoped — a JVM per test would
    dominate the suite's runtime. The repetition gives the 4% floor enough
    baskets to admit a non-trivial answer, including a three-item set, which
    is what makes the multi-item-consequent check below meaningful.
    """
    base = [
        ["bread", "butter", "milk"],
        ["bread", "butter", "milk"],
        ["bread", "butter"],
        ["bread", "butter", "jam"],
        ["bread", "butter", "milk", "jam"],
        ["bread", "milk"],
        ["milk", "eggs"],
        ["milk", "eggs", "cheese"],
        ["eggs", "cheese"],
        ["milk", "cheese"],
    ]
    return base * 6


@pytest.fixture(scope="module")
def spark_result(spark_session, transactions):
    return run_spark_fpgrowth(
        transactions=transactions,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        session=spark_session,
        stop_session=False,
    )


@pytest.fixture(scope="module")
def local_result(transactions):
    basket = baskets_from_transactions(transactions)
    # Uncapped, to match Spark's FPGrowth, which has no length limit.
    return fpgrowth(basket, min_support=MIN_SUPPORT, max_len=None), basket


def _itemsets(frame):
    column = "count" if "count" in frame.columns else frame.columns[1]
    return {frozenset(items): int(n) for items, n in zip(frame["itemsets"], frame[column])}


def test_spark_finds_the_same_frequent_itemsets(spark_result, local_result):
    local, _ = local_result
    spark_sets = _itemsets(spark_result["itemsets"])
    local_sets = _itemsets(local.itemsets)

    assert spark_sets, "Spark returned nothing — the fixture is too sparse to be a test"
    assert set(spark_sets) == set(local_sets)


def test_support_counts_agree_on_every_shared_itemset(spark_result, local_result):
    """Counts are absolute, so they must match exactly even though the two
    paths divide by slightly different denominators."""
    local, _ = local_result
    spark_sets = _itemsets(spark_result["itemsets"])
    local_sets = _itemsets(local.itemsets)

    mismatched = {
        tuple(sorted(s)): (local_sets[s], spark_sets[s])
        for s in set(spark_sets) & set(local_sets)
        if spark_sets[s] != local_sets[s]
    }
    assert not mismatched, f"support counts diverge: {mismatched}"


def test_single_item_consequent_rules_are_identical(spark_result, local_result):
    local, basket = local_result
    ours = association_rules(
        local.itemsets,
        min_confidence=MIN_CONFIDENCE,
        min_lift=0.0,  # Spark filters on confidence alone; match it.
        n_transactions=basket.matrix.shape[0],
    )
    ours_single = {
        (frozenset(a), frozenset(c))
        for a, c in zip(ours["antecedents"], ours["consequents"])
        if len(c) == 1
    }
    spark_rules = {
        (frozenset(r.antecedents), frozenset(r.consequents))
        for r in spark_result["rules"].itertuples()
    }

    assert spark_rules == ours_single


def test_we_find_rules_spark_cannot_express(spark_result, local_result):
    """Not a defect on either side — a documented difference in scope.

    Spark emits single-item consequents only. If this ever stops holding,
    the restriction in the test above has become unnecessary and should go.
    """
    local, basket = local_result
    ours = association_rules(
        local.itemsets, min_confidence=MIN_CONFIDENCE, min_lift=0.0,
        n_transactions=basket.matrix.shape[0],
    )
    assert all(len(c) == 1 for c in spark_result["rules"]["consequents"])
    assert any(len(c) > 1 for c in ours["consequents"]), (
        "fixture produces no multi-item consequents, so this asserts nothing"
    )
