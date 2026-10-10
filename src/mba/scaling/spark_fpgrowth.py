"""Distributed FP-Growth: the reference architecture for terabyte-scale data.

Where this fits
---------------
The three mining paths in this project form a ladder, each taking over where
the previous one stops:

==================  ====================  ==============================================
Path                Practical ceiling     Bound by
==================  ====================  ==============================================
in-memory           ~10^5-10^6 baskets    the sparse matrix must fit in RAM
SON out-of-core     ~10^7 baskets         single-node CPU; data streams, compute does not
Spark FP-Growth     10^8+ baskets         cluster size
==================  ====================  ==============================================

How Spark parallelises FP-Growth
--------------------------------
Spark implements **PFP** (Parallel FP-Growth, Li et al., RecSys 2008), which
splits the problem by *item* rather than by row:

1. **Parallel counting.** A distributed ``reduceByKey`` produces the global
   1-item frequency list (the *F-list*).
2. **Grouping.** Frequent items are divided into *G* groups. Each group owns
   a disjoint slice of the pattern space.
3. **Parallel projection.** Every transaction is mapped to one conditional
   transaction per group it touches, and shuffled so that *all* the data a
   group needs lands on one executor.
4. **Local FP-Growth.** Each executor builds an FP-tree over its group's
   projected database and mines it with the sequential algorithm — the same
   algorithm as :mod:`mba.mining.fpgrowth`, just on a shard.
5. **Aggregation.** Results are unioned; no cross-partition merge is needed
   because the groups partition the pattern space by construction.

The win is that step 4 never needs the whole dataset on one machine, and the
shuffle in step 3 is the only global synchronisation point.

Running it
----------
PySpark and a JVM are optional extras — nothing else in this project imports
them. Install with ``pip install -r requirements-optional.txt`` and run::

    python -m mba.cli spark --min-support 0.02

Without PySpark the module still imports; :func:`spark_available` returns
``False`` and the CLI prints the architecture notes instead of failing.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import pandas as pd

from mba.config import Config

log = logging.getLogger(__name__)


def spark_available() -> bool:
    """True when PySpark can be imported in this interpreter."""
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


def build_spark_session(cfg: Config | None = None, **overrides: Any):
    """Create (or reuse) a tuned :class:`SparkSession`.

    The configuration below is what actually matters for FP-Growth at scale:

    ``spark.sql.shuffle.partitions``
        Step 3 above is a shuffle. The default of 200 is far too many for a
        laptop and far too few for a real cluster — size it to roughly
        2-3x total executor cores.
    ``spark.sql.execution.arrow.pyspark.enabled``
        Makes ``toPandas()`` on the mined rules a columnar transfer rather
        than a row-by-row pickle round-trip.
    ``spark.serializer`` (Kryo)
        FP-tree shards and projected transactions shuffle far more compactly
        than with Java serialisation.
    """
    from pyspark.sql import SparkSession

    app_name = overrides.get("app_name") or (cfg.get("scaling.spark.app_name") if cfg else None)
    master = overrides.get("master") or (cfg.get("scaling.spark.master") if cfg else None)
    partitions = overrides.get("shuffle_partitions") or (
        cfg.get("scaling.spark.shuffle_partitions") if cfg else None
    )

    builder = SparkSession.builder.appName(app_name or "mba-fpgrowth")
    if master:
        builder = builder.master(master)

    builder = (
        builder.config("spark.sql.shuffle.partitions", str(partitions or 8))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
    )
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    log.info("Spark session ready: %s", session.version)
    return session


def transactions_to_spark_df(session, transactions: Iterable[Sequence[str]], n_partitions: int = 8):
    """Build the ``(id, items: array<string>)`` DataFrame that PFP consumes."""
    rows = [(i, sorted(set(t))) for i, t in enumerate(transactions)]
    return session.createDataFrame(rows, ["id", "items"]).repartition(n_partitions)


def read_transactions_from_parquet(session, path: str, invoice_col: str, item_col: str):
    """Read a partitioned Parquet lake and collapse it into baskets.

    This is the production entry point: the line-level table stays on
    distributed storage and only the grouped arrays are materialised, so the
    driver never sees the full dataset.
    """
    from pyspark.sql import functions as F

    return (
        session.read.parquet(path)
        .select(invoice_col, item_col)
        .dropna()
        .groupBy(invoice_col)
        .agg(F.collect_set(item_col).alias("items"))
        .withColumnRenamed(invoice_col, "id")
    )


def run_spark_fpgrowth(
    transactions: Iterable[Sequence[str]] | None = None,
    cfg: Config | None = None,
    min_support: float | None = None,
    min_confidence: float | None = None,
    parquet_path: str | None = None,
    session=None,
    stop_session: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run distributed FP-Growth and return itemsets and rules as pandas frames.

    Either *transactions* or *parquet_path* must be supplied. The returned
    frames use the same column names as the single-node path, so the
    dashboard and report writers accept either source unchanged.

    Raises
    ------
    ImportError
        When PySpark is not installed — the caller is expected to fall back
        to :func:`mba.scaling.outofcore.mine_out_of_core`.
    """
    if not spark_available():
        raise ImportError(
            "PySpark is not installed. `pip install -r requirements-optional.txt` "
            "(a JVM is also required), or use the out-of-core miner instead."
        )

    from pyspark.ml.fpm import FPGrowth

    min_support = min_support if min_support is not None else (
        cfg.get("scaling.spark.min_support", 0.02) if cfg else 0.02
    )
    min_confidence = min_confidence if min_confidence is not None else (
        cfg.get("scaling.spark.min_confidence", 0.3) if cfg else 0.3
    )

    owns_session = session is None
    session = session or build_spark_session(cfg)

    try:
        if parquet_path:
            schema = cfg["schema"] if cfg else {}
            df = read_transactions_from_parquet(
                session,
                parquet_path,
                schema.get("invoice_col", "InvoiceNo"),
                schema.get("item_col", "Description"),
            )
        elif transactions is not None:
            partitions = int(cfg.get("scaling.spark.shuffle_partitions", 8)) if cfg else 8
            df = transactions_to_spark_df(session, transactions, partitions)
        else:
            raise ValueError("supply either `transactions` or `parquet_path`")

        df.cache()
        n_transactions = df.count()
        log.info("Spark FP-Growth over %s transactions", f"{n_transactions:,}")

        model = FPGrowth(
            itemsCol="items",
            minSupport=min_support,
            minConfidence=min_confidence,
        ).fit(df)

        itemsets = model.freqItemsets.toPandas()
        rules = model.associationRules.toPandas()

        # Normalise to this project's schema.
        if not itemsets.empty:
            itemsets = itemsets.rename(columns={"freq": "count", "items": "itemsets"})
            itemsets["itemsets"] = itemsets["itemsets"].map(frozenset)
            itemsets["support"] = itemsets["count"] / n_transactions
            itemsets["length"] = itemsets["itemsets"].map(len)

        if not rules.empty:
            rules = rules.rename(columns={"antecedent": "antecedents", "consequent": "consequents"})
            rules["antecedents"] = rules["antecedents"].map(frozenset)
            rules["consequents"] = rules["consequents"].map(frozenset)

        return {"itemsets": itemsets, "rules": rules, "n_transactions": n_transactions}
    finally:
        if owns_session and stop_session:
            session.stop()


ARCHITECTURE_NOTES = """\
Distributed FP-Growth (PFP) — reference architecture
=====================================================

  Object store (S3 / GCS / HDFS)
    transactions/           Parquet, partitioned by date and country
      year=2011/month=05/country=UK/part-*.parquet
           |
           v
  [1] Parallel counting        reduceByKey over items -> global F-list
           |
  [2] Grouping                 frequent items split into G disjoint groups
           |
  [3] Parallel projection      each transaction -> one conditional
      (the only shuffle)       transaction per group it touches
           |
  [4] Local FP-Growth          executor builds an FP-tree per group and
                               mines it sequentially
           |
  [5] Aggregation              union of per-group patterns; groups
                               partition the pattern space, so no merge
           |
           v
  Rules table (Delta / Parquet) -> dashboard, recommender, campaign tooling

Sizing guidance
---------------
* spark.sql.shuffle.partitions ~ 2-3x total executor cores. Step [3] is the
  only global shuffle and it dominates the wall clock.
* Partition the source by date and region. Most business questions are
  regional, and partition pruning then removes most of the scan.
* Raise minSupport before raising cluster size. Support is the exponent in
  this problem; hardware is only a constant factor.
* Persist the mined rules, not the itemsets. Rules are the small table and
  the one the serving layer actually queries.

Cost ladder
-----------
    in-memory   ~10^5-10^6 baskets   bounded by RAM for the sparse matrix
    SON         ~10^7 baskets        bounded by single-node CPU
    Spark PFP   10^8+ baskets        bounded by cluster size
"""
