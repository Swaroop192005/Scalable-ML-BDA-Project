"""Scalable Market Basket Analysis.

A modular, end-to-end platform for mining association rules out of retail
transaction data and turning them into cross-selling decisions.

The package is deliberately layered so each stage can be used on its own:

    mba.data       loading, cleaning and sparse basket encoding
    mba.mining     Apriori, FP-Growth, rule generation, spurious-rule triage
    mba.scaling    benchmarking, out-of-core mining, the Spark reference
    mba.analytics  EDA, co-occurrence networks, regional splits, insights
    mba.viz        the plotting layer shared by reports and the dashboard
"""

__version__ = "1.0.0"
__all__ = ["__version__", "load_config", "Config"]


def __getattr__(name):  # pragma: no cover - thin lazy re-export
    if name in ("load_config", "Config"):
        from mba.config import Config, load_config

        return {"load_config": load_config, "Config": Config}[name]
    raise AttributeError(f"module 'mba' has no attribute {name!r}")
