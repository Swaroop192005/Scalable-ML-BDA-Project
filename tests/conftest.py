"""Shared fixtures.

The suite runs entirely on small synthetic data so it is fast and
deterministic, and so it passes on a fresh clone with no dataset downloaded.
Tests that need the real file are marked ``slow`` and skip when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mba.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """The real project configuration."""
    return load_config()


@pytest.fixture
def toy_transactions() -> list[list[str]]:
    """Ten baskets with a planted association.

    ``bread`` and ``butter`` co-occur in 5 of 10 baskets; ``milk`` is in 7 of
    10, so a bread -> milk rule has high confidence but low lift, while
    bread -> butter has both. That contrast is what several tests assert on.
    """
    return [
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


@pytest.fixture
def toy_frame(toy_transactions) -> pd.DataFrame:
    """The toy baskets as a line-level transaction frame."""
    rows = []
    for index, basket in enumerate(toy_transactions):
        for item in basket:
            rows.append(
                {
                    "InvoiceNo": f"{10000 + index}",
                    "StockCode": item[:3].upper(),
                    "Description": item.upper(),
                    "Quantity": 2,
                    "InvoiceDate": pd.Timestamp("2011-01-01") + pd.Timedelta(days=index),
                    "UnitPrice": 1.5,
                    "CustomerID": 1000.0 + (index % 3),
                    "Country": "United Kingdom" if index % 3 else "France",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def messy_frame(toy_frame) -> pd.DataFrame:
    """The toy frame with every defect the cleaner is supposed to remove."""
    extra = pd.DataFrame(
        [
            # A cancellation (negative quantity, 'C' prefix).
            {"InvoiceNo": "C20001", "StockCode": "BRE", "Description": "BREAD",
             "Quantity": -2, "InvoiceDate": pd.Timestamp("2011-02-01"),
             "UnitPrice": 1.5, "CustomerID": 1001.0, "Country": "United Kingdom"},
            # Postage: a non-product service line.
            {"InvoiceNo": "20002", "StockCode": "POST", "Description": "POSTAGE",
             "Quantity": 1, "InvoiceDate": pd.Timestamp("2011-02-02"),
             "UnitPrice": 18.0, "CustomerID": 1002.0, "Country": "United Kingdom"},
            # Missing description.
            {"InvoiceNo": "20003", "StockCode": "XXX", "Description": None,
             "Quantity": 1, "InvoiceDate": pd.Timestamp("2011-02-03"),
             "UnitPrice": 2.0, "CustomerID": 1003.0, "Country": "United Kingdom"},
            # Zero price adjustment line.
            {"InvoiceNo": "20004", "StockCode": "BRE", "Description": "BREAD",
             "Quantity": 1, "InvoiceDate": pd.Timestamp("2011-02-04"),
             "UnitPrice": 0.0, "CustomerID": 1004.0, "Country": "United Kingdom"},
            # Whitespace / case / trailing-punctuation variants of one product.
            {"InvoiceNo": "20005", "StockCode": "BRE", "Description": "  bread  ",
             "Quantity": 1, "InvoiceDate": pd.Timestamp("2011-02-05"),
             "UnitPrice": 1.5, "CustomerID": 1005.0, "Country": "United Kingdom"},
            {"InvoiceNo": "20005", "StockCode": "BUT", "Description": "BUTTER.",
             "Quantity": 1, "InvoiceDate": pd.Timestamp("2011-02-05"),
             "UnitPrice": 1.5, "CustomerID": 1005.0, "Country": "United Kingdom"},
            # Exact duplicate basket line.
            {"InvoiceNo": "20005", "StockCode": "BRE", "Description": "BREAD",
             "Quantity": 1, "InvoiceDate": pd.Timestamp("2011-02-05"),
             "UnitPrice": 1.5, "CustomerID": 1005.0, "Country": "United Kingdom"},
        ]
    )
    return pd.concat([toy_frame, extra], ignore_index=True)


@pytest.fixture
def toy_basket(toy_transactions):
    from mba.data.basket import baskets_from_transactions

    return baskets_from_transactions(toy_transactions)


@pytest.fixture(scope="session")
def real_data_path(cfg) -> Path | None:
    """Path to the *full* dataset, or ``None`` when it has not been downloaded.

    ``resolve_source`` deliberately falls back to the committed 2,500-invoice
    sample so the pipeline runs on a fresh clone. The tests in
    ``test_real_dataset.py`` assert properties of the full 541,909-row file,
    so the sample satisfying the lookup is exactly wrong: those tests would
    run against it and *fail* the shape assertions rather than skip.

    Anything resolving inside the sample directory is therefore reported as
    "not downloaded", which is what the skip in those tests keys on.
    """
    try:
        from mba.data.loader import resolve_source

        source = resolve_source(cfg)
    except FileNotFoundError:
        return None

    sample_dir = cfg.path("paths.sample_dir").resolve()
    resolved = source.resolve()
    if resolved == sample_dir or sample_dir in resolved.parents:
        return None
    return source


def brute_force_itemsets(
    transactions: list[list[str]], min_support: float, max_len: int
) -> dict[frozenset, int]:
    """Exhaustive reference implementation, used to verify both miners.

    Deliberately the slowest possible correct algorithm: enumerate every
    combination and count it. If Apriori or FP-Growth disagrees with this,
    the optimised one is wrong.
    """
    import math
    from itertools import combinations

    n = len(transactions)
    threshold = math.ceil(min_support * n)
    items = sorted({i for t in transactions for i in t})
    sets = [set(t) for t in transactions]

    out: dict[frozenset, int] = {}
    for size in range(1, max_len + 1):
        for combo in combinations(items, size):
            candidate = set(combo)
            count = sum(1 for basket in sets if candidate <= basket)
            if count >= threshold:
                out[frozenset(combo)] = count
    return out
