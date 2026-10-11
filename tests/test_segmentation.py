"""RFM segmentation, k-means and per-segment rules."""

import numpy as np
import pandas as pd
import pytest

from mba.analytics.segmentation import (
    choose_k,
    compare_segment_rules,
    compute_rfm,
    kmeans,
    kmeans_plusplus_init,
    segment_customers,
    silhouette_score,
)


@pytest.fixture
def blobs():
    """Three well-separated Gaussian blobs with known membership."""
    rng = np.random.default_rng(0)
    points = np.vstack(
        [rng.normal(centre, 0.3, size=(80, 2)) for centre in ([0, 0], [10, 10], [0, 10])]
    )
    truth = np.repeat([0, 1, 2], 80)
    return points, truth


@pytest.fixture
def customer_frame():
    """Synthetic customers in three deliberately distinct RFM regimes."""
    rows = []
    base = pd.Timestamp("2011-01-01")
    configurations = [
        # (n_customers, n_orders, days_ago, unit_price)
        (40, 12, 5, 20.0),    # champions: frequent, recent, high value
        (40, 3, 60, 8.0),     # mid
        (40, 1, 300, 3.0),    # one-time, long lapsed
    ]
    customer_id = 0
    for n_customers, n_orders, days_ago, price in configurations:
        for _ in range(n_customers):
            customer_id += 1
            for order in range(n_orders):
                invoice = f"INV{customer_id:04d}-{order}"
                for item in ("ALPHA MUG", "BETA BOWL", "GAMMA PLATE"):
                    rows.append(
                        {
                            "InvoiceNo": invoice,
                            "StockCode": item[:5],
                            "Description": item,
                            "Quantity": 2,
                            "InvoiceDate": base
                            + pd.Timedelta(days=360 - days_ago - order),
                            "UnitPrice": price,
                            "CustomerID": float(customer_id),
                            "Country": "United Kingdom",
                        }
                    )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- k-means
def test_kmeans_recovers_separated_blobs(blobs):
    points, truth = blobs
    labels, centroids, inertia = kmeans(points, k=3, seed=1)
    assert centroids.shape == (3, 2)
    assert inertia > 0
    # Each true blob must map onto exactly one predicted cluster.
    for blob in range(3):
        assert len(set(labels[truth == blob].tolist())) == 1
    # ...and the three must be distinct clusters.
    assert len({labels[truth == b][0] for b in range(3)}) == 3


def test_kmeans_is_deterministic_for_a_seed(blobs):
    points, _ = blobs
    a, _, ia = kmeans(points, k=3, seed=7)
    b, _, ib = kmeans(points, k=3, seed=7)
    assert np.array_equal(a, b)
    assert ia == pytest.approx(ib)


def test_inertia_falls_as_k_rises(blobs):
    points, _ = blobs
    inertias = [kmeans(points, k=k, seed=3)[2] for k in (1, 2, 3, 4)]
    assert inertias == sorted(inertias, reverse=True)


def test_kmeans_plusplus_spreads_its_seeds(blobs):
    points, _ = blobs
    rng = np.random.default_rng(0)
    centres = kmeans_plusplus_init(points, 3, rng)
    assert centres.shape == (3, 2)
    # Seeds should not collapse onto one another.
    distances = [
        np.linalg.norm(centres[i] - centres[j]) for i in range(3) for j in range(i + 1, 3)
    ]
    assert min(distances) > 1.0


def test_kmeans_handles_k_larger_than_n():
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels, centroids, _ = kmeans(points, k=10, seed=0)
    assert len(centroids) <= len(points)
    assert len(labels) == 2


def test_kmeans_on_empty_input():
    labels, centroids, inertia = kmeans(np.empty((0, 3)), k=3)
    assert len(labels) == 0 and inertia == 0.0


def test_silhouette_rewards_separation(blobs):
    points, truth = blobs
    good = silhouette_score(points, truth)
    rng = np.random.default_rng(1)
    scrambled = rng.permutation(truth)
    assert good > 0.7
    assert good > silhouette_score(points, scrambled)


def test_silhouette_of_a_single_cluster_is_zero(blobs):
    points, _ = blobs
    assert silhouette_score(points, np.zeros(len(points), dtype=int)) == 0.0


# ----------------------------------------------------------------- RFM
def test_rfm_columns_and_shape(customer_frame, cfg):
    rfm = compute_rfm(customer_frame, cfg)
    assert len(rfm) == customer_frame["CustomerID"].nunique()
    for column in ("customer", "recency", "frequency", "monetary", "avg_basket_value"):
        assert column in rfm.columns


def test_rfm_values_are_correct(customer_frame, cfg):
    rfm = compute_rfm(customer_frame, cfg)
    assert (rfm["recency"] >= 1).all(), "recency must stay positive for the log"
    assert (rfm["frequency"] >= 1).all()
    assert (rfm["monetary"] > 0).all()
    # Frequency must equal the customer's distinct invoice count.
    expected = customer_frame.groupby("CustomerID")["InvoiceNo"].nunique()
    merged = rfm.set_index("customer")["frequency"]
    assert (merged.sort_index().to_numpy() == expected.sort_index().to_numpy()).all()


def test_rfm_without_customers_returns_empty(cfg):
    frame = pd.DataFrame(
        {
            "InvoiceNo": ["1"], "Description": ["X"], "Quantity": [1],
            "InvoiceDate": [pd.Timestamp("2011-01-01")], "UnitPrice": [1.0],
            "CustomerID": [np.nan], "Country": ["UK"], "StockCode": ["X"],
        }
    )
    assert compute_rfm(frame, cfg).empty


# -------------------------------------------------------- segmentation
def test_segments_are_formed_and_named(customer_frame, cfg):
    result = segment_customers(customer_frame, cfg, k=3, mine_rules=False)
    assert result.k == 3
    assert len(result.profiles) == 3
    assert result.profiles["segment"].nunique() == 3, "segment names must be unique"
    assert result.customers["segment"].notna().all()


def test_every_customer_lands_in_exactly_one_segment(customer_frame, cfg):
    result = segment_customers(customer_frame, cfg, k=3, mine_rules=False)
    assert len(result.customers) == customer_frame["CustomerID"].nunique()
    assert result.profiles["n_customers"].sum() == len(result.customers)


def test_shares_sum_to_one(customer_frame, cfg):
    result = segment_customers(customer_frame, cfg, k=3, mine_rules=False)
    assert result.profiles["revenue_share"].sum() == pytest.approx(1.0)
    assert result.profiles["customer_share"].sum() == pytest.approx(1.0)


def test_distinct_regimes_separate(customer_frame, cfg):
    """The three planted RFM regimes should not be merged."""
    result = segment_customers(customer_frame, cfg, k=3, mine_rules=False)
    frequencies = sorted(result.profiles["frequency"].tolist())
    assert frequencies[0] < frequencies[-1], "segments have identical frequency"


def test_per_segment_rules_are_tagged(customer_frame, cfg):
    result = segment_customers(
        customer_frame, cfg, k=3, mine_rules=True, min_support=0.1,
        min_confidence=0.1, min_segment_size=10,
    )
    for segment, rules in result.rules.items():
        if not rules.empty:
            assert (rules["segment"] == segment).all()


def test_choose_k_reports_both_criteria(customer_frame, cfg):
    table = choose_k(customer_frame, cfg, k_values=(2, 3, 4))
    assert list(table["k"]) == [2, 3, 4]
    assert table["inertia"].is_monotonic_decreasing
    assert table["silhouette"].between(-1, 1).all()


def test_compare_segment_rules_flattens(customer_frame, cfg):
    result = segment_customers(
        customer_frame, cfg, k=3, mine_rules=True, min_support=0.1,
        min_confidence=0.1, min_segment_size=10,
    )
    comparison = compare_segment_rules(result, top_n=3)
    if not comparison.empty:
        assert {"segment", "antecedents", "consequents", "lift"} <= set(comparison.columns)
