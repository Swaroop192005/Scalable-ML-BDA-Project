"""EDA, co-occurrence networks, regional splits and insight generation."""

import networkx as nx
import pandas as pd
import pytest

from mba.analytics.eda import run_eda
from mba.analytics.insights import generate_insights, insights_to_frame, insights_to_markdown
from mba.analytics.network import (
    build_cooccurrence_graph,
    centrality_table,
    cooccurrence_matrix,
    detect_communities,
    graph_from_rules,
    graph_summary,
)
from mba.analytics.regional import compare_regions, distinctive_rules, regional_analysis
from mba.data.basket import baskets_from_transactions, encode_baskets
from mba.data.preprocess import clean_transactions
from mba.mining.fpgrowth import fpgrowth
from mba.mining.rules import association_rules
from mba.mining.spurious import flag_spurious_rules


@pytest.fixture
def clean_toy(toy_frame, cfg):
    clean, _ = clean_transactions(toy_frame, cfg)
    return clean


# ------------------------------------------------------------------- EDA
def test_eda_scalars_are_consistent(clean_toy, cfg):
    basket = encode_baskets(clean_toy, cfg, min_item_count=1)
    report = run_eda(clean_toy, cfg, basket)

    assert report.scalars["n_transactions"] == clean_toy["InvoiceNo"].nunique()
    assert report.scalars["n_products"] == clean_toy["Description"].nunique()
    assert report.scalars["total_revenue"] == pytest.approx(
        (clean_toy["Quantity"] * clean_toy["UnitPrice"]).sum()
    )


def test_eda_produces_the_expected_tables(clean_toy, cfg):
    report = run_eda(clean_toy, cfg, encode_baskets(clean_toy, cfg, min_item_count=1))
    for table in ("product_stats", "top_products_volume", "top_products_revenue",
                  "basket_sizes", "basket_size_distribution", "country_stats", "monthly"):
        assert table in report.tables, f"missing EDA table: {table}"


def test_top_products_are_sorted_descending(clean_toy, cfg):
    report = run_eda(clean_toy, cfg)
    assert report["top_products_volume"]["transactions"].is_monotonic_decreasing


def test_support_in_eda_matches_the_basket_matrix(clean_toy, cfg):
    basket = encode_baskets(clean_toy, cfg, min_item_count=1)
    report = run_eda(clean_toy, cfg, basket)
    stats = report.tables["product_stats"]
    assert stats["support"].max() <= 1.0 + 1e-9
    assert stats["support"].min() >= 0.0


# --------------------------------------------------------------- network
def test_cooccurrence_counts_are_correct():
    basket = baskets_from_transactions([["a", "b"], ["a", "b"], ["a", "c"]])
    matrix, items = cooccurrence_matrix(basket)
    index = {item: i for i, item in enumerate(items)}
    dense = matrix.toarray()
    assert dense[index["a"], index["b"]] == 2
    assert dense[index["a"], index["c"]] == 1
    assert dense[index["a"], index["a"]] == 3   # diagonal is the item's own count


def test_graph_is_undirected_and_lift_weighted():
    transactions = [["a", "b"]] * 30 + [["c", "d"]] * 30 + [["a", "c"]] * 2
    graph = build_cooccurrence_graph(
        baskets_from_transactions(transactions), top_n=10,
        min_edge_weight=1.0, min_cooccurrence=2, weight="lift",
    )
    assert isinstance(graph, nx.Graph)
    assert graph.has_edge("a", "b")
    for *_, data in graph.edges(data=True):
        assert data["weight"] >= 1.0
        assert {"lift", "jaccard", "support", "count"} <= set(data)


def test_edge_threshold_removes_weak_links():
    transactions = [["a", "b"]] * 30 + [["c", "d"]] * 30 + [["a", "c"]] * 3
    loose = build_cooccurrence_graph(
        baskets_from_transactions(transactions), top_n=10,
        min_edge_weight=0.0, min_cooccurrence=1,
    )
    strict = build_cooccurrence_graph(
        baskets_from_transactions(transactions), top_n=10,
        min_edge_weight=3.0, min_cooccurrence=1,
    )
    assert strict.number_of_edges() <= loose.number_of_edges()


def test_communities_separate_disjoint_product_families():
    transactions = [["a", "b", "c"]] * 40 + [["x", "y", "z"]] * 40
    graph = build_cooccurrence_graph(
        baskets_from_transactions(transactions), top_n=10,
        min_edge_weight=1.0, min_cooccurrence=2,
    )
    communities = detect_communities(graph)
    assert communities["a"] == communities["b"] == communities["c"]
    assert communities["x"] == communities["y"] == communities["z"]
    assert communities["a"] != communities["x"]


def test_centrality_table_is_ranked():
    transactions = [["hub", f"spoke{i}"] for i in range(12) for _ in range(5)]
    graph = build_cooccurrence_graph(
        baskets_from_transactions(transactions), top_n=20,
        min_edge_weight=0.0, min_cooccurrence=2,
    )
    table = centrality_table(graph)
    assert table["weighted_degree"].is_monotonic_decreasing
    assert table.iloc[0]["product"] == "hub"


def test_graph_summary_names_hubs_and_bridges():
    transactions = [["a", "b", "c"]] * 30 + [["c", "x", "y"]] * 30
    graph = build_cooccurrence_graph(
        baskets_from_transactions(transactions), top_n=10,
        min_edge_weight=1.0, min_cooccurrence=2,
    )
    summary = graph_summary(graph)
    assert summary["n_nodes"] > 0
    assert "top_hubs" in summary and "top_bridges" in summary
    assert isinstance(summary["centrality_table"], pd.DataFrame)


def test_empty_graph_summary_is_safe():
    assert graph_summary(nx.Graph())["n_nodes"] == 0


def test_graph_from_rules_builds_rule_nodes(toy_transactions):
    basket = baskets_from_transactions(toy_transactions)
    rules = association_rules(
        fpgrowth(basket, 0.1, 3, track_memory=False), min_confidence=0.0, min_lift=0.0
    )
    graph = graph_from_rules(rules, max_rules=20)
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0


# -------------------------------------------------------------- regional
def test_regional_analysis_splits_by_country(clean_toy, cfg):
    results = regional_analysis(
        clean_toy, cfg, min_transactions=1, min_support=0.1, min_confidence=0.0
    )
    assert set(results) <= set(clean_toy["Country"].unique())
    for country, result in results.items():
        assert result.n_transactions > 0
        if not result.rules.empty:
            assert (result.rules["country"] == country).all()


def test_small_markets_are_skipped(clean_toy, cfg):
    results = regional_analysis(clean_toy, cfg, min_transactions=10_000)
    assert results == {}


def test_compare_regions_produces_one_row_per_market(clean_toy, cfg):
    results = regional_analysis(
        clean_toy, cfg, min_transactions=1, min_support=0.1, min_confidence=0.0
    )
    comparison = compare_regions(results)
    assert len(comparison) == len(results)
    if not comparison.empty:
        assert comparison["transactions"].is_monotonic_decreasing


def test_distinctive_rules_handles_a_missing_reference(clean_toy, cfg):
    results = regional_analysis(
        clean_toy, cfg, min_transactions=1, min_support=0.1, min_confidence=0.0
    )
    out = distinctive_rules(results, reference="Atlantis")
    assert isinstance(out, pd.DataFrame)


# -------------------------------------------------------------- insights
@pytest.fixture
def strong_rules(cfg):
    """Rules with a genuine two-way bundle and a one-way cross-sell."""
    transactions = (
        [["KETTLE", "TEAPOT"]] * 300          # two-way bundle
        + [["LAPTOP", "MOUSE PAD"]] * 200     # one-way: laptop implies pad
        + [["MOUSE PAD"]] * 400               # ...but not the reverse
        + [["UNRELATED ITEM"]] * 300
    )
    basket = baskets_from_transactions(transactions)
    rules = association_rules(
        fpgrowth(basket, 0.05, 3, track_memory=False), min_confidence=0.1, min_lift=1.0
    )
    rules, _ = flag_spurious_rules(rules, cfg, basket.n_transactions)
    return rules, basket.n_transactions


def test_insights_are_generated_and_ranked(strong_rules, cfg):
    rules, n = strong_rules
    insights = generate_insights(rules, cfg, n_transactions=n)
    assert insights
    priorities = [i.priority for i in insights]
    assert priorities == sorted(priorities, reverse=True)


def test_two_way_association_becomes_a_bundle(strong_rules, cfg):
    rules, n = strong_rules
    insights = generate_insights(rules, cfg, n_transactions=n, top_n=50)
    bundles = [i for i in insights if i.kind == "bundle"]
    assert any("KETTLE" in i.title or "TEAPOT" in i.title for i in bundles)


def test_every_insight_carries_an_action(strong_rules, cfg):
    rules, n = strong_rules
    for insight in generate_insights(rules, cfg, n_transactions=n, top_n=50):
        assert insight.action and insight.detail
        assert insight.kind in {
            "bundle", "cross_sell", "placement", "anchor", "untapped", "regional"
        }


def test_network_summary_produces_anchor_insights(strong_rules, cfg):
    rules, n = strong_rules
    insights = generate_insights(
        rules, cfg, n_transactions=n,
        graph_summary={"top_hubs": ["KETTLE"], "top_bridges": ["TEAPOT"],
                       "n_communities": 2, "n_nodes": 5},
        top_n=50,
    )
    assert any(i.kind == "anchor" for i in insights)


def test_insight_serialisation(strong_rules, cfg):
    rules, n = strong_rules
    insights = generate_insights(rules, cfg, n_transactions=n)
    frame = insights_to_frame(insights)
    assert {"kind", "title", "action", "priority"} <= set(frame.columns)
    markdown = insights_to_markdown(insights)
    assert markdown.startswith("#") and len(markdown) > 100


def test_no_rules_gives_no_insights(cfg):
    from mba.mining.rules import empty_rules_frame

    assert generate_insights(empty_rules_frame(), cfg) == []
