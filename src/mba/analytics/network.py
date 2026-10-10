"""Product co-occurrence networks.

A rules table is a list; a network is a *map*. Modelling products as nodes
and associations as edges exposes structure no sorted table shows:

* **Communities** are naturally-occurring product families — the cluster
  that forms a themed range, discovered from behaviour rather than from the
  merchandising hierarchy.
* **Degree** identifies hub products that pull many other items into the
  basket; they belong at the entrance and in every cross-sell slot.
* **Betweenness** identifies bridges between otherwise separate communities.
  These are the products that move a shopper from one category into another,
  which is exactly what a cross-category campaign needs.

Edges are weighted by **lift**, not raw co-occurrence, so two popular items
are not joined merely for both being popular.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from mba.config import Config
from mba.data.basket import BasketMatrix

log = logging.getLogger(__name__)


def cooccurrence_matrix(basket: BasketMatrix, top_n: int | None = None):
    """Item x item co-occurrence counts as a sparse matrix.

    ``B.T @ B`` gives, for every pair, the number of baskets containing both;
    the diagonal holds each item's own basket count. The multiply happens in
    scipy's compiled sparse kernel, never densified.
    """
    matrix = basket.matrix
    items = basket.items

    if top_n is not None and top_n < basket.n_items:
        keep = np.argsort(basket.support_counts())[::-1][:top_n]
        keep.sort()
        matrix = matrix[:, keep]
        items = items[keep]

    binary = matrix.astype(np.int32)
    co = (binary.T @ binary).tocsr()
    return co, items


def build_cooccurrence_graph(
    basket: BasketMatrix,
    cfg: Config | None = None,
    top_n: int | None = None,
    min_edge_weight: float | None = None,
    min_cooccurrence: int = 5,
    weight: str = "lift",
) -> nx.Graph:
    """Build an undirected co-occurrence graph over the most frequent items.

    Parameters
    ----------
    top_n:
        Number of items (by basket count) to include as nodes. The full
        catalogue would give a hairball; 30-50 is readable.
    min_edge_weight:
        Edges below this weight are dropped. With ``weight="lift"`` a value
        of 1.0 keeps only pairs that co-occur more than chance predicts.
    weight:
        ``"lift"`` (default), ``"jaccard"``, ``"support"`` or ``"count"``.
    """
    if cfg is not None:
        top_n = top_n if top_n is not None else int(cfg.get("analytics.network.top_n_items", 40))
        if min_edge_weight is None:
            min_edge_weight = float(cfg.get("analytics.network.min_edge_weight", 0.01))
    top_n = top_n or 40
    min_edge_weight = 1.0 if min_edge_weight is None else min_edge_weight

    co, items = cooccurrence_matrix(basket, top_n=top_n)
    n_tx = basket.n_transactions
    diagonal = np.asarray(co.diagonal()).ravel()

    graph = nx.Graph()
    for index, item in enumerate(items):
        count = int(diagonal[index])
        graph.add_node(
            str(item),
            count=count,
            support=count / n_tx if n_tx else 0.0,
        )

    coo = co.tocoo()
    for i, j, pair_count in zip(coo.row, coo.col, coo.data):
        if i >= j:
            continue  # symmetric; keep the upper triangle only
        pair_count = int(pair_count)
        if pair_count < min_cooccurrence:
            continue

        s_i, s_j = diagonal[i] / n_tx, diagonal[j] / n_tx
        s_ij = pair_count / n_tx
        lift = s_ij / (s_i * s_j) if s_i and s_j else 0.0
        union = diagonal[i] + diagonal[j] - pair_count
        jaccard = pair_count / union if union else 0.0

        value = {
            "lift": lift,
            "jaccard": jaccard,
            "support": s_ij,
            "count": float(pair_count),
        }[weight]
        if value < min_edge_weight:
            continue

        graph.add_edge(
            str(items[i]),
            str(items[j]),
            weight=float(value),
            lift=float(lift),
            jaccard=float(jaccard),
            support=float(s_ij),
            count=pair_count,
        )

    graph.remove_nodes_from(list(nx.isolates(graph)))
    log.info(
        "co-occurrence graph: %d nodes, %d edges (weight=%s, min=%.3g)",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        weight,
        min_edge_weight,
    )
    return graph


def graph_from_rules(rules: pd.DataFrame, max_rules: int = 150) -> nx.DiGraph:
    """Directed graph of ``A => C`` rules, for the rule-network visualisation.

    Multi-item antecedents become a small square 'rule node' with edges in
    from each antecedent and out to each consequent — the standard way to
    draw a hypergraph without losing the grouping.
    """
    graph = nx.DiGraph()
    if rules.empty:
        return graph

    for index, row in rules.head(max_rules).iterrows():
        antecedents = sorted(row["antecedents"])
        consequents = sorted(row["consequents"])
        attrs = {
            "lift": float(row.get("lift", 0.0)),
            "confidence": float(row.get("confidence", 0.0)),
            "support": float(row.get("support", 0.0)),
        }
        if len(antecedents) == 1 and len(consequents) == 1:
            graph.add_node(antecedents[0], kind="item")
            graph.add_node(consequents[0], kind="item")
            graph.add_edge(antecedents[0], consequents[0], **attrs)
        else:
            rule_node = f"R{index}"
            graph.add_node(rule_node, kind="rule", **attrs)
            for a in antecedents:
                graph.add_node(a, kind="item")
                graph.add_edge(a, rule_node, **attrs)
            for c in consequents:
                graph.add_node(c, kind="item")
                graph.add_edge(rule_node, c, **attrs)
    return graph


def detect_communities(graph: nx.Graph, resolution: float = 1.0, seed: int = 42) -> dict[str, int]:
    """Partition the graph into product families with greedy modularity.

    Louvain is used where networkx provides it (better modularity on graphs
    this size); otherwise the greedy modularity algorithm, which is always
    available, is the fallback.
    """
    if graph.number_of_nodes() == 0:
        return {}
    try:
        communities = nx.community.louvain_communities(
            graph, weight="weight", resolution=resolution, seed=seed
        )
    except (AttributeError, ImportError):  # pragma: no cover - older networkx
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")

    return {
        node: community_id
        for community_id, members in enumerate(communities)
        for node in members
    }


def centrality_table(graph: nx.Graph) -> pd.DataFrame:
    """Per-product centrality measures, ranked by weighted degree."""
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=["product", "degree", "weighted_degree", "betweenness", "clustering", "community"]
        )

    degree = dict(graph.degree())
    weighted = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight=None, normalized=True)
    clustering = nx.clustering(graph, weight="weight")
    communities = detect_communities(graph)

    frame = pd.DataFrame(
        {
            "product": list(graph.nodes()),
            "degree": [degree[n] for n in graph.nodes()],
            "weighted_degree": [weighted[n] for n in graph.nodes()],
            "betweenness": [betweenness[n] for n in graph.nodes()],
            "clustering": [clustering[n] for n in graph.nodes()],
            "community": [communities.get(n, -1) for n in graph.nodes()],
            "support": [graph.nodes[n].get("support", 0.0) for n in graph.nodes()],
        }
    )
    return frame.sort_values("weighted_degree", ascending=False).reset_index(drop=True)


def graph_summary(graph: nx.Graph) -> dict[str, Any]:
    """Topology summary plus the named hubs, bridges and communities."""
    if graph.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0}

    table = centrality_table(graph)
    communities = detect_communities(graph)
    n_communities = len(set(communities.values())) if communities else 0

    members: dict[int, list[str]] = {}
    for node, community_id in communities.items():
        members.setdefault(community_id, []).append(node)

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "n_components": nx.number_connected_components(graph),
        "n_communities": n_communities,
        "avg_clustering": nx.average_clustering(graph, weight="weight"),
        "top_hubs": table.head(5)["product"].tolist(),
        "top_bridges": table.sort_values("betweenness", ascending=False)
        .head(5)["product"]
        .tolist(),
        "communities": {
            str(k): sorted(v, key=lambda n: -graph.nodes[n].get("support", 0))[:8]
            for k, v in sorted(members.items(), key=lambda kv: -len(kv[1]))
        },
        "centrality_table": table,
    }
