"""FP-Growth: frequent patterns without candidate generation.

Implementation of Han, Pei & Yin (SIGMOD 2000). Two database passes, then
all the work happens inside a prefix tree:

**Pass 1** counts single items and discards the infrequent ones.

**Pass 2** inserts each transaction into an *FP-tree*: items sorted by
descending global frequency, so transactions sharing a prefix share a path.
A header table links every occurrence of an item across the tree.

**Mining** walks the header table from the *least* frequent item upwards. For
item ``i`` it collects the *conditional pattern base* — every prefix path
ending at ``i``, weighted by that path's count — builds a *conditional
FP-tree* from those prefixes and recurses. Patterns come out of the recursion
already frequent, so no candidate is ever generated and then discarded.

That is the structural difference from Apriori. Apriori's cost is dominated
by producing and testing candidates, which explodes as support drops.
FP-Growth pays a fixed tree-construction cost and then only ever touches
patterns that actually exist in the data — which is why the gap between the
two widens dramatically at low support (see ``mba bench``).

A compression shortcut matters in practice: when a conditional tree is a
single path, every subset of that path is frequent, so the implementation
enumerates them combinatorially instead of recursing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Iterable, Iterator, Sequence

import numpy as np

from mba.data.basket import BasketMatrix
from mba.mining.base import (
    MiningResult,
    MiningStats,
    _Timer,
    build_itemsets_frame,
    register_miner,
)

log = logging.getLogger(__name__)


class FPNode:
    """A single node of the FP-tree.

    ``__slots__`` matters here: the Online Retail tree holds hundreds of
    thousands of nodes, and slots cut per-node overhead by roughly 60%
    compared with a normal ``__dict__``-backed object.
    """

    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item: int | None, parent: "FPNode | None" = None) -> None:
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[int, FPNode] = {}
        self.link: FPNode | None = None  # next node for the same item

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"FPNode(item={self.item}, count={self.count}, children={len(self.children)})"

    def path_to_root(self) -> list[int]:
        """Items on the path from this node's parent up to (not incl.) the root."""
        path: list[int] = []
        node = self.parent
        while node is not None and node.item is not None:
            path.append(node.item)
            node = node.parent
        return path


class FPTree:
    """FP-tree with its header table.

    The header table maps each item to ``(total_count, first_node)``; nodes
    for the same item are chained through :attr:`FPNode.link`, which is what
    makes collecting a conditional pattern base an O(occurrences) walk rather
    than a tree traversal.
    """

    __slots__ = ("root", "headers", "item_counts", "n_nodes")

    def __init__(self) -> None:
        self.root = FPNode(None)
        self.headers: dict[int, FPNode] = {}       # item -> first node in chain
        self.item_counts: dict[int, int] = defaultdict(int)
        self.n_nodes = 0

    def add_transaction(self, items: Sequence[int], count: int = 1) -> None:
        """Insert an **already frequency-sorted** item sequence into the tree."""
        node = self.root
        for item in items:
            child = node.children.get(item)
            if child is None:
                child = FPNode(item, node)
                node.children[item] = child
                self.n_nodes += 1
                # Push onto the front of this item's node chain.
                child.link = self.headers.get(item)
                self.headers[item] = child
            child.count += count
            self.item_counts[item] += count
            node = child

    def nodes_for(self, item: int) -> Iterator[FPNode]:
        """Walk the node chain for *item*."""
        node = self.headers.get(item)
        while node is not None:
            yield node
            node = node.link

    def is_empty(self) -> bool:
        return not self.root.children

    def single_path(self) -> list[FPNode] | None:
        """Return the node list if the tree is one linear path, else ``None``.

        A single-path tree lets the miner emit every subset directly instead
        of recursing — the single biggest constant-factor win in FP-Growth.
        """
        path: list[FPNode] = []
        node = self.root
        while node.children:
            if len(node.children) > 1:
                return None
            node = next(iter(node.children.values()))
            path.append(node)
        return path

    def conditional_pattern_base(self, item: int) -> list[tuple[list[int], int]]:
        """Prefix paths ending at *item*, each with the count of that path."""
        base: list[tuple[list[int], int]] = []
        for node in self.nodes_for(item):
            prefix = node.path_to_root()
            if prefix:
                base.append((prefix, node.count))
        return base


def _build_tree(
    transactions: Iterable[Sequence[int]],
    order: dict[int, int],
    counts: Iterable[int] | None = None,
) -> FPTree:
    """Construct an FP-tree, projecting each transaction onto *order*.

    *order* maps the frequent items to their rank (0 = most frequent). Items
    absent from it are infrequent and dropped on the way in.
    """
    tree = FPTree()
    counts_iter = counts if counts is not None else None
    for idx, transaction in enumerate(transactions):
        filtered = [i for i in transaction if i in order]
        if not filtered:
            continue
        filtered.sort(key=order.__getitem__)
        weight = 1 if counts_iter is None else counts_iter[idx]
        tree.add_transaction(filtered, weight)
    return tree


def _mine_tree(
    tree: FPTree,
    suffix: tuple[int, ...],
    threshold: int,
    max_len: int,
    out: list[tuple[frozenset, int]],
    labels: np.ndarray,
    stats: MiningStats,
) -> None:
    """Recursive FP-Growth core; appends ``(itemset, count)`` pairs to *out*."""
    if len(suffix) >= max_len:
        return

    # Fast path: a single-path tree encodes every subset of that path.
    path = tree.single_path()
    if path is not None:
        budget = max_len - len(suffix)
        for r in range(1, min(len(path), budget) + 1):
            for combo in combinations(path, r):
                count = min(node.count for node in combo)
                if count >= threshold:
                    items = tuple(node.item for node in combo) + suffix
                    out.append((frozenset(labels[i] for i in items), count))
        stats.extra["single_path_shortcuts"] = stats.extra.get("single_path_shortcuts", 0) + 1
        return

    # Mine least-frequent item first so the conditional trees stay small.
    items_by_count = sorted(tree.item_counts.items(), key=lambda kv: (kv[1], kv[0]))
    for item, count in items_by_count:
        if count < threshold:
            continue
        new_suffix = (item,) + suffix
        out.append((frozenset(labels[i] for i in new_suffix), count))
        stats.itemsets_per_level[len(new_suffix)] = (
            stats.itemsets_per_level.get(len(new_suffix), 0) + 1
        )

        if len(new_suffix) >= max_len:
            continue

        base = tree.conditional_pattern_base(item)
        if not base:
            continue

        # Re-count items within the conditional base and drop infrequent ones.
        cond_counts: dict[int, int] = defaultdict(int)
        for prefix, weight in base:
            for i in prefix:
                cond_counts[i] += weight
        cond_order = {
            i: rank
            for rank, (i, c) in enumerate(
                sorted(
                    ((i, c) for i, c in cond_counts.items() if c >= threshold),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            )
        }
        if not cond_order:
            continue

        cond_tree = _build_tree(
            (prefix for prefix, _ in base),
            cond_order,
            counts=[w for _, w in base],
        )
        stats.candidates_per_level[len(new_suffix) + 1] = (
            stats.candidates_per_level.get(len(new_suffix) + 1, 0) + 1
        )
        stats.extra["conditional_trees"] = stats.extra.get("conditional_trees", 0) + 1

        if not cond_tree.is_empty():
            _mine_tree(cond_tree, new_suffix, threshold, max_len, out, labels, stats)


def fpgrowth(
    basket: BasketMatrix,
    min_support: float = 0.02,
    max_len: int | None = 3,
    min_count: int | None = None,
    track_memory: bool = True,
    verbose: bool = False,
) -> MiningResult:
    """Mine frequent itemsets with FP-Growth.

    Signature mirrors :func:`mba.mining.apriori.apriori` exactly so the two
    are drop-in interchangeable for benchmarking.
    """
    import sys

    n_tx = basket.n_transactions
    stats = MiningStats(
        algorithm="fpgrowth",
        min_support=min_support,
        max_len=max_len or -1,
        n_transactions=n_tx,
        n_items=basket.n_items,
    )
    if n_tx == 0 or basket.n_items == 0:
        return MiningResult(build_itemsets_frame([], 1), stats)

    threshold = int(min_count) if min_count is not None else int(np.ceil(min_support * n_tx))
    threshold = max(threshold, 1)
    limit = max_len if max_len and max_len > 0 else basket.n_items
    labels = basket.items
    results: list[tuple[frozenset, int]] = []

    # Deep conditional-tree recursion on dense data can exceed the default
    # limit; raise it for the duration of the run and restore afterwards.
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10_000))

    try:
        with _Timer(track_memory) as timer:
            # -- Pass 1: single-item counts -----------------------------
            counts = basket.support_counts()
            frequent = [(int(i), int(c)) for i, c in enumerate(counts) if c >= threshold]
            frequent.sort(key=lambda kv: (-kv[1], kv[0]))
            order = {item: rank for rank, (item, _) in enumerate(frequent)}
            stats.candidates_per_level[1] = int(basket.n_items)
            stats.itemsets_per_level[1] = len(frequent)

            if verbose:
                log.info("pass 1: %d/%d items frequent", len(frequent), basket.n_items)

            if frequent:
                # -- Pass 2: build the FP-tree --------------------------
                tree = _build_tree(basket.iter_transactions(), order)
                stats.extra["fp_tree_nodes"] = tree.n_nodes
                stats.extra["fp_tree_compression"] = (
                    round(basket.matrix.nnz / tree.n_nodes, 2) if tree.n_nodes else 0.0
                )
                if verbose:
                    log.info(
                        "pass 2: FP-tree with %d nodes (%.1fx compression over %d basket lines)",
                        tree.n_nodes,
                        basket.matrix.nnz / tree.n_nodes if tree.n_nodes else 0,
                        basket.matrix.nnz,
                    )

                # -- Mine ------------------------------------------------
                stats.itemsets_per_level.clear()  # recounted during recursion
                _mine_tree(tree, (), threshold, limit, results, labels, stats)
    finally:
        sys.setrecursionlimit(old_limit)

    stats.runtime_sec = timer.elapsed
    stats.peak_memory_mb = timer.peak_mb
    stats.n_itemsets = len(results)
    stats.extra["min_count_threshold"] = threshold

    log.info("%s", stats.summary())
    return MiningResult(build_itemsets_frame(results, n_tx), stats)


def fpgrowth_from_transactions(
    transactions: Iterable[Sequence[str]],
    min_support: float = 0.02,
    max_len: int | None = 3,
    **kwargs,
) -> MiningResult:
    """Convenience wrapper for list-of-lists input (used by the tests)."""
    from mba.data.basket import baskets_from_transactions

    return fpgrowth(
        baskets_from_transactions(transactions),
        min_support=min_support,
        max_len=max_len,
        **kwargs,
    )


register_miner("fpgrowth", fpgrowth)
register_miner("fp-growth", fpgrowth)
