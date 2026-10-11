"""Frequent-itemset mining and association-rule generation."""

from mba.mining.apriori import apriori
from mba.mining.base import MiningResult, MiningStats, mine
from mba.mining.fpgrowth import FPTree, fpgrowth
from mba.mining.rules import association_rules, rule_metrics
from mba.mining.compact import (
    CompactResult,
    closed_itemsets,
    compact_representations,
    maximal_itemsets,
)
from mba.mining.spurious import (
    SpuriousFlags,
    benjamini_hochberg,
    flag_spurious_rules,
)

__all__ = [
    "CompactResult",
    "FPTree",
    "MiningResult",
    "MiningStats",
    "SpuriousFlags",
    "apriori",
    "association_rules",
    "benjamini_hochberg",
    "closed_itemsets",
    "compact_representations",
    "flag_spurious_rules",
    "fpgrowth",
    "maximal_itemsets",
    "mine",
    "rule_metrics",
]
