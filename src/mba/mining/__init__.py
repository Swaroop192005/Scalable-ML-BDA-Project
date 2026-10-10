"""Frequent-itemset mining and association-rule generation."""

from mba.mining.apriori import apriori
from mba.mining.base import MiningResult, MiningStats, mine
from mba.mining.fpgrowth import FPTree, fpgrowth
from mba.mining.rules import association_rules, rule_metrics
from mba.mining.spurious import SpuriousFlags, flag_spurious_rules

__all__ = [
    "FPTree",
    "MiningResult",
    "MiningStats",
    "SpuriousFlags",
    "apriori",
    "association_rules",
    "flag_spurious_rules",
    "fpgrowth",
    "mine",
    "rule_metrics",
]
