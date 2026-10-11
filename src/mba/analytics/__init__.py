"""Analytics: EDA, networks, regions, segments, drift and insights."""

from mba.analytics.eda import EDAReport, run_eda
from mba.analytics.insights import Insight, generate_insights
from mba.analytics.network import build_cooccurrence_graph, graph_summary
from mba.analytics.regional import regional_analysis
from mba.analytics.segmentation import (
    SegmentationResult,
    choose_k,
    compute_rfm,
    kmeans,
    segment_customers,
    silhouette_score,
)
from mba.analytics.temporal import (
    TemporalResult,
    rule_trajectory_matrix,
    seasonal_rules,
    temporal_analysis,
    trending_rules,
)

__all__ = [
    "EDAReport",
    "Insight",
    "SegmentationResult",
    "TemporalResult",
    "build_cooccurrence_graph",
    "choose_k",
    "compute_rfm",
    "generate_insights",
    "graph_summary",
    "kmeans",
    "regional_analysis",
    "rule_trajectory_matrix",
    "run_eda",
    "seasonal_rules",
    "segment_customers",
    "silhouette_score",
    "temporal_analysis",
    "trending_rules",
]
