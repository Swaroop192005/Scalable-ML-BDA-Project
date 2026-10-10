"""Analytics: EDA, co-occurrence networks, regional splits and insights."""

from mba.analytics.eda import EDAReport, run_eda
from mba.analytics.insights import Insight, generate_insights
from mba.analytics.network import build_cooccurrence_graph, graph_summary
from mba.analytics.regional import regional_analysis

__all__ = [
    "EDAReport",
    "Insight",
    "build_cooccurrence_graph",
    "generate_insights",
    "graph_summary",
    "regional_analysis",
    "run_eda",
]
