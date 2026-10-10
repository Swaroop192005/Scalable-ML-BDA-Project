"""Visualisation: a shared theme plus the figures used in reports and the app."""

from mba.viz.theme import PALETTE, apply_theme, colors_for
from mba.viz.plots import (
    plot_basket_size_distribution,
    plot_benchmark,
    plot_country_breakdown,
    plot_monthly_trend,
    plot_network,
    plot_rule_scatter,
    plot_scaling_curve,
    plot_support_distribution,
    plot_top_products,
    save_all_figures,
)

__all__ = [
    "PALETTE",
    "apply_theme",
    "colors_for",
    "plot_basket_size_distribution",
    "plot_benchmark",
    "plot_country_breakdown",
    "plot_monthly_trend",
    "plot_network",
    "plot_rule_scatter",
    "plot_scaling_curve",
    "plot_support_distribution",
    "plot_top_products",
    "save_all_figures",
]
