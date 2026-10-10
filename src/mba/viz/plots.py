"""Static figures for the report bundle.

Each function takes data plus an output path, writes one PNG and returns the
path. Form follows the data's job:

* ranked magnitude -> horizontal bars, one colour, sorted, direct value labels
* a distribution -> histogram with the mean marked, not a pie
* two algorithms over a swept parameter -> one line per series on a log y-axis
  (runtimes span three orders of magnitude), legend plus direct labels
* rules -> support/confidence scatter with lift on a *diverging* ramp centred
  at 1.0, which is the genuine neutral point for lift

Deliberately absent: dual axes, rainbow ramps on ordered data, and gradients
that encode a bar's own height a second time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mba.viz.theme import (
    DIVERGING,
    STATUS,
    apply_theme,
    colors_for,
    sequential_colors,
    style_axes,
    truncate,
)

log = logging.getLogger(__name__)


def _new_figure(figsize=(11, 6), mode: str = "light", dpi: int = 130):
    import matplotlib.pyplot as plt

    theme = apply_theme(mode, dpi)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, theme


def _finish(fig, path: str | Path, title: str | None = None, subtitle: str | None = None) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    log.info("figure written: %s", path)
    return path


def _bar_labels(ax, bars, values, theme, fmt="{:,.0f}", pad=0.01) -> None:
    """Direct value labels at the end of each bar, in ink rather than series colour."""
    span = max(values) if len(values) else 1
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + span * pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=9,
            color=theme["text_secondary"],
        )


def _titles(ax, theme, title: str, subtitle: str = "") -> None:
    """Left-aligned title with an optional subtitle beneath it.

    The subtitle sits in the padding the title reserves, so the pad has to
    grow with it — otherwise the two render on the same baseline.
    """
    ax.set_title(title, loc="left", pad=30 if subtitle else 14)
    if subtitle:
        ax.text(
            0, 1.012, subtitle, transform=ax.transAxes,
            fontsize=9.5, color=theme["text_secondary"], va="bottom",
        )


# ---------------------------------------------------------------- products
def plot_top_products(
    frame: pd.DataFrame,
    path: str | Path,
    label_col: str,
    value_col: str,
    title: str,
    subtitle: str = "",
    value_fmt: str = "{:,.0f}",
    mode: str = "light",
    top_n: int = 15,
) -> Path:
    """Ranked horizontal bars — the correct form for 'top N by magnitude'.

    One series, so one colour: the bar length already encodes the value and a
    gradient over it would encode the same number twice while implying a
    categorical split that does not exist.
    """
    data = frame.nlargest(top_n, value_col).sort_values(value_col)
    fig, ax, theme = _new_figure((11, 0.42 * len(data) + 2.2), mode)

    positions = np.arange(len(data))
    bars = ax.barh(
        positions,
        data[value_col].to_numpy(),
        height=0.68,
        color=theme["categorical"][0],
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([truncate(v) for v in data[label_col]], fontsize=9)
    ax.set_xlabel(value_col.replace("_", " ").title())
    _titles(ax, theme, title, subtitle)
    _bar_labels(ax, bars, data[value_col].to_numpy(), theme, value_fmt)
    ax.margins(x=0.14)
    style_axes(ax, theme, xgrid=True)
    ax.tick_params(axis="y", length=0)
    return _finish(fig, path)


# ----------------------------------------------------------- distributions
def plot_basket_size_distribution(
    sizes: Sequence[int] | pd.Series,
    path: str | Path,
    title: str = "Basket size distribution",
    mode: str = "light",
    clip: int = 60,
) -> Path:
    """Histogram with the mean and median marked.

    The long tail is clipped and the clipped mass folded into the final bin,
    labelled as such — a 198-item wholesale basket would otherwise flatten
    the entire visible range.
    """
    values = np.asarray(pd.Series(sizes).dropna(), dtype=float)
    fig, ax, theme = _new_figure((11, 5.2), mode)

    clipped = np.clip(values, 0, clip)
    ax.hist(clipped, bins=np.arange(0, clip + 2) - 0.5, color=theme["categorical"][0], rwidth=0.82)

    mean, median = float(values.mean()), float(np.median(values))
    for value, label, color in (
        (median, f"median {median:.0f}", theme["categorical"][1]),
        (mean, f"mean {mean:.1f}", theme["categorical"][2]),
    ):
        ax.axvline(value, color=color, linewidth=2.0, linestyle="--")
        ax.text(value + 0.6, ax.get_ylim()[1] * 0.92, label, color=color, fontsize=9.5, va="top")

    ax.set_xlabel(f"Distinct items per basket (clipped at {clip})")
    ax.set_ylabel("Invoices")
    over = int((values > clip).sum())
    _titles(
        ax, theme, title,
        f"{over:,} baskets larger than {clip} items are folded into the final bin" if over else "",
    )
    style_axes(ax, theme)
    return _finish(fig, path)


def plot_support_distribution(
    supports: pd.Series,
    path: str | Path,
    title: str = "Item support follows a long tail",
    mode: str = "light",
) -> Path:
    """Rank-support curve on log-log axes.

    The straight-ish line is the point: product popularity is roughly
    power-law, which is *why* a support threshold prunes so aggressively and
    why only a few hundred of several thousand items are ever mineable.
    """
    values = np.sort(np.asarray(supports, dtype=float))[::-1]
    values = values[values > 0]
    fig, ax, theme = _new_figure((11, 5.2), mode)

    ax.plot(np.arange(1, len(values) + 1), values, color=theme["categorical"][0], linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Product rank (log)")
    ax.set_ylabel("Support (log)")
    ax.set_title(title, loc="left")

    for threshold, color in ((0.02, STATUS["warning"]), (0.01, STATUS["serious"])):
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1.6)
        n_above = int((values >= threshold).sum())
        ax.text(
            1.05, threshold * 1.12,
            f"support = {threshold:.0%}  ->  {n_above:,} items qualify",
            color=color, fontsize=9, va="bottom",
        )
    style_axes(ax, theme)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    return _finish(fig, path)


# -------------------------------------------------------------- benchmark
def plot_benchmark(
    frame: pd.DataFrame,
    path: str | Path,
    title: str = "Apriori vs FP-Growth: runtime against support threshold",
    mode: str = "light",
) -> Path:
    """Runtime curves, with the candidate counts that explain them beneath.

    Two panels sharing one x-axis rather than two y-scales on one plot. The
    lower panel is the whole explanation: Apriori's candidate count explodes
    by two orders of magnitude as support falls, while FP-Growth generates
    no candidates at all and its conditional-tree count barely moves. The
    runtime crossover in the upper panel is a direct consequence.

    Series are labelled directly at the left end, where the curves are well
    separated, instead of carrying a legend box into a crowded right edge.
    """
    import matplotlib.pyplot as plt

    theme = apply_theme(mode)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.42},
    )
    wide = frame.pivot_table(
        index="min_support", columns="algorithm", values="runtime_sec", aggfunc="mean"
    ).sort_index(ascending=False)
    palette = colors_for(max(len(wide.columns), 2), mode)
    color_of = dict(zip(wide.columns, palette))

    # -- upper panel: runtime -------------------------------------------
    for column in wide.columns:
        series = wide[column].dropna()
        if series.empty:
            continue
        axes[0].plot(
            series.index, series.to_numpy(), marker="o", markersize=8, linewidth=2.0,
            color=color_of[column], markeredgecolor=theme["surface"], markeredgewidth=2,
        )
        axes[0].annotate(
            column, (series.index[0], series.iloc[0]),
            textcoords="offset points", xytext=(-12, 0),
            ha="right", va="center", fontsize=10.5, fontweight="bold",
            color=color_of[column],
        )
    axes[0].set_ylabel("Runtime, seconds (log)")
    axes[0].set_yscale("log")
    axes[0].invert_xaxis()
    axes[0].margins(x=0.16)

    crossover = None
    if {"apriori", "fpgrowth"}.issubset(wide.columns):
        ratio = (wide["apriori"] / wide["fpgrowth"]).dropna()
        faster = ratio.loc[ratio > 1.0]
        if not faster.empty:
            crossover = float(faster.index.max())

    _titles(
        axes[0], theme, title,
        f"Identical itemsets from both; only the cost differs. FP-Growth overtakes "
        f"Apriori at support ≤ {crossover:.3g}." if crossover else
        "Identical itemsets from both; only the cost differs.",
    )

    # -- lower panel: why ------------------------------------------------
    if "total_candidates" in frame.columns:
        candidates = frame.pivot_table(
            index="min_support", columns="algorithm",
            values="total_candidates", aggfunc="mean",
        ).sort_index(ascending=False)
        # Both curves start at the same candidate count (every single item is
        # a level-1 candidate for either algorithm), so the left-end labels
        # are staggered vertically to keep them apart.
        offsets = (10, -10)
        for index, column in enumerate(candidates.columns):
            series = candidates[column].dropna()
            if series.empty:
                continue
            axes[1].plot(
                series.index, series.to_numpy(), marker="o", markersize=7,
                linewidth=2.0, color=color_of.get(column, theme["muted"]),
                markeredgecolor=theme["surface"], markeredgewidth=2,
            )
            axes[1].annotate(
                column, (series.index[0], series.iloc[0]),
                textcoords="offset points", xytext=(-12, offsets[index % len(offsets)]),
                ha="right", va="center", fontsize=10.5, fontweight="bold",
                color=color_of.get(column, theme["muted"]),
            )
        axes[1].set_yscale("log")
        axes[1].set_ylabel("Candidates evaluated (log)")
        _titles(
            axes[1], theme, "Why: candidate generation is the cost that explodes",
            "Apriori must generate and test every candidate; FP-Growth touches only "
            "patterns that exist.",
        )
    axes[1].set_xlabel("Minimum support threshold (harder →)")
    axes[1].margins(x=0.16)

    for ax in axes:
        if crossover is not None:
            ax.axvline(crossover, color=theme["muted"], linestyle=":", linewidth=1.6)
        style_axes(ax, theme)
    return _finish(fig, path)


def plot_scaling_curve(
    frame: pd.DataFrame,
    path: str | Path,
    title: str = "Runtime scales near-linearly with transaction volume",
    mode: str = "light",
) -> Path:
    """Runtime against transaction count, with a linear reference line."""
    fig, ax, theme = _new_figure((11, 5.6), mode)
    palette = colors_for(2, mode)

    x = frame["n_transactions"].to_numpy(dtype=float)
    y = frame["runtime_sec"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", color=palette[0], linewidth=2.0,
            markeredgecolor=theme["surface"], markeredgewidth=2, label="measured")

    if len(x) > 1 and x[0] > 0 and y[0] > 0:
        ax.plot(x, y[0] * x / x[0], linestyle="--", color=theme["muted"],
                linewidth=1.6, label="linear reference")

    ax.set_xlabel("Transactions mined")
    ax.set_ylabel("Runtime, seconds")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    style_axes(ax, theme)
    return _finish(fig, path)


# ------------------------------------------------------------------ rules
def plot_rule_scatter(
    rules: pd.DataFrame,
    path: str | Path,
    title: str = "Association rules: support, confidence and lift",
    mode: str = "light",
    max_points: int = 600,
) -> Path:
    """Support x confidence, coloured by lift on a diverging ramp.

    Lift has a true neutral point — 1.0 means independence — so a diverging
    ramp centred there is the honest encoding: blue below, red above, grey at
    nothing-going-on. A sequential ramp would imply 0 is the reference, which
    for lift is meaningless.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fig, ax, theme = _new_figure((11, 6.4), mode)
    data = rules.nlargest(max_points, "lift") if len(rules) > max_points else rules
    if data.empty:
        ax.text(0.5, 0.5, "No rules cleared the thresholds", ha="center", va="center",
                transform=ax.transAxes, color=theme["text_secondary"])
        return _finish(fig, path)

    midpoint = DIVERGING["mid_light"] if mode == "light" else DIVERGING["mid_dark"]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "lift", [DIVERGING["low"], midpoint, DIVERGING["high"]]
    )
    lift = data["lift"].to_numpy()
    norm = mcolors.TwoSlopeNorm(
        vmin=min(lift.min(), 0.95), vcenter=1.0, vmax=max(lift.max(), 1.05)
    )

    sizes = 24 + 340 * (data["support"] / data["support"].max())
    scatter = ax.scatter(
        data["support"], data["confidence"],
        c=lift, cmap=cmap, norm=norm, s=sizes,
        edgecolors=theme["surface"], linewidths=2, alpha=0.95,
    )
    bar = fig.colorbar(scatter, ax=ax, pad=0.015)
    bar.set_label("Lift  (1.0 = independent)", color=theme["text_secondary"], fontsize=9.5)
    bar.ax.tick_params(colors=theme["muted"], labelsize=9)
    bar.outline.set_visible(False)

    ax.set_xlabel("Support — how often the rule fires")
    ax.set_ylabel("Confidence — P(consequent | antecedent)")
    _titles(
        ax, theme, title,
        f"{len(data):,} rules; marker size is support. Top-right and red = frequent, "
        "reliable and non-obvious.",
    )

    # Label a few strong rules directly rather than every point. The highest-
    # lift rules tend to be variants of one another and land on top of each
    # other, so candidates are accepted greedily only when they are far
    # enough from an already-placed label to stay readable.
    x_span = max(data["support"].max() - data["support"].min(), 1e-9)
    y_span = max(data["confidence"].max() - data["confidence"].min(), 1e-9)
    placed: list[tuple[float, float]] = []
    for _, row in data.nlargest(40, "lift").iterrows():
        if len(placed) >= 4:
            break
        x, y = float(row["support"]), float(row["confidence"])
        if any(
            abs(x - px) / x_span < 0.18 and abs(y - py) / y_span < 0.10
            for px, py in placed
        ):
            continue
        placed.append((x, y))
        label = f"{', '.join(sorted(row['antecedents']))} → {', '.join(sorted(row['consequents']))}"
        ax.annotate(
            truncate(label, 38), (x, y),
            textcoords="offset points", xytext=(11, 9),
            fontsize=8.5, color=theme["text_secondary"],
        )
    style_axes(ax, theme)
    return _finish(fig, path)


def _sparsify(graph, keep_per_node: int = 4):
    """Keep each node's strongest *keep_per_node* edges, union across nodes.

    This is a readability filter for drawing only — the analysis in
    :mod:`mba.analytics.network` always runs on the full graph. Taking the
    union of per-node top edges (rather than a single global cut) keeps
    peripheral products attached instead of stranding them as isolates.
    """
    import networkx as nx

    keep = set()
    for node in graph.nodes():
        edges = sorted(
            graph.edges(node, data=True),
            key=lambda e: -e[2].get("weight", 0.0),
        )[:keep_per_node]
        keep.update((min(u, v), max(u, v)) for u, v, _ in edges)

    out = nx.Graph()
    out.add_nodes_from(graph.nodes(data=True))
    for u, v in keep:
        out.add_edge(u, v, **graph.edges[u, v])
    return out


def plot_network(
    graph,
    path: str | Path,
    title: str = "Product co-occurrence network",
    mode: str = "light",
    max_labels: int = 18,
    seed: int = 42,
) -> Path:
    """Spring-layout network, nodes coloured by detected community.

    Community is an identity attribute, so it takes categorical slots in
    fixed order. Past eight communities the remainder folds into a neutral
    'other' rather than inventing hues nobody can tell apart.
    """
    import networkx as nx

    from mba.analytics.network import detect_communities

    fig, ax, theme = _new_figure((12, 9), mode)
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No co-occurrence edges above threshold", ha="center",
                va="center", transform=ax.transAxes, color=theme["text_secondary"])
        ax.axis("off")
        return _finish(fig, path)

    communities = detect_communities(graph, seed=seed)
    unique = sorted(set(communities.values()))
    slots = colors_for(min(len(unique), 8), mode)
    color_of = {c: slots[i] if i < len(slots) else theme["muted"] for i, c in enumerate(unique)}

    # A lift-weighted graph over popular products is close to complete, and a
    # near-complete graph drawn in full is a grey disc. Keep each node's
    # strongest few links so the community structure survives the drawing.
    drawn = _sparsify(graph, keep_per_node=4)

    pos = nx.spring_layout(drawn, weight="weight", seed=seed, k=1.5, iterations=400)
    supports = np.array([drawn.nodes[n].get("support", 0.01) for n in drawn.nodes()])
    sizes = 90 + 1250 * supports / max(supports.max(), 1e-9)
    weights = np.array([d.get("weight", 1.0) for *_, d in drawn.edges(data=True)])
    widths = 0.6 + 2.6 * (weights / max(weights.max(), 1e-9))

    nx.draw_networkx_edges(
        drawn, pos, ax=ax, width=widths, edge_color=theme["axis"], alpha=0.6
    )
    nx.draw_networkx_nodes(
        drawn, pos, ax=ax, node_size=sizes,
        node_color=[color_of[communities[n]] for n in drawn.nodes()],
        edgecolors=theme["surface"], linewidths=2,
    )
    # Label only the most central nodes, offset below the marker and set on a
    # surface-coloured halo so a label never disappears into an edge.
    central = sorted(drawn.degree(weight="weight"), key=lambda kv: -kv[1])
    placed: list[tuple[float, float]] = []
    for node, _ in central:
        if len(placed) >= max_labels:
            break
        x, y = pos[node]
        # Tight clusters would stack labels on top of each other; skip a
        # label whose slot is already taken and move to the next node down
        # the centrality ranking.
        if any(abs(x - px) < 0.22 and abs(y - py) < 0.075 for px, py in placed):
            continue
        placed.append((x, y - 0.052))
        ax.text(
            x, y - 0.052, truncate(node, 24),
            fontsize=7.8, color=theme["text_primary"], ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=theme["surface"],
                      edgecolor="none", alpha=0.82),
        )

    _titles(
        ax, theme, title,
        f"{drawn.number_of_nodes()} products, {drawn.number_of_edges()} strongest "
        f"lift-weighted links, {len(unique)} communities. "
        "Node size = support; colour = community.",
    )
    ax.axis("off")
    return _finish(fig, path)


# ----------------------------------------------------------- regional/time
def plot_country_breakdown(
    frame: pd.DataFrame,
    path: str | Path,
    title: str = "Transactions by country",
    mode: str = "light",
    top_n: int = 12,
) -> Path:
    """Ranked bars with the dominant market called out in the subtitle."""
    # The label column is whatever the groupby index was called ("Country",
    # "country", ...), so resolve it rather than assuming a spelling.
    label_col = next(
        (c for c in frame.columns if c.lower() == "country"),
        frame.columns[0],
    )
    data = frame.nlargest(top_n, "transactions").sort_values("transactions")
    fig, ax, theme = _new_figure((11, 0.42 * len(data) + 2.2), mode)

    positions = np.arange(len(data))
    bars = ax.barh(positions, data["transactions"].to_numpy(), height=0.68,
                   color=theme["categorical"][0])
    ax.set_yticks(positions)
    ax.set_yticklabels([truncate(c, 26) for c in data[label_col]], fontsize=9)
    _bar_labels(ax, bars, data["transactions"].to_numpy(), theme)

    total = frame["transactions"].sum()
    top = frame.nlargest(1, "transactions").iloc[0]
    _titles(
        ax, theme, title,
        f"{top[label_col]} alone is {top['transactions'] / total:.0%} of all baskets — "
        "which is why rules are also mined per market.",
    )
    ax.set_xlabel("Transactions")
    ax.margins(x=0.14)
    style_axes(ax, theme, xgrid=True)
    ax.tick_params(axis="y", length=0)
    return _finish(fig, path)


def plot_monthly_trend(
    frame: pd.DataFrame,
    path: str | Path,
    title: str = "Monthly transactions and revenue",
    mode: str = "light",
) -> Path:
    """Two stacked panels, never two y-axes on one plot.

    Transactions and revenue have unrelated scales. Overlaying them on a dual
    axis would let the choice of scale invent or erase a correlation; small
    multiples sharing an x-axis keep the comparison honest.
    """
    import matplotlib.pyplot as plt

    theme = apply_theme(mode)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"hspace": 0.28})
    palette = colors_for(2, mode)

    axes[0].plot(frame["month"], frame["transactions"], marker="o", color=palette[0],
                 markeredgecolor=theme["surface"], markeredgewidth=2)
    axes[0].set_ylabel("Transactions")
    axes[0].set_title(title, loc="left")

    axes[1].plot(frame["month"], frame["revenue"], marker="o", color=palette[1],
                 markeredgecolor=theme["surface"], markeredgewidth=2)
    axes[1].set_ylabel("Revenue (£)")
    axes[1].set_xlabel("Month")

    for ax in axes:
        style_axes(ax, theme)
        ax.tick_params(axis="x", rotation=45)
    return _finish(fig, path)


# ----------------------------------------------------------------- bundle
def save_all_figures(
    artifacts: dict[str, Any],
    figures_dir: str | Path,
    mode: str = "light",
) -> dict[str, Path]:
    """Render every figure the pipeline can produce from *artifacts*.

    Each figure is attempted independently so one failure (an empty table, a
    missing optional stage) never costs the rest of the bundle.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def attempt(name: str, fn, *args, **kwargs) -> None:
        try:
            written[name] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            log.warning("figure %r skipped: %s", name, exc)

    eda = artifacts.get("eda")
    if eda is not None:
        if "top_products_volume" in eda.tables:
            attempt(
                "top_products_volume", plot_top_products,
                eda["top_products_volume"], figures_dir / "01_top_products_volume.png",
                eda["top_products_volume"].columns[0], "transactions",
                "Top products by number of baskets",
                "How often each product appears in an order — the raw material for support.",
                mode=mode,
            )
        if "top_products_revenue" in eda.tables:
            attempt(
                "top_products_revenue", plot_top_products,
                eda["top_products_revenue"], figures_dir / "02_top_products_revenue.png",
                eda["top_products_revenue"].columns[0], "revenue",
                "Top products by revenue (£)",
                "Volume and value rank differently — a reason not to merchandise on volume alone.",
                value_fmt="£{:,.0f}", mode=mode,
            )
        if "basket_sizes" in eda.tables:
            attempt("basket_sizes", plot_basket_size_distribution,
                    eda["basket_sizes"]["items"], figures_dir / "03_basket_size_distribution.png",
                    mode=mode)
        if "item_support_distribution" in eda.tables:
            attempt("support_distribution", plot_support_distribution,
                    eda["item_support_distribution"]["support"],
                    figures_dir / "04_item_support_longtail.png", mode=mode)
        if "top_countries" in eda.tables:
            attempt("countries", plot_country_breakdown, eda["top_countries"],
                    figures_dir / "05_country_breakdown.png", mode=mode)
        if "monthly" in eda.tables:
            attempt("monthly", plot_monthly_trend, eda["monthly"],
                    figures_dir / "06_monthly_trend.png", mode=mode)

    rules = artifacts.get("rules")
    if rules is not None and not rules.empty:
        attempt("rule_scatter", plot_rule_scatter, rules,
                figures_dir / "07_rule_scatter.png", mode=mode)

    graph = artifacts.get("graph")
    if graph is not None:
        attempt("network", plot_network, graph, figures_dir / "08_cooccurrence_network.png",
                mode=mode)

    benchmark = artifacts.get("benchmark")
    if benchmark is not None and not benchmark.empty:
        attempt("benchmark", plot_benchmark, benchmark,
                figures_dir / "09_benchmark_apriori_vs_fpgrowth.png", mode=mode)

    scaling = artifacts.get("scaling")
    if scaling is not None and not scaling.empty:
        attempt("scaling", plot_scaling_curve, scaling,
                figures_dir / "10_scaling_curve.png", mode=mode)

    log.info("wrote %d figures to %s", len(written), figures_dir)
    return written
