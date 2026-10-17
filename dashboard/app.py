"""Interactive dashboard for the market basket analysis platform.

Everything on screen is computed live from the configured dataset: move a
threshold and the itemsets, rules and charts are re-mined, not re-filtered
from a precomputed table. Streamlit's cache keys on the parameters, so a
threshold you have already visited comes back instantly while a new one
actually re-runs the miner.

Run with ``mba dashboard`` or ``streamlit run dashboard/app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run dashboard/app.py` from a source checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mba.analytics.eda import run_eda
from mba.analytics.insights import generate_insights
from mba.analytics.network import build_cooccurrence_graph, graph_summary
from mba.config import load_config
from mba.data.basket import BasketMatrix, encode_baskets
from mba.data.loader import load_raw
from mba.data.preprocess import clean_transactions
from mba.mining.base import mine
from mba.mining.rules import association_rules, format_rules
from mba.mining.spurious import flag_spurious_rules
from mba.viz.theme import CATEGORICAL_DARK, DARK, SEQUENTIAL_BLUE, STATUS

ALL_COUNTRIES = "All countries"

cfg = load_config()

st.set_page_config(
    page_title=cfg.get("dashboard.title", "Market Basket Analysis"),
    page_icon=cfg.get("dashboard.page_icon", "🛒"),
    layout=cfg.get("dashboard.layout", "wide"),
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
      .metric-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 18px; }}
      .metric-card {{
          flex: 1 1 170px; background: {DARK['surface']};
          border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
          padding: 16px 18px;
      }}
      .metric-value {{
          font-size: 1.95rem; font-weight: 700; color: {CATEGORICAL_DARK[0]};
          line-height: 1.15;
      }}
      .metric-label {{
          font-size: 0.72rem; letter-spacing: .09em; text-transform: uppercase;
          color: {DARK['text_secondary']}; margin-top: 4px;
      }}
      .metric-note {{ font-size: 0.72rem; color: {DARK['muted']}; margin-top: 6px; }}
      .caption {{ color: {DARK['text_secondary']}; font-size: 0.9rem; margin-bottom: 10px; }}
      div[data-testid="stDataFrame"] {{ border-radius: 10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

def _streamlit_version() -> tuple[int, ...]:
    """(major, minor, patch) of the installed Streamlit, missing parts as 0."""
    parts: list[int] = []
    for chunk in st.__version__.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


# Streamlit 1.49 replaced `use_container_width=True` with `width="stretch"`
# and warns loudly on every element until you switch — which, on a dashboard
# with 25 of them, floods the console on every rerun.
#
# requirements.txt allows streamlit>=1.28, so hard-coding either spelling
# breaks half the supported range: the new one raises TypeError on old
# versions, the old one warns on new ones. Pick at import time instead and
# splat it at each call site.
FULL_WIDTH = (
    {"width": "stretch"} if _streamlit_version() >= (1, 49) else {"use_container_width": True}
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK["surface"],
    plot_bgcolor=DARK["surface"],
    font=dict(color=DARK["text_secondary"], size=12,
              family="system-ui, -apple-system, Segoe UI, sans-serif"),
    title=dict(font=dict(color=DARK["text_primary"], size=15), x=0, xanchor="left"),
    margin=dict(l=10, r=20, t=56, b=40),
    xaxis=dict(gridcolor=DARK["grid"], zerolinecolor=DARK["axis"], linecolor=DARK["axis"]),
    yaxis=dict(gridcolor=DARK["grid"], zerolinecolor=DARK["axis"], linecolor=DARK["axis"]),
    hoverlabel=dict(bgcolor=DARK["page"], bordercolor=DARK["axis"],
                    font=dict(color=DARK["text_primary"])),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, yanchor="bottom"),
)


# ----------------------------------------------------------------- caching
@st.cache_data(show_spinner="Loading and cleaning transactions…")
def load_clean() -> pd.DataFrame:
    raw = load_raw(cfg)
    clean, _ = clean_transactions(raw, cfg)
    return clean


@st.cache_data(show_spinner="Encoding baskets…")
def build_basket(country: str) -> tuple[BasketMatrix, pd.DataFrame]:
    frame = load_clean()
    if country != ALL_COUNTRIES:
        frame = frame.loc[frame[cfg["schema"]["country_col"]] == country]
    return encode_baskets(frame, cfg, min_item_count=2), frame


@st.cache_data(show_spinner="Mining frequent itemsets…")
def mine_rules(
    country: str, algorithm: str, min_support: float,
    min_confidence: float, min_lift: float, max_len: int,
):
    basket, frame = build_basket(country)
    result = mine(
        basket, algorithm=algorithm, min_support=min_support,
        max_len=max_len, track_memory=False,
    )
    rules = association_rules(
        result, min_confidence=min_confidence, min_lift=min_lift, sort_by="lift"
    )
    flags = None
    if not rules.empty:
        rules, flags = flag_spurious_rules(rules, cfg, basket.n_transactions)
    return basket, frame, result, rules, flags


@st.cache_data(show_spinner="Running exploratory analysis…")
def eda_for(country: str):
    basket, frame = build_basket(country)
    return run_eda(frame, cfg, basket)


@st.cache_data(show_spinner="Building the co-occurrence network…")
def network_for(country: str, top_n: int, min_lift: float):
    basket, _ = build_basket(country)
    graph = build_cooccurrence_graph(
        basket, cfg, top_n=top_n, weight="lift", min_edge_weight=min_lift
    )
    return graph, graph_summary(graph)


@st.cache_data(show_spinner="Scoring recommenders on a held-out future…")
def recommender_evaluation(test_fraction: float, k_values: tuple[int, ...],
                           min_support: float, max_cases: int | None):
    from mba.recommend import compare_recommenders, temporal_split
    from mba.recommend.engine import build_recommender
    from mba.recommend.evaluate import lift_over_baseline

    frame = load_clean()
    schema = cfg["schema"]
    train, test, meta = temporal_split(
        frame, schema["invoice_col"], schema["item_col"], schema["date_col"],
        test_fraction=test_fraction,
    )
    models = {
        "association": build_recommender(
            "association", mine_kwargs={"min_support": min_support, "max_len": 3}
        ),
        "cooccurrence": build_recommender("cooccurrence"),
        "popularity": build_recommender("popularity"),
        "random": build_recommender("random"),
    }
    scores = compare_recommenders(
        train, test, models=models, k_values=k_values, max_cases=max_cases
    )
    return lift_over_baseline(scores), meta


@st.cache_data(show_spinner="Clustering customers on RFM…")
def segmentation(k: int, min_support: float):
    from mba.analytics.segmentation import compare_segment_rules, segment_customers

    result = segment_customers(load_clean(), cfg, k=k, min_support=min_support)
    return result.profiles, result.customers, compare_segment_rules(result, top_n=5), result.silhouette


@st.cache_data(show_spinner="Mining each period separately…")
def temporal(freq: str, min_support: float):
    from mba.analytics.temporal import rule_trajectory_matrix, temporal_analysis

    result = temporal_analysis(load_clean(), cfg, freq=freq, min_support=min_support)
    return result.period_stats, result.drift, rule_trajectory_matrix(result, 18), result.periods


@st.cache_data(show_spinner="Streaming in one bounded-memory pass…")
def streaming(min_support: float, epsilon: float):
    from mba.scaling.streaming import compare_with_exact, stream_frequent_items

    basket, _ = build_basket(ALL_COUNTRIES)
    transactions = basket.as_item_lists()
    frame, stats = stream_frequent_items(transactions, min_support, epsilon)
    return frame, stats.to_dict(), compare_with_exact(transactions, min_support, epsilon)


# -------------------------------------------------------------- components
def metric_cards(cards: list[tuple[str, str, str]]) -> None:
    """Render a row of stat tiles. A number is a tile, not a chart."""
    html = "".join(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-note">{note}</div></div>'
        for value, label, note in cards
    )
    st.markdown(f'<div class="metric-row">{html}</div>', unsafe_allow_html=True)


def ranked_bar(frame: pd.DataFrame, label_col: str, value_col: str,
               title: str, value_fmt: str = ",.0f", prefix: str = "") -> go.Figure:
    """Horizontal ranked bars — one series, therefore one colour.

    A gradient keyed to bar length would encode the value twice and imply a
    category that is not in the data.
    """
    data = frame.sort_values(value_col).tail(15)
    figure = go.Figure(
        go.Bar(
            x=data[value_col], y=data[label_col], orientation="h",
            marker=dict(color=CATEGORICAL_DARK[0], line=dict(width=0)),
            text=[f"{prefix}{v:{value_fmt}}" for v in data[value_col]],
            textposition="outside",
            textfont=dict(color=DARK["text_secondary"], size=11),
            hovertemplate=f"<b>%{{y}}</b><br>{value_col}: {prefix}%{{x:{value_fmt}}}<extra></extra>",
        )
    )
    figure.update_layout(
        **PLOTLY_LAYOUT, title_text=title,
        height=max(380, 30 * len(data)), bargap=0.3, showlegend=False,
    )
    # Outside bar labels need headroom or the longest one is clipped at the
    # plot edge — the exact bar a reader looks at first.
    figure.update_xaxes(
        title_text=value_col.replace("_", " ").title(),
        range=[0, float(data[value_col].max()) * 1.22],
    )
    figure.update_yaxes(automargin=True, ticksuffix="  ")
    return figure


def rule_scatter(rules: pd.DataFrame) -> go.Figure:
    """Support x confidence, lift on a diverging ramp centred at 1.0."""
    data = rules.nlargest(500, "lift")
    labels = [
        f"{', '.join(sorted(a))} → {', '.join(sorted(c))}"
        for a, c in zip(data["antecedents"], data["consequents"])
    ]
    # Lift cannot go below 0 and its neutral point is 1.0, so the diverging
    # ramp is anchored asymmetrically: the grey midpoint is placed at wherever
    # 1.0 actually falls between the observed min and max. A symmetric scale
    # (plotly's `cmid`) would put negative lift on the colorbar, which does
    # not exist.
    low = min(float(data["lift"].min()), 0.9)
    high = max(float(data["lift"].max()), 1.1)
    neutral = (1.0 - low) / (high - low)
    figure = go.Figure(
        go.Scatter(
            x=data["support"], y=data["confidence"], mode="markers", text=labels,
            marker=dict(
                size=9 + 26 * data["support"] / data["support"].max(),
                color=data["lift"],
                colorscale=[
                    [0.0, "#1c5cab"],
                    [max(neutral - 0.001, 0.0), DARK["grid"]],
                    [min(neutral + 0.001, 1.0), DARK["grid"]],
                    [1.0, "#d03b3b"],
                ],
                cmin=low, cmax=high,
                colorbar=dict(
                    title=dict(text="Lift", side="right"),
                    outlinewidth=0, tickfont=dict(color=DARK["muted"]),
                ),
                line=dict(color=DARK["surface"], width=1.5),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>support %{x:.3%}<br>confidence %{y:.1%}"
                "<br>lift %{marker.color:.2f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **PLOTLY_LAYOUT, height=560,
        title_text="Rule landscape — lift 1.0 means the items are independent",
    )
    figure.update_xaxes(title_text="Support", tickformat=".1%")
    figure.update_yaxes(title_text="Confidence", tickformat=".0%")
    return figure


def network_figure(graph, summary: dict) -> go.Figure:
    """Spring-laid co-occurrence network, nodes coloured by community."""
    import networkx as nx

    from mba.analytics.network import detect_communities
    from mba.viz.plots import _sparsify

    drawn = _sparsify(graph, keep_per_node=4)
    pos = nx.spring_layout(drawn, weight="weight", seed=42, k=1.4, iterations=320)
    communities = detect_communities(drawn, seed=42)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in drawn.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    supports = [drawn.nodes[n].get("support", 0.01) for n in drawn.nodes()]
    peak = max(supports) if supports else 1.0
    slots = CATEGORICAL_DARK

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="skip",
                   line=dict(color=DARK["axis"], width=1), showlegend=False)
    )
    figure.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in drawn.nodes()],
            y=[pos[n][1] for n in drawn.nodes()],
            mode="markers+text",
            text=[n.title()[:22] for n in drawn.nodes()],
            textposition="bottom center",
            textfont=dict(size=9, color=DARK["text_secondary"]),
            marker=dict(
                size=[14 + 30 * s / peak for s in supports],
                color=[slots[communities.get(n, 0) % len(slots)] for n in drawn.nodes()],
                line=dict(color=DARK["surface"], width=2),
            ),
            customdata=[
                [drawn.nodes[n].get("support", 0), drawn.degree(n)] for n in drawn.nodes()
            ],
            hovertemplate=(
                "<b>%{text}</b><br>support %{customdata[0]:.2%}"
                "<br>linked products %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    # Report the community count of the graph actually drawn. The full graph
    # partitions differently from the sparsified one, so quoting the full
    # graph's count here would contradict the colours on screen.
    figure.update_layout(
        **PLOTLY_LAYOUT, height=650,
        title_text=(
            f"{drawn.number_of_nodes()} products · {drawn.number_of_edges()} strongest links · "
            f"{len(set(communities.values()))} communities"
        ),
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def rules_table(rules: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Rule columns formatted for display, highest lift first."""
    columns = [
        "rule", "support", "confidence", "lift", "leverage",
        "conviction", "zhangs_metric", "count", "quality_tier", "flag_reasons",
    ]
    shown = format_rules(rules.nlargest(limit, "lift"))
    return shown[[c for c in columns if c in shown.columns]]


# ------------------------------------------------------------------ layout
st.title("🛒 Scalable Market Basket Analysis")
st.markdown(
    '<div class="caption">Association rule mining over retail transactions — '
    "Apriori and FP-Growth, mined live from the thresholds you set.</div>",
    unsafe_allow_html=True,
)

try:
    clean = load_clean()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Run `python scripts/download_data.py`, then reload this page.")
    st.stop()

country_col = cfg["schema"]["country_col"]
countries = [ALL_COUNTRIES] + (
    clean[country_col].value_counts().index.tolist() if country_col in clean.columns else []
)

with st.sidebar:
    st.header("⚙️ Mining controls")
    st.caption("Every control re-runs the miner. Results are cached per parameter set.")

    country = st.selectbox("Market", countries, index=0)
    algorithm = st.radio(
        "Algorithm", ["fpgrowth", "apriori"], horizontal=True,
        help="Both return identical itemsets — they differ only in cost. "
             "See the Benchmark tab.",
    )
    min_support = st.slider(
        "Minimum support", 0.005, 0.10, float(cfg.get("mining.min_support", 0.02)),
        0.005, format="%.3f",
        help="Fraction of baskets an itemset must appear in. The single most "
             "expensive knob — halving it can multiply runtime tenfold.",
    )
    min_confidence = st.slider(
        "Minimum confidence", 0.05, 0.95,
        float(cfg.get("mining.min_confidence", 0.3)), 0.05,
        help="P(consequent | antecedent).",
    )
    min_lift = st.slider(
        "Minimum lift", 1.0, 10.0, float(cfg.get("mining.min_lift", 1.2)), 0.1,
        help="1.0 means the items are statistically independent.",
    )
    max_len = st.slider("Maximum itemset size", 2, 4, int(cfg.get("mining.max_len", 3)))
    hide_spurious = st.checkbox(
        "Hide rules flagged as misleading", value=True,
        help="Filters rules that clear the thresholds but fail the quality "
             "checks — rare pairs, near-universal consequents, colour variants "
             "of one product, and associations that fail a chi-squared "
             "independence test.",
    )
    st.divider()
    st.caption(
        f"Source: {cfg.path('paths.raw_file').name}  \n"
        f"{len(clean):,} cleaned basket lines"
    )

basket, frame, itemsets, rules, flags = mine_rules(
    country, algorithm, min_support, min_confidence, min_lift, max_len
)
eda = eda_for(country)

visible = rules
if hide_spurious and "is_spurious" in rules.columns:
    visible = rules.loc[~rules["is_spurious"]]

metric_cards([
    (f"{eda.scalars['n_transactions']:,}", "Baskets", "invoices with 2+ items"),
    (f"{eda.scalars['n_products']:,}", "Products", f"{basket.n_items:,} after pruning"),
    (f"{eda.scalars.get('n_customers', 0):,}", "Customers", "distinct IDs"),
    (f"{len(itemsets):,}", "Frequent itemsets", f"at support ≥ {min_support:.1%}"),
    (f"{len(visible):,}", "Rules", f"of {len(rules):,} before quality filter"),
    (f"£{eda.scalars.get('total_revenue', 0):,.0f}", "Revenue", "in scope"),
])

tabs = st.tabs([
    "📊 Overview", "🔗 Rules", "🕸️ Network", "🌍 Regions",
    "🎯 Recommender", "👥 Segments", "📈 Drift",
    "⚡ Benchmark", "🌊 Streaming", "💡 Insights", "🧾 Data quality",
])

# -- Overview ------------------------------------------------------------
with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            ranked_bar(
                eda["top_products_volume"], eda["top_products_volume"].columns[0],
                "transactions", "Top products by number of baskets",
            ), **FULL_WIDTH,
        )
    with right:
        st.plotly_chart(
            ranked_bar(
                eda["top_products_revenue"], eda["top_products_revenue"].columns[0],
                "revenue", "Top products by revenue", prefix="£",
            ), **FULL_WIDTH,
        )

    left, right = st.columns(2)
    with left:
        sizes = eda["basket_size_distribution"]
        figure = go.Figure(
            go.Bar(
                x=sizes["basket_size"], y=sizes["n_invoices"],
                marker_color=CATEGORICAL_DARK[0],
                hovertemplate="%{x} items<br>%{y:,} invoices<extra></extra>",
            )
        )
        figure.update_layout(**PLOTLY_LAYOUT, height=400, bargap=0.15,
                             title_text="Basket size distribution")
        figure.update_xaxes(title_text="Distinct items per basket", range=[0, 60])
        figure.update_yaxes(title_text="Invoices")
        st.plotly_chart(figure, **FULL_WIDTH)
        st.caption(
            f"Median {eda.scalars['median_basket_size']:.0f} items, "
            f"mean {eda.scalars['mean_basket_size']:.1f}, "
            f"largest {eda.scalars['max_basket_size']:,}. "
            "The long right tail is wholesale orders."
        )
    with right:
        if "monthly" in eda.tables:
            monthly = eda["monthly"]
            figure = go.Figure(
                go.Scatter(
                    x=monthly["month"], y=monthly["transactions"],
                    mode="lines+markers", line=dict(color=CATEGORICAL_DARK[0], width=2),
                    marker=dict(size=8, line=dict(color=DARK["surface"], width=2)),
                    hovertemplate="%{x}<br>%{y:,} baskets<extra></extra>",
                )
            )
            figure.update_layout(**PLOTLY_LAYOUT, height=400,
                                 title_text="Baskets per month")
            figure.update_yaxes(title_text="Baskets")
            st.plotly_chart(figure, **FULL_WIDTH)
            st.caption(
                "Revenue is plotted separately in the report bundle rather than "
                "on a second y-axis here — two scales on one chart can invent a "
                "correlation that is not in the data."
            )

    st.subheader("Sparsity — why a dense matrix is not an option")
    stats = basket.describe()
    metric_cards([
        (f"{stats['density']:.3%}", "Matrix density", "non-zero cells"),
        (f"{stats['sparse_mb']:.1f} MB", "Sparse (CSR)", "what we actually hold"),
        (f"{stats['dense_equivalent_mb']:.0f} MB", "Dense equivalent", "what pivot_table builds"),
        (f"{stats['compression_ratio']:.0f}×", "Compression", "sparse vs dense"),
    ])

# -- Rules ---------------------------------------------------------------
with tabs[1]:
    if visible.empty:
        st.warning(
            "No rules cleared these thresholds. Lower the minimum support or "
            "confidence in the sidebar."
        )
    else:
        st.plotly_chart(rule_scatter(visible), **FULL_WIDTH)
        st.subheader(f"{len(visible):,} rules")
        st.dataframe(
            rules_table(visible, int(cfg.get("dashboard.max_rules_displayed", 500))), **FULL_WIDTH, height=440, hide_index=True,
        )
        st.download_button(
            "Download rules as CSV",
            format_rules(visible).to_csv(index=False).encode("utf-8"),
            file_name=f"association_rules_{country.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

        with st.expander("What each metric means"):
            st.markdown(
                """
| Metric | Definition | Reads as |
|---|---|---|
| **support** | `s(A ∪ C)` | how often the pair actually happens |
| **confidence** | `s(A ∪ C) / s(A)` | P(consequent given antecedent) |
| **lift** | `confidence / s(C)` | ratio over the base rate; 1.0 = independent |
| **leverage** | `s(A ∪ C) − s(A)·s(C)` | the *absolute* excess co-occurrence |
| **conviction** | `(1 − s(C)) / (1 − confidence)` | how often the rule would be wrong by chance |
| **Zhang's metric** | bounded association in [−1, 1] | stays meaningful when supports are skewed |

Lift alone is unbounded and unstable for rare items: a pair seen four times
can top the chart. Leverage keeps the size of the effect in view, and Zhang's
metric stays interpretable when the marginals are lopsided.
                """
            )

# -- Network -------------------------------------------------------------
with tabs[2]:
    top_n = st.slider("Products in the network", 15, 70, 34, 1)
    edge_lift = st.slider("Minimum edge lift", 1.0, 5.0, 1.5, 0.1)
    graph, summary = network_for(country, top_n, edge_lift)

    if graph.number_of_nodes() == 0:
        st.warning("No co-occurrence links above this lift threshold.")
    else:
        st.plotly_chart(network_figure(graph, summary), **FULL_WIDTH)
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Centrality")
            st.caption(
                "Weighted degree finds the hubs that pull whole baskets. "
                "Betweenness finds the bridges between product communities — "
                "the cheapest route into a second category."
            )
            st.dataframe(
                summary["centrality_table"].head(20), **FULL_WIDTH,
                hide_index=True,
            )
        with right:
            st.subheader("Communities")
            st.caption("Product families discovered from behaviour, not from the catalogue.")
            for community_id, members in list(summary.get("communities", {}).items())[:5]:
                st.markdown(f"**Family {int(community_id) + 1}** — {len(members)} products")
                st.caption(" · ".join(m.title() for m in members[:6]))

# -- Regions -------------------------------------------------------------
with tabs[3]:
    st.caption(
        "The UK is the large majority of this dataset, so a single global rules "
        "table is effectively a UK table. Each market is mined separately at a "
        "locally calibrated threshold."
    )
    if st.button("Run regional analysis", type="primary"):
        from mba.analytics.regional import compare_regions, distinctive_rules, regional_analysis

        with st.spinner("Mining each market separately…"):
            regions = regional_analysis(clean, cfg)
            comparison = compare_regions(regions)
            distinctive = distinctive_rules(regions)
        st.session_state["regions"] = (comparison, distinctive)

    if "regions" in st.session_state:
        comparison, distinctive = st.session_state["regions"]
        st.subheader("Markets compared")
        st.dataframe(comparison, **FULL_WIDTH, hide_index=True)
        if not distinctive.empty:
            st.subheader("Associations much stronger than in the UK")
            st.caption(
                "A rule that holds everywhere is a product fact. A rule with "
                "several times the lift in one market is a local fact — and the "
                "only kind worth a localised campaign."
            )
            st.dataframe(distinctive, **FULL_WIDTH, hide_index=True)

# -- Recommender ---------------------------------------------------------
with tabs[4]:
    st.caption(
        "Mining proves a pattern existed. This proves it **predicts**. Models "
        "train on the earliest baskets and are scored on later ones they have "
        "never seen — a temporal split, because in production you only ever "
        "have the past."
    )
    left, right, far = st.columns(3)
    with left:
        test_fraction = st.slider("Test share (most recent baskets)", 0.1, 0.4, 0.25, 0.05)
    with right:
        rec_support = st.select_slider(
            "Rule-mining support (training split only)",
            options=[0.003, 0.005, 0.01, 0.02], value=0.005,
        )
    with far:
        cap = st.select_slider(
            "Cap test cases", options=[500, 1000, 2000, 0], value=1000,
            format_func=lambda v: "no cap" if v == 0 else f"{v:,}",
            help="A full run scores every test basket and takes a few minutes.",
        )

    if st.button("Run held-out evaluation", type="primary"):
        st.session_state["rec"] = recommender_evaluation(
            test_fraction, (1, 5, 10), rec_support, cap or None
        )

    if "rec" in st.session_state:
        scores, meta = st.session_state["rec"]
        st.info(
            f"**Train** {meta['train_start']} → {meta['train_end']} "
            f"({meta['n_train']:,} baskets)  ·  "
            f"**Test** {meta['cut_date']} → {meta['test_end']} "
            f"({meta['n_test']:,} baskets). Rules are mined on the training "
            "split only, so no test-period co-occurrence reaches the model."
        )

        at_k = scores.loc[scores["k"] == 5].sort_values("hit_rate@k", ascending=False)
        if not at_k.empty:
            best = at_k.iloc[0]
            metric_cards([
                (f"{best['hit_rate@k']:.1%}", f"Hit rate @5 — {best['model']}",
                 "held-out baskets recovered"),
                (f"{best['vs_popularity']:.1f}×", "vs popularity", "the baseline to beat"),
                (f"{best['MAP@k']:.3f}", "MAP @5", "rewards ranking it high"),
                (f"{best['coverage']:.1%}", "Catalogue coverage", "breadth of what it suggests"),
            ])

        figure = go.Figure()
        for index, model in enumerate(scores["model"].unique()):
            subset = scores.loc[scores["model"] == model].sort_values("k")
            figure.add_trace(
                go.Scatter(
                    x=subset["k"], y=subset["hit_rate@k"], name=model,
                    mode="lines+markers",
                    line=dict(color=CATEGORICAL_DARK[index % len(CATEGORICAL_DARK)], width=2),
                    marker=dict(size=9, line=dict(color=DARK["surface"], width=2)),
                    hovertemplate=f"{model}<br>k=%{{x}}<br>hit rate %{{y:.2%}}<extra></extra>",
                )
            )
        figure.update_layout(**PLOTLY_LAYOUT, height=420,
                             title_text="Hit rate against k — association vs the baselines")
        figure.update_xaxes(title_text="k (recommendations shown)", dtick=1)
        figure.update_yaxes(title_text="Hit rate", tickformat=".0%")
        st.plotly_chart(figure, **FULL_WIDTH)

        st.dataframe(
            scores[[c for c in [
                "model", "k", "n_test_baskets", "hit_rate@k", "precision@k",
                "MAP@k", "MRR", "coverage", "novelty", "vs_popularity",
            ] if c in scores.columns]], **FULL_WIDTH, hide_index=True,
        )
        st.caption(
            "Read coverage and novelty next to accuracy. A model can win on "
            "hit rate while only ever recommending bestsellers — accurate, and "
            "commercially worthless. The association model is the most "
            "accurate here but recommends a narrower slice of the catalogue "
            "than raw co-occurrence does."
        )

# -- Segments ------------------------------------------------------------
with tabs[5]:
    st.caption(
        "A wholesale buyer and a one-time gift buyer are both 'customers'. "
        "Pooled, their rules average into something describing neither. "
        "Customers are clustered on recency, frequency and monetary value, "
        "then each segment is mined separately."
    )
    left, right = st.columns(2)
    with left:
        seg_k = st.slider("Segments (k)", 2, 8, int(cfg.get("analytics.segmentation.k", 4)))
    with right:
        seg_support = st.select_slider(
            "Per-segment support", options=[0.015, 0.02, 0.03, 0.04], value=0.02,
            help="Lower than the global threshold: a segment holds a fraction "
                 "of the baskets.",
        )

    if st.button("Run segmentation", type="primary"):
        st.session_state["seg"] = segmentation(seg_k, seg_support)

    if "seg" in st.session_state:
        profiles, customers, segment_rules, silhouette = st.session_state["seg"]

        metric_cards([
            (f"{len(customers):,}", "Customers", "with an identifiable ID"),
            (f"{len(profiles)}", "Segments", f"silhouette {silhouette:.2f}"),
            (f"{profiles['revenue_share'].max():.0%}", "Top segment's revenue",
             f"from {profiles.loc[profiles['revenue_share'].idxmax(), 'customer_share']:.0%} of customers"),
        ])

        chart = profiles.sort_values("revenue_share")
        figure = go.Figure()
        figure.add_trace(go.Bar(
            x=chart["customer_share"], y=chart["segment"], orientation="h",
            name="share of customers", marker_color=CATEGORICAL_DARK[0],
            hovertemplate="%{y}<br>%{x:.1%} of customers<extra></extra>",
        ))
        figure.add_trace(go.Bar(
            x=chart["revenue_share"], y=chart["segment"], orientation="h",
            name="share of revenue", marker_color=CATEGORICAL_DARK[1],
            hovertemplate="%{y}<br>%{x:.1%} of revenue<extra></extra>",
        ))
        figure.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
                             title_text="Where the customers are, and where the money is")
        figure.update_xaxes(title_text="Share", tickformat=".0%")
        st.plotly_chart(figure, **FULL_WIDTH)
        st.caption(
            "The gap between the two bars is the finding: a small segment "
            "usually carries a disproportionate share of revenue."
        )

        st.subheader("Segment profiles")
        st.dataframe(
            profiles[["segment", "n_customers", "customer_share", "recency",
                      "frequency", "monetary", "avg_basket_value", "revenue_share"]], **FULL_WIDTH, hide_index=True,
        )
        if not segment_rules.empty:
            st.subheader("Strongest rules per segment")
            st.caption("Where these differ between segments, a single global rules table was hiding it.")
            st.dataframe(segment_rules, **FULL_WIDTH, hide_index=True)

# -- Drift ---------------------------------------------------------------
with tabs[6]:
    st.caption(
        "A rule mined over a whole year states that an association held *on "
        "average*. That hides the difference between a stable product fact and "
        "a seasonal one. Each period is mined independently."
    )
    left, right = st.columns(2)
    with left:
        freq = st.radio("Period", ["M", "Q"], horizontal=True,
                        format_func=lambda v: {"M": "Monthly", "Q": "Quarterly"}[v])
    with right:
        drift_support = st.select_slider(
            "Per-period support", options=[0.02, 0.03, 0.04, 0.05], value=0.03
        )

    if st.button("Run drift analysis", type="primary"):
        st.session_state["drift"] = temporal(freq, drift_support)

    if "drift" in st.session_state:
        period_stats, drift, matrix, periods = st.session_state["drift"]

        if not drift.empty:
            counts = drift["rule_class"].value_counts()
            metric_cards([
                (f"{len(periods)}", "Periods mined", "each independently"),
                (f"{len(drift):,}", "Distinct rules", "across all periods"),
                (f"{int(counts.get('stable', 0))}", "Stable", "hold throughout"),
                (f"{int(counts.get('emerging', 0))}", "Emerging", "trending up"),
                (f"{int(counts.get('decaying', 0))}", "Decaying", "trending down"),
                (f"{int(counts.get('seasonal', 0))}", "Seasonal", "high lift variance"),
                (f"{int(counts.get('intermittent', 0))}", "Intermittent",
                 "steady but near the threshold"),
            ])
            if int(counts.get("seasonal", 0)) == 0:
                st.info(
                    "**No rule is statistically seasonal.** No association's lift "
                    "varies enough across periods to clear the threshold — basket "
                    "*volume* here is strongly seasonal while product *affinities* "
                    "are not. Rules present in only part of the year are classed "
                    "`intermittent` instead: they sit near the support threshold "
                    "rather than genuinely coming and going with the season."
                )

        figure = go.Figure()
        figure.add_trace(go.Bar(
            x=period_stats["period"], y=period_stats["baskets"],
            marker_color=[
                STATUS["warning"] if partial else CATEGORICAL_DARK[0]
                for partial in period_stats.get("partial", [False] * len(period_stats))
            ],
            hovertemplate="%{x}<br>%{y:,} baskets<extra></extra>",
        ))
        figure.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                             title_text="Baskets per period (amber = partial period)")
        st.plotly_chart(figure, **FULL_WIDTH)

        if not matrix.empty:
            st.subheader("Lift trajectory of the strongest rules")
            heat = go.Figure(go.Heatmap(
                z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
                colorscale=[[0, "#cde2fb"], [1, "#0d366b"]],
                hovertemplate="%{y}<br>%{x}<br>lift %{z:.1f}<extra></extra>",
                colorbar=dict(title="Lift", outlinewidth=0),
            ))
            heat.update_layout(**PLOTLY_LAYOUT, height=560,
                               title_text="Blank = the rule did not clear the threshold that period")
            st.plotly_chart(heat, **FULL_WIDTH)
            st.caption(
                "A blank cell is not a lift of zero — it means the rule fell "
                "below the support threshold that period, which is itself "
                "information."
            )

        if not drift.empty:
            st.subheader("Rule drift")
            st.dataframe(
                drift.drop(columns=["trajectory"], errors="ignore"), **FULL_WIDTH, height=400, hide_index=True,
            )

# -- Benchmark -----------------------------------------------------------
with tabs[7]:
    st.caption(
        "Both algorithms return identical itemsets, so this is purely about "
        "cost. Apriori pays for candidate generation, which explodes as support "
        "falls; FP-Growth pays a fixed tree-construction cost up front and then "
        "only touches patterns that exist."
    )
    grid = st.multiselect(
        "Support thresholds to sweep",
        [0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.012, 0.01],
        default=[0.05, 0.03, 0.02, 0.015],
    )
    if st.button("Run benchmark", type="primary") and grid:
        from mba.scaling.benchmark import benchmark_algorithms

        with st.spinner("Running both algorithms across the grid…"):
            result = benchmark_algorithms(
                basket, support_grid=sorted(grid, reverse=True), max_len=max_len, cfg=cfg
            )
        st.session_state["benchmark"] = result

    if "benchmark" in st.session_state:
        result = st.session_state["benchmark"]
        wide = result.pivot("runtime_sec")
        figure = go.Figure()
        for index, column in enumerate(wide.columns):
            figure.add_trace(
                go.Scatter(
                    x=wide.index, y=wide[column], name=column, mode="lines+markers",
                    line=dict(color=CATEGORICAL_DARK[index], width=2),
                    marker=dict(size=9, line=dict(color=DARK["surface"], width=2)),
                    hovertemplate=f"{column}<br>support %{{x:.3f}}<br>%{{y:.3f}}s<extra></extra>",
                )
            )
        figure.update_layout(**PLOTLY_LAYOUT, height=480,
                             title_text="Runtime against support threshold")
        figure.update_xaxes(title_text="Minimum support (harder →)", autorange="reversed")
        figure.update_yaxes(title_text="Runtime, seconds (log)", type="log")
        st.plotly_chart(figure, **FULL_WIDTH)

        crossover = result.crossover()
        if crossover is not None:
            st.success(
                f"FP-Growth overtakes Apriori at support ≤ {crossover:.3g}. "
                "Above that, Apriori's candidate sets are small enough that it "
                "wins outright — FP-Growth is still paying to build its tree."
            )
        else:
            st.info(
                "Apriori stayed ahead across this grid. At these thresholds the "
                "frequent-item count is small, so candidate generation is cheap "
                "and FP-Growth's tree construction does not pay for itself. "
                "Extend the grid downward to find the crossover."
            )
        st.dataframe(
            result.frame[[
                "algorithm", "min_support", "runtime_sec", "n_itemsets",
                "peak_memory_mb", "total_candidates",
            ]], **FULL_WIDTH, hide_index=True,
        )
        st.caption(
            "Itemsets identical across algorithms: "
            + ("yes ✅" if result.verify_agreement() else "NO — investigate ⚠️")
        )

# -- Streaming -----------------------------------------------------------
with tabs[8]:
    st.caption(
        "Every other miner here assumes the data is finite and re-readable. "
        "A live checkout feed is neither. Lossy Counting trades a *bounded, "
        "known* amount of accuracy for bounded memory in a single pass."
    )
    left, right = st.columns(2)
    with left:
        stream_support = st.select_slider(
            "Minimum support", options=[0.01, 0.02, 0.03, 0.05], value=0.02
        )
    with right:
        epsilon = st.select_slider(
            "Error tolerance ε", options=[0.0005, 0.001, 0.002, 0.005], value=0.002,
            help="Must sit well below the support threshold. Smaller ε means "
                 "tighter counts and a bigger table.",
        )

    if st.button("Run single pass", type="primary"):
        st.session_state["stream"] = streaming(stream_support, epsilon)

    if "stream" in st.session_state:
        frame, stats, comparison = st.session_state["stream"]
        reported = comparison.loc[comparison["reported"]]

        metric_cards([
            (f"{stats['n_observations']:,}", "Observations", "single pass"),
            (f"{stats['distinct_seen']:,}", "Distinct items", "seen in the stream"),
            (f"{stats['peak_entries']:,}", "Entries held", f"{stats['memory_saving']:.1f}× fewer"),
            (f"±{stats['max_count_error']:,.0f}", "Guaranteed error", "worst-case undercount"),
        ])

        st.subheader("The guarantees, checked against an exact count")
        false_negatives = int(comparison["false_negative"].sum())
        overestimates = bool((reported["error"] < 0).any())
        worst = int(reported["error"].max()) if not reported.empty else 0
        checks = pd.DataFrame([
            {"guarantee": "No false negatives",
             "expected": "0", "observed": f"{false_negatives}",
             "holds": "✅" if false_negatives == 0 else "❌"},
            {"guarantee": "Counts never overestimate",
             "expected": "False", "observed": str(overestimates),
             "holds": "✅" if not overestimates else "❌"},
            {"guarantee": f"Undercount ≤ ε·N ({stats['max_count_error']:,.0f})",
             "expected": f"≤ {stats['max_count_error']:,.0f}", "observed": f"{worst}",
             "holds": "✅" if worst <= stats["max_count_error"] else "❌"},
            {"guarantee": "Table within the space bound",
             "expected": f"≤ {stats['space_bound']:,.0f}",
             "observed": f"{stats['peak_entries']:,}",
             "holds": "✅" if stats["within_space_bound"] else "❌"},
        ])
        st.dataframe(checks, **FULL_WIDTH, hide_index=True)
        st.caption(
            "The classical (1/ε)·log(εN) bound assumes one item per "
            "observation. A basket inserts ~24 at once, so the applicable "
            "bound is scaled — reporting the unscaled one here would claim a "
            "guarantee the algorithm does not give in this setting."
        )

        if not frame.empty:
            st.subheader(f"Frequent items ({len(frame):,} above threshold)")
            st.dataframe(frame.head(30), **FULL_WIDTH, hide_index=True)

# -- Insights ------------------------------------------------------------
with tabs[9]:
    insights = generate_insights(
        rules, cfg, frame=frame, n_transactions=basket.n_transactions,
        graph_summary=graph_summary(network_for(country, 34, 1.5)[0]),
    )
    if not insights:
        st.info("No insights cleared the thresholds. Relax the sliders in the sidebar.")
    for insight in insights:
        with st.container(border=True):
            st.markdown(f"**{insight.title}**")
            st.caption(f"`{insight.kind}` · priority {insight.priority:.1f}")
            st.write(insight.detail)
            st.markdown(f"**Action:** {insight.action}")
            if insight.estimated_value:
                st.caption(
                    f"Indicative annual value £{insight.estimated_value:,.0f} "
                    "(15% attach-rate assumption — a sizing aid, not a forecast)"
                )

# -- Data quality --------------------------------------------------------
with tabs[10]:
    st.subheader("Rule quality triage")
    if flags is not None:
        st.write(flags.summary())
        counts = pd.DataFrame(
            sorted(flags.counts.items(), key=lambda kv: -kv[1]),
            columns=["reason", "rules flagged"],
        )
        figure = go.Figure(
            go.Bar(
                x=counts["rules flagged"], y=counts["reason"], orientation="h",
                marker_color=STATUS["warning"],
                hovertemplate="%{y}: %{x} rules<extra></extra>",
            )
        )
        figure.update_layout(**PLOTLY_LAYOUT, height=340,
                             title_text="Why rules were flagged")
        st.plotly_chart(figure, **FULL_WIDTH)

        st.markdown(
            """
A rule can clear every threshold and still be worthless:

- **rare_pair** — few baskets behind it; lift is a ratio of small numbers.
- **ubiquitous_consequent** — the consequent is in most baskets anyway.
- **negligible_leverage** — high lift, negligible absolute co-occurrence.
- **weak_association** — Zhang's metric near zero; near independence once skew is accounted for.
- **not_significant** — fails a chi-squared independence test.
- **redundant** — a simpler rule already does as well.
- **self_referential** — colour or size variants of the same product. Statistically strong, commercially empty.
            """
        )
        st.subheader("Flagged rules")
        flagged = rules.loc[rules["is_spurious"]] if "is_spurious" in rules.columns else rules.head(0)
        if not flagged.empty:
            st.dataframe(
                format_rules(flagged.nlargest(50, "lift"))[
                    ["rule", "support", "confidence", "lift", "count", "flag_reasons"]
                ], **FULL_WIDTH, hide_index=True,
            )

    st.divider()
    st.subheader("Cleaning audit trail")
    st.caption("Every row dropped between the raw file and the mined matrix.")
    raw_rows = load_raw(cfg)
    _, report = clean_transactions(raw_rows, cfg)
    st.dataframe(report.to_frame(), **FULL_WIDTH, hide_index=True)
    st.write(report.summary())
