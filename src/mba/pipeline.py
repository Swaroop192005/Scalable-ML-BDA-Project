"""The end-to-end pipeline: raw file in, reports and figures out.

One function, :func:`run_pipeline`, wires every stage together and returns an
:class:`PipelineArtifacts` bundle. The CLI, the test suite and the dashboard
all go through it, so there is exactly one definition of "the analysis".
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mba.analytics.eda import EDAReport, run_eda
from mba.analytics.insights import Insight, generate_insights, insights_to_frame, insights_to_markdown
from mba.analytics.network import build_cooccurrence_graph, graph_summary
from mba.analytics.regional import RegionResult, compare_regions, distinctive_rules, regional_analysis
from mba.config import Config
from mba.data.basket import BasketMatrix, encode_baskets
from mba.data.loader import load_raw, save_processed
from mba.data.preprocess import CleaningReport, clean_transactions
from mba.mining.base import MiningResult, mine
from mba.mining.rules import association_rules, format_rules
from mba.mining.spurious import SpuriousFlags, flag_spurious_rules

log = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    """Everything a run produces, in memory."""

    clean: pd.DataFrame
    basket: BasketMatrix
    cleaning_report: CleaningReport
    eda: EDAReport
    itemsets: MiningResult
    rules: pd.DataFrame
    spurious: SpuriousFlags | None = None
    graph: Any = None
    graph_stats: dict[str, Any] = field(default_factory=dict)
    regions: dict[str, RegionResult] = field(default_factory=dict)
    region_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    distinctive: pd.DataFrame = field(default_factory=pd.DataFrame)
    insights: list[Insight] = field(default_factory=list)
    benchmark: pd.DataFrame = field(default_factory=pd.DataFrame)
    scaling: pd.DataFrame = field(default_factory=pd.DataFrame)
    timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        high_value = (
            int((~self.rules["is_spurious"]).sum())
            if "is_spurious" in self.rules.columns
            else len(self.rules)
        )
        return (
            f"{self.basket.n_transactions:,} baskets | {self.basket.n_items:,} items | "
            f"{len(self.itemsets):,} itemsets | {len(self.rules):,} rules "
            f"({high_value:,} high value) | {len(self.insights)} insights | "
            f"{sum(self.timings.values()):.1f}s total"
        )


class _Stage:
    """Times a stage and logs a banner so long runs are legible."""

    def __init__(self, name: str, timings: dict[str, float]) -> None:
        self.name, self.timings = name, timings

    def __enter__(self):
        self._t0 = time.perf_counter()
        log.info("=" * 68)
        log.info(">> %s", self.name)
        return self

    def __exit__(self, *exc):
        self.timings[self.name] = time.perf_counter() - self._t0
        log.info("   %s finished in %.2fs", self.name, self.timings[self.name])


def run_pipeline(
    cfg: Config,
    source: str | Path | None = None,
    nrows: int | None = None,
    with_regional: bool = True,
    with_network: bool = True,
    with_benchmark: bool = False,
    with_scaling: bool = False,
    save_outputs: bool = True,
    save_figures: bool = True,
) -> PipelineArtifacts:
    """Run every stage and (optionally) write the report bundle to disk."""
    timings: dict[str, float] = {}

    with _Stage("load", timings):
        raw = load_raw(cfg, path=source, nrows=nrows)

    with _Stage("clean", timings):
        clean, cleaning_report = clean_transactions(raw, cfg)
        del raw

    with _Stage("encode", timings):
        basket = encode_baskets(clean, cfg)

    with _Stage("eda", timings):
        eda = run_eda(clean, cfg, basket)

    with _Stage("mine", timings):
        itemsets = mine(
            basket,
            algorithm=cfg.get("mining.algorithm", "fpgrowth"),
            min_support=float(cfg.get("mining.min_support", 0.02)),
            max_len=int(cfg.get("mining.max_len", 3)),
            verbose=True,
        )

    with _Stage("rules", timings):
        rules = association_rules(
            itemsets,
            min_confidence=float(cfg.get("mining.min_confidence", 0.3)),
            min_lift=float(cfg.get("mining.min_lift", 1.0)),
            metrics=list(cfg.get("mining.metrics", []) or []) or None,
            sort_by=cfg.get("mining.sort_by", "lift"),
        )
        spurious = None
        if not rules.empty:
            rules, spurious = flag_spurious_rules(rules, cfg, basket.n_transactions)

    graph, stats = None, {}
    if with_network:
        with _Stage("network", timings):
            graph = build_cooccurrence_graph(basket, cfg, weight="lift", min_edge_weight=1.5)
            stats = graph_summary(graph)

    regions: dict[str, RegionResult] = {}
    comparison = pd.DataFrame()
    distinctive = pd.DataFrame()
    if with_regional:
        with _Stage("regional", timings):
            regions = regional_analysis(clean, cfg)
            comparison = compare_regions(regions)
            distinctive = distinctive_rules(regions)

    with _Stage("insights", timings):
        insights = generate_insights(
            rules, cfg, frame=clean, n_transactions=basket.n_transactions,
            graph_summary=stats, regional=distinctive,
        )

    benchmark = pd.DataFrame()
    if with_benchmark:
        with _Stage("benchmark", timings):
            from mba.scaling.benchmark import benchmark_algorithms

            benchmark = benchmark_algorithms(
                basket,
                cfg=cfg,
                algorithms=list(cfg.get("scaling.benchmark.algorithms", ["apriori", "fpgrowth"])),
                max_len=int(cfg.get("scaling.benchmark.max_len", 3)),
                repeats=int(cfg.get("scaling.benchmark.repeats", 1)),
            ).frame

    scaling = pd.DataFrame()
    if with_scaling:
        with _Stage("scaling", timings):
            from mba.scaling.benchmark import scaling_curve

            scaling = scaling_curve(
                basket,
                algorithm=cfg.get("mining.algorithm", "fpgrowth"),
                min_support=float(cfg.get("mining.min_support", 0.02)),
                max_len=int(cfg.get("mining.max_len", 3)),
            )

    artifacts = PipelineArtifacts(
        clean=clean, basket=basket, cleaning_report=cleaning_report, eda=eda,
        itemsets=itemsets, rules=rules, spurious=spurious, graph=graph,
        graph_stats=stats, regions=regions, region_comparison=comparison,
        distinctive=distinctive, insights=insights, benchmark=benchmark,
        scaling=scaling, timings=timings,
    )

    if save_outputs:
        with _Stage("write_outputs", timings):
            write_outputs(artifacts, cfg, save_figures=save_figures)

    log.info("=" * 68)
    log.info("PIPELINE COMPLETE: %s", artifacts.summary())
    return artifacts


def write_outputs(
    artifacts: PipelineArtifacts,
    cfg: Config,
    save_figures: bool = True,
) -> dict[str, Path]:
    """Persist tables, figures, the insight report and a run manifest."""
    reports_dir = cfg.path("paths.reports_dir")
    figures_dir = cfg.path("paths.figures_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def dump(name: str, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        target = reports_dir / f"{name}.csv"
        frame.to_csv(target, index=False)
        written[name] = target

    # Processed transaction table, in both formats.
    for fmt, path in save_processed(artifacts.clean, cfg).items():
        written[f"processed_{fmt}"] = path

    dump("cleaning_report", artifacts.cleaning_report.to_frame())
    dump("frequent_itemsets", artifacts.itemsets.to_display_frame())
    if not artifacts.rules.empty:
        dump("association_rules", format_rules(artifacts.rules))
        high_value = artifacts.rules
        if "is_spurious" in high_value.columns:
            dump("rules_high_value", format_rules(high_value.loc[~high_value["is_spurious"]]))
            dump("rules_flagged_spurious", format_rules(high_value.loc[high_value["is_spurious"]]))
    dump("eda_top_products_volume", artifacts.eda.tables.get("top_products_volume"))
    dump("eda_top_products_revenue", artifacts.eda.tables.get("top_products_revenue"))
    dump("eda_country_stats", artifacts.eda.tables.get("top_countries"))
    dump("regional_comparison", artifacts.region_comparison)
    dump("regional_distinctive_rules", artifacts.distinctive)
    dump("benchmark_apriori_vs_fpgrowth", artifacts.benchmark)
    dump("scaling_curve", artifacts.scaling)
    dump("insights", insights_to_frame(artifacts.insights))
    if artifacts.graph_stats.get("centrality_table") is not None:
        dump("network_centrality", artifacts.graph_stats["centrality_table"])

    # Per-country rules, concatenated.
    regional_rules = [r.rules for r in artifacts.regions.values() if not r.rules.empty]
    if regional_rules:
        dump("regional_rules", format_rules(pd.concat(regional_rules, ignore_index=True)))

    # Narrative insight report.
    insight_md = reports_dir / "INSIGHTS.md"
    insight_md.write_text(
        insights_to_markdown(artifacts.insights, "Market Basket Analysis — Business Insights"),
        encoding="utf-8",
    )
    written["insights_md"] = insight_md

    # Machine-readable manifest of the run.
    manifest = {
        "config": cfg.to_dict(),
        "summary": artifacts.summary(),
        "eda_scalars": artifacts.eda.scalars,
        "cleaning": {
            "rows_in": artifacts.cleaning_report.rows_in,
            "rows_out": artifacts.cleaning_report.rows_out,
            "retention_rate": artifacts.cleaning_report.retention_rate,
            "steps": artifacts.cleaning_report.steps,
        },
        "basket": artifacts.basket.describe(),
        "mining": artifacts.itemsets.stats.to_dict(),
        "rules": {
            "n_rules": len(artifacts.rules),
            "spurious": artifacts.spurious.to_dict() if artifacts.spurious else None,
        },
        "network": {k: v for k, v in artifacts.graph_stats.items() if k != "centrality_table"},
        "regions": {
            c: {"transactions": r.n_transactions, "rules": len(r.rules)}
            for c, r in artifacts.regions.items()
        },
        "timings_sec": {k: round(v, 3) for k, v in artifacts.timings.items()},
    }
    manifest_path = reports_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    written["manifest"] = manifest_path

    if save_figures:
        from mba.viz.plots import save_all_figures

        figures = save_all_figures(
            {
                "eda": artifacts.eda,
                "rules": artifacts.rules,
                "graph": artifacts.graph,
                "benchmark": artifacts.benchmark,
                "scaling": artifacts.scaling,
            },
            figures_dir,
        )
        written.update({f"figure_{k}": v for k, v in figures.items()})

    log.info("wrote %d output files under %s", len(written), reports_dir)
    return written
