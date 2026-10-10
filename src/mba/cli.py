"""Command-line interface for the whole platform.

    mba run          full pipeline: clean -> mine -> rules -> insights -> report
    mba eda          exploratory analysis only
    mba mine         frequent itemsets + rules, printed
    mba bench        Apriori vs FP-Growth across a support grid
    mba ooc          out-of-core (SON) mining, with a synthetic load-test mode
    mba spark        distributed FP-Growth (or the architecture notes)
    mba regional     per-country analysis
    mba dashboard    launch the Streamlit app
    mba sample       write a small committed sample dataset

Every subcommand accepts ``--set key.path=value`` to override any key in
``config/config.yaml`` without editing the file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mba import __version__
from mba.config import PROJECT_ROOT, Config, load_config


def _setup_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # These are noisy at DEBUG and never interesting here.
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# ---------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    from mba.pipeline import run_pipeline

    artifacts = run_pipeline(
        cfg,
        source=args.input,
        nrows=args.nrows,
        with_regional=not args.no_regional,
        with_network=not args.no_network,
        with_benchmark=args.benchmark,
        with_scaling=args.scaling,
        save_outputs=not args.no_save,
        save_figures=not args.no_figures,
    )

    _banner("RESULTS")
    print(artifacts.summary())

    if artifacts.insights:
        _banner("TOP BUSINESS INSIGHTS")
        for i, insight in enumerate(artifacts.insights[:5], 1):
            print(f"\n{i}. [{insight.kind}] {insight.title}")
            print(f"   {insight.detail}")
            print(f"   -> {insight.action}")

    if not args.no_save:
        print(f"\nReports written to {cfg.path('paths.reports_dir')}")
        print(f"Figures written to {cfg.path('paths.figures_dir')}")
    return 0


def cmd_eda(args: argparse.Namespace, cfg: Config) -> int:
    from mba.analytics.eda import run_eda
    from mba.data import clean_transactions, encode_baskets, load_raw

    raw = load_raw(cfg, path=args.input, nrows=args.nrows)
    clean, report = clean_transactions(raw, cfg)
    basket = encode_baskets(clean, cfg)
    eda = run_eda(clean, cfg, basket)

    _banner("CLEANING")
    print(report.to_frame().to_string(index=False))
    _banner("DATASET")
    print(eda.summary())
    for key, value in eda.scalars.items():
        print(f"  {key:<42s} {value}")
    _banner("TOP PRODUCTS BY BASKET COUNT")
    print(eda["top_products_volume"].to_string(index=False))
    if "top_countries" in eda.tables:
        _banner("TOP COUNTRIES")
        print(eda["top_countries"].to_string(index=False))
    return 0


def cmd_mine(args: argparse.Namespace, cfg: Config) -> int:
    from mba.data import clean_transactions, encode_baskets, load_raw
    from mba.mining.base import mine
    from mba.mining.rules import association_rules, format_rules
    from mba.mining.spurious import flag_spurious_rules

    raw = load_raw(cfg, path=args.input, nrows=args.nrows)
    clean, _ = clean_transactions(raw, cfg)
    basket = encode_baskets(clean, cfg)

    result = mine(
        basket,
        algorithm=args.algorithm or cfg.get("mining.algorithm", "fpgrowth"),
        min_support=args.min_support,
        max_len=args.max_len,
        verbose=True,
    )
    _banner(f"FREQUENT ITEMSETS — {result.stats.algorithm}")
    print(result.stats.summary())
    print("\nitemsets per size:")
    print(result.by_length().to_string())

    rules = association_rules(
        result, min_confidence=args.min_confidence, min_lift=args.min_lift, sort_by="lift"
    )
    if rules.empty:
        print("\nNo rules cleared the thresholds. Lower --min-support or --min-confidence.")
        return 0

    rules, flags = flag_spurious_rules(rules, cfg, basket.n_transactions)
    _banner("RULE QUALITY")
    print(flags.summary())

    columns = ["rule", "support", "confidence", "lift", "leverage", "conviction", "count"]
    high_value = rules.loc[~rules["is_spurious"]].nlargest(args.top, "lift")
    _banner(f"TOP {len(high_value)} HIGH-VALUE RULES")
    print(format_rules(high_value)[columns].to_string(index=False))

    flagged = rules.loc[rules["is_spurious"]].nlargest(min(args.top, 10), "lift")
    if not flagged.empty:
        _banner("FLAGGED AS MISLEADING (high metrics, low value)")
        print(
            format_rules(flagged)[
                ["rule", "support", "confidence", "lift", "count", "flag_reasons"]
            ].to_string(index=False)
        )
    return 0


def cmd_bench(args: argparse.Namespace, cfg: Config) -> int:
    from mba.data import clean_transactions, encode_baskets, load_raw
    from mba.scaling.benchmark import benchmark_algorithms, scaling_curve
    from mba.viz.plots import plot_benchmark, plot_scaling_curve

    raw = load_raw(cfg, path=args.input, nrows=args.nrows)
    clean, _ = clean_transactions(raw, cfg)
    basket = encode_baskets(clean, cfg)

    grid = args.support_grid or cfg.get("scaling.benchmark.support_grid")
    result = benchmark_algorithms(
        basket, support_grid=grid, max_len=args.max_len, repeats=args.repeats, cfg=cfg
    )

    _banner("BENCHMARK: APRIORI vs FP-GROWTH")
    print(result.frame[
        ["algorithm", "min_support", "runtime_sec", "n_itemsets", "peak_memory_mb", "total_candidates"]
    ].to_string(index=False))
    print()
    print(result.summary())

    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(reports / "benchmark_apriori_vs_fpgrowth.csv", index=False)
    plot_benchmark(result.frame, cfg.path("paths.figures_dir") / "09_benchmark_apriori_vs_fpgrowth.png")

    if args.scaling:
        curve = scaling_curve(
            basket, algorithm=cfg.get("mining.algorithm", "fpgrowth"),
            min_support=float(cfg.get("mining.min_support", 0.02)),
        )
        _banner("SCALING IN TRANSACTION VOLUME")
        print(curve.to_string(index=False))
        curve.to_csv(reports / "scaling_curve.csv", index=False)
        plot_scaling_curve(curve, cfg.path("paths.figures_dir") / "10_scaling_curve.png")

    print(f"\nWritten to {reports}")
    return 0


def cmd_ooc(args: argparse.Namespace, cfg: Config) -> int:
    from mba.scaling.outofcore import SyntheticTransactions, mine_out_of_core, peak_rss_mb

    if args.synthetic:
        print(
            f"Streaming {args.synthetic:,} synthetic transactions "
            f"({args.synthetic_items:,} item catalogue) — generated lazily, "
            "never held in memory as a whole."
        )
        transactions = SyntheticTransactions(
            n_transactions=args.synthetic, n_items=args.synthetic_items
        )
    else:
        from mba.data import clean_transactions, load_raw
        from mba.data.basket import transactions_from_frame

        raw = load_raw(cfg, path=args.input, nrows=args.nrows)
        clean, _ = clean_transactions(raw, cfg)
        transactions = transactions_from_frame(
            clean, cfg["schema"]["invoice_col"], cfg["schema"]["item_col"]
        )

    result = mine_out_of_core(
        transactions,
        min_support=args.min_support,
        max_len=args.max_len,
        chunk_size=args.chunk_size,
        memory_budget_mb=args.memory_budget,
        algorithm=cfg.get("mining.algorithm", "fpgrowth"),
    )

    _banner("OUT-OF-CORE MINING (SON, two streaming passes)")
    print(result.summary())
    print("\nper-chunk profile:")
    print(result.chunk_profile.to_string(index=False))
    print(f"\nprocess peak RSS: {peak_rss_mb():.0f} MB (budget {args.memory_budget:.0f} MB)")
    if not result.itemsets.empty:
        print("\ntop itemsets:")
        top = result.as_mining_result().top(15).copy()
        top["itemsets"] = top["itemsets"].map(lambda s: ", ".join(sorted(s)))
        print(top.to_string(index=False))
    return 0


def cmd_spark(args: argparse.Namespace, cfg: Config) -> int:
    from mba.scaling.spark_fpgrowth import ARCHITECTURE_NOTES, run_spark_fpgrowth, spark_available

    if not spark_available():
        _banner("PYSPARK NOT INSTALLED — ARCHITECTURE REFERENCE")
        print(ARCHITECTURE_NOTES)
        print(
            "To run it: pip install -r requirements-optional.txt  (a JVM is also required)\n"
            "Until then, `mba ooc` handles datasets beyond RAM on a single node."
        )
        return 0

    from mba.data import clean_transactions, load_raw
    from mba.data.basket import transactions_from_frame

    raw = load_raw(cfg, path=args.input, nrows=args.nrows)
    clean, _ = clean_transactions(raw, cfg)
    transactions = transactions_from_frame(
        clean, cfg["schema"]["invoice_col"], cfg["schema"]["item_col"]
    )

    output = run_spark_fpgrowth(
        transactions, cfg=cfg, min_support=args.min_support, min_confidence=args.min_confidence
    )
    _banner("SPARK FP-GROWTH")
    print(f"transactions: {output['n_transactions']:,}")
    print(f"itemsets:     {len(output['itemsets']):,}")
    print(f"rules:        {len(output['rules']):,}")
    if not output["rules"].empty:
        print(output["rules"].head(20).to_string(index=False))
    return 0


def cmd_regional(args: argparse.Namespace, cfg: Config) -> int:
    from mba.analytics.regional import compare_regions, distinctive_rules, regional_analysis
    from mba.data import clean_transactions, load_raw

    raw = load_raw(cfg, path=args.input, nrows=args.nrows)
    clean, _ = clean_transactions(raw, cfg)
    results = regional_analysis(clean, cfg, countries=args.countries)

    _banner("REGIONAL MARKET BASKET ANALYSIS")
    comparison = compare_regions(results)
    print(comparison.to_string(index=False))

    distinctive = distinctive_rules(results, reference=args.reference)
    if not distinctive.empty:
        _banner(f"RULES MUCH STRONGER THAN IN {args.reference.upper()}")
        print(distinctive.to_string(index=False))
    return 0


def cmd_dashboard(args: argparse.Namespace, cfg: Config) -> int:
    import subprocess

    app = PROJECT_ROOT / "dashboard" / "app.py"
    if not app.exists():
        print(f"dashboard not found at {app}", file=sys.stderr)
        return 1
    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.port", str(args.port),
        "--server.address", args.address,
    ]
    print(f"Launching dashboard: http://{args.address}:{args.port}")
    return subprocess.call(command)


def cmd_sample(args: argparse.Namespace, cfg: Config) -> int:
    """Write a small, committable sample so the repo is runnable without a download."""
    from mba.data import clean_transactions, load_raw

    raw = load_raw(cfg, path=args.input)
    clean, _ = clean_transactions(raw, cfg)

    invoice_col = cfg["schema"]["invoice_col"]
    invoices = clean[invoice_col].drop_duplicates()
    chosen = invoices.sample(
        n=min(args.n_invoices, len(invoices)), random_state=int(cfg.get("project.random_seed", 42))
    )
    sample = clean.loc[clean[invoice_col].isin(set(chosen))]

    target = Path(args.output) if args.output else cfg.path("paths.sample_dir") / "sample_retail.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(target, index=False)
    print(
        f"Wrote {len(sample):,} rows across {sample[invoice_col].nunique():,} invoices "
        f"to {target} ({target.stat().st_size / 1e6:.2f} MB)"
    )
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mba",
        description="Scalable Market Basket Analysis — end-to-end pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"mba {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", help="path to config.yaml")
    common.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="override any config key, e.g. --set mining.min_support=0.01",
    )
    common.add_argument("-i", "--input", help="override the raw input file")
    common.add_argument("--nrows", type=int, help="read only the first N rows")
    common.add_argument("-v", "--verbose", action="count", default=1,
                        help="-v info (default), -vv debug, use -q for warnings only")
    common.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", parents=[common], help="run the full pipeline")
    p.add_argument("--benchmark", action="store_true", help="also benchmark the two algorithms")
    p.add_argument("--scaling", action="store_true", help="also compute the scaling curve")
    p.add_argument("--no-regional", action="store_true")
    p.add_argument("--no-network", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-save", action="store_true", help="print only, write nothing")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("eda", parents=[common], help="exploratory data analysis")
    p.set_defaults(func=cmd_eda)

    p = sub.add_parser("mine", parents=[common], help="mine itemsets and rules")
    p.add_argument("-a", "--algorithm", choices=["apriori", "fpgrowth"])
    p.add_argument("-s", "--min-support", type=float, default=0.02)
    p.add_argument("--min-confidence", type=float, default=0.30)
    p.add_argument("--min-lift", type=float, default=1.20)
    p.add_argument("-l", "--max-len", type=int, default=3)
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_mine)

    p = sub.add_parser("bench", parents=[common], help="Apriori vs FP-Growth")
    p.add_argument("-g", "--support-grid", type=float, nargs="+")
    p.add_argument("-l", "--max-len", type=int, default=3)
    p.add_argument("-r", "--repeats", type=int, default=1)
    p.add_argument("--scaling", action="store_true", help="also sweep transaction volume")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("ooc", parents=[common], help="out-of-core (SON) mining")
    p.add_argument("-s", "--min-support", type=float, default=0.02)
    p.add_argument("-l", "--max-len", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=50_000)
    p.add_argument("--memory-budget", type=float, default=800.0, help="hard RSS ceiling in MB")
    p.add_argument("--synthetic", type=int, metavar="N",
                   help="load-test with N synthetic transactions instead of the real file")
    p.add_argument("--synthetic-items", type=int, default=5000)
    p.set_defaults(func=cmd_ooc)

    p = sub.add_parser("spark", parents=[common], help="distributed FP-Growth")
    p.add_argument("-s", "--min-support", type=float, default=0.02)
    p.add_argument("--min-confidence", type=float, default=0.30)
    p.set_defaults(func=cmd_spark)

    p = sub.add_parser("regional", parents=[common], help="per-country analysis")
    p.add_argument("--countries", nargs="+")
    p.add_argument("--reference", default="United Kingdom")
    p.set_defaults(func=cmd_regional)

    p = sub.add_parser("dashboard", parents=[common], help="launch the Streamlit app")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--address", default="localhost")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("sample", parents=[common], help="write a small sample dataset")
    p.add_argument("-n", "--n-invoices", type=int, default=3000)
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_sample)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(0 if args.quiet else args.verbose)

    try:
        cfg = load_config(args.config, args.overrides)
    except Exception as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        return args.func(args, cfg)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
