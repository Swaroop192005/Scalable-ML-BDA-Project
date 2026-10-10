"""End-to-end pipeline, CLI wiring and the report bundle."""

import json

import pytest

from mba.cli import build_parser, main
from mba.config import load_config
from mba.pipeline import run_pipeline, write_outputs


@pytest.fixture
def toy_cfg(tmp_path, toy_frame, cfg):
    """A config pointed at a temporary copy of the toy dataset."""
    source = tmp_path / "toy.csv"
    # Repeat the toy data so supports are stable and rules clear thresholds.
    import pandas as pd

    frames = []
    for repeat in range(40):
        copy = toy_frame.copy()
        copy["InvoiceNo"] = copy["InvoiceNo"].astype(str) + f"-{repeat}"
        frames.append(copy)
    pd.concat(frames, ignore_index=True).to_csv(source, index=False)

    overrides = [
        f"paths.raw_file={source}",
        f"paths.reports_dir={tmp_path / 'reports'}",
        f"paths.figures_dir={tmp_path / 'reports' / 'figures'}",
        f"paths.processed_dir={tmp_path / 'processed'}",
        "paths.raw_file_fallbacks=[]",
        "mining.min_support=0.1",
        "mining.min_confidence=0.3",
        "mining.min_lift=1.0",
        "analytics.regional.min_transactions=10",
        "analytics.regional.min_support=0.1",
    ]
    return load_config(overrides=overrides), tmp_path


def test_pipeline_runs_end_to_end(toy_cfg):
    config, _ = toy_cfg
    artifacts = run_pipeline(config, save_outputs=False, save_figures=False)

    assert artifacts.basket.n_transactions > 0
    assert len(artifacts.itemsets) > 0
    assert not artifacts.rules.empty
    assert artifacts.cleaning_report.rows_out > 0
    assert "load" in artifacts.timings and "mine" in artifacts.timings
    assert "baskets" in artifacts.summary()


def test_pipeline_writes_the_report_bundle(toy_cfg):
    config, tmp_path = toy_cfg
    artifacts = run_pipeline(config, save_outputs=False, save_figures=False)
    written = write_outputs(artifacts, config, save_figures=False)

    reports = tmp_path / "reports"
    assert (reports / "association_rules.csv").exists()
    assert (reports / "frequent_itemsets.csv").exists()
    assert (reports / "cleaning_report.csv").exists()
    assert (reports / "INSIGHTS.md").exists()

    manifest = json.loads((reports / "run_manifest.json").read_text())
    assert manifest["mining"]["algorithm"] in {"apriori", "fpgrowth"}
    assert manifest["cleaning"]["rows_out"] > 0
    assert "config" in manifest and "timings_sec" in manifest
    assert written


def test_pipeline_stages_can_be_disabled(toy_cfg):
    config, _ = toy_cfg
    artifacts = run_pipeline(
        config, with_regional=False, with_network=False,
        save_outputs=False, save_figures=False,
    )
    assert artifacts.graph is None
    assert artifacts.regions == {}
    assert "network" not in artifacts.timings


def test_figures_are_rendered(toy_cfg):
    config, tmp_path = toy_cfg
    artifacts = run_pipeline(config, save_outputs=False, save_figures=False)
    write_outputs(artifacts, config, save_figures=True)
    figures = list((tmp_path / "reports" / "figures").glob("*.png"))
    assert figures, "no figures were produced"


def test_both_algorithms_give_the_same_pipeline_rules(toy_cfg):
    config, _ = toy_cfg
    config.set("mining.algorithm", "apriori")
    with_apriori = run_pipeline(config, with_regional=False, with_network=False,
                                save_outputs=False, save_figures=False)
    config.set("mining.algorithm", "fpgrowth")
    with_fpgrowth = run_pipeline(config, with_regional=False, with_network=False,
                                 save_outputs=False, save_figures=False)
    assert len(with_apriori.rules) == len(with_fpgrowth.rules)


# --------------------------------------------------------------------- CLI
def test_parser_exposes_every_subcommand():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    expected = {"run", "eda", "mine", "bench", "ooc", "spark", "regional",
                "dashboard", "sample"}
    assert expected <= set(actions[0].choices)


def test_cli_mine_runs(toy_cfg, capsys):
    config, tmp_path = toy_cfg
    source = config.get("paths.raw_file")
    code = main([
        "mine", "-i", str(source), "-s", "0.1", "--min-confidence", "0.3",
        "--min-lift", "1.0", "-q",
    ])
    assert code == 0
    assert "FREQUENT ITEMSETS" in capsys.readouterr().out


def test_cli_eda_runs(toy_cfg, capsys):
    config, _ = toy_cfg
    code = main(["eda", "-i", str(config.get("paths.raw_file")), "-q"])
    assert code == 0
    assert "DATASET" in capsys.readouterr().out


def test_cli_reports_a_missing_input_file(capsys):
    code = main(["eda", "-i", "/nonexistent/file.csv", "-q"])
    assert code != 0


def test_cli_override_changes_behaviour(toy_cfg, capsys):
    config, _ = toy_cfg
    code = main([
        "mine", "-i", str(config.get("paths.raw_file")),
        "--set", "mining.algorithm=apriori", "-s", "0.2", "-q",
    ])
    assert code == 0
