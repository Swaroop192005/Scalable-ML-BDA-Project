"""Configuration loading, dotted access and CLI overrides."""

import pytest

from mba.config import Config, ConfigError, apply_overrides, load_config


def test_loads_every_expected_section(cfg):
    for section in ("project", "paths", "schema", "preprocess", "basket",
                    "mining", "spurious", "scaling", "analytics", "viz", "dashboard"):
        assert section in cfg, f"missing config section: {section}"


def test_three_access_styles_agree(cfg):
    assert cfg["mining"]["min_support"] == cfg.get("mining.min_support")
    assert cfg.mining.min_support == cfg.get("mining.min_support")


def test_get_returns_default_for_missing_key(cfg):
    assert cfg.get("nope.not.here", "fallback") == "fallback"


def test_require_raises_for_missing_key(cfg):
    with pytest.raises(ConfigError):
        cfg.require("nope.not.here")


def test_paths_resolve_against_project_root(cfg):
    assert cfg.path("paths.reports_dir").is_absolute()


@pytest.mark.parametrize(
    "override, expected",
    [
        ("mining.min_support=0.05", 0.05),
        ("mining.max_len=4", 4),
        ("preprocess.drop_cancellations=false", False),
        ("mining.algorithm=apriori", "apriori"),
    ],
)
def test_override_coerces_types(override, expected):
    cfg = load_config(overrides=[override])
    assert cfg.get(override.split("=")[0]) == expected


def test_override_can_create_a_nested_key():
    cfg = load_config(overrides=["brand.new.key=7"])
    assert cfg.get("brand.new.key") == 7


def test_malformed_override_raises():
    with pytest.raises(ConfigError):
        apply_overrides(Config({}), ["no_equals_sign"])


def test_to_dict_is_a_deep_copy(cfg):
    snapshot = cfg.to_dict()
    snapshot["mining"]["min_support"] = 999
    assert cfg.get("mining.min_support") != 999
