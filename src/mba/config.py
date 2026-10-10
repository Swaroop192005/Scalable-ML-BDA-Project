"""YAML-backed configuration with dotted access and CLI overrides.

The whole pipeline is parameterised from ``config/config.yaml``. This module
loads it once, wraps it in a small mapping that supports dotted lookups, and
gives the CLI a way to override any leaf with ``--set key.path=value``.
"""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

#: Repository root — two levels up from ``src/mba/config.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or malformed."""


class Config(Mapping):
    """Immutable-ish view over the nested config dictionary.

    Supports three access styles so call sites stay readable::

        cfg["mining"]["min_support"]
        cfg.get("mining.min_support")
        cfg.mining.min_support
    """

    __slots__ = ("_data", "_path")

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        if not isinstance(data, dict):
            raise ConfigError(f"configuration root must be a mapping, got {type(data)!r}")
        self._data = data
        self._path = path

    # -- Mapping protocol ------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        where = f" from {self._path}" if self._path else ""
        return f"<Config{where} keys={sorted(self._data)}>"

    # -- Dotted access ---------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - attribute error semantics
            raise AttributeError(name) from exc

    def get(self, dotted: str, default: Any = None) -> Any:
        """Return ``cfg['a']['b']['c']`` for ``get('a.b.c')``, else *default*."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def require(self, dotted: str) -> Any:
        """Like :meth:`get` but raises when the key is absent."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required configuration key: {dotted!r}")
        return value

    def set(self, dotted: str, value: Any) -> None:
        """Set a (possibly nested) key in place, creating parents as needed."""
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot descend into non-mapping key {part!r}")
        node[parts[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy as plain Python containers (JSON/YAML friendly)."""
        return copy.deepcopy(self._data)

    # -- Path helpers ----------------------------------------------------
    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the project root."""
        raw = self.require(dotted)
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def ensure_dirs(self, *dotted: str) -> list[Path]:
        """Create the directories named by the given config keys."""
        made = []
        for key in dotted:
            p = self.path(key)
            p.mkdir(parents=True, exist_ok=True)
            made.append(p)
        return made


def _coerce(text: str) -> Any:
    """Turn a CLI string into a native value ('0.02' -> 0.02, 'true' -> True)."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", ""}:
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(cfg: Config, overrides: Iterable[str] | None) -> Config:
    """Apply ``key.path=value`` strings onto *cfg* and return it."""
    for item in overrides or ():
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        key, _, raw = item.partition("=")
        cfg.set(key.strip(), _coerce(raw))
    return cfg


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Iterable[str] | None = None,
) -> Config:
    """Load the YAML configuration, apply CLI overrides and return a :class:`Config`.

    Parameters
    ----------
    path:
        Explicit config file. Falls back to ``$MBA_CONFIG`` and then to
        ``config/config.yaml`` at the repository root.
    overrides:
        Iterable of ``dotted.key=value`` strings, applied in order.
    """
    candidate = Path(path or os.environ.get("MBA_CONFIG") or DEFAULT_CONFIG_PATH)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.exists():
        raise ConfigError(f"configuration file not found: {candidate}")

    with candidate.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    return apply_overrides(Config(data, candidate), overrides)
