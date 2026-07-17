"""YAML configuration loader supporting inheritance and overrides.

Allows loading hierarchical configuration files, merging nested dictionaries,
applying command-line dot-notation overrides, and saving resolved configurations
for experiment reproducibility.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _coerce(value: str) -> Any:
    """Parse a CLI override string into a Python scalar via YAML rules."""
    return yaml.safe_load(value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, merge any `include:` files, then apply CLI overrides.

    `include` entries are resolved relative to the main config's directory and
    are merged *under* the main file (the main file wins on conflicts).
    Overrides are `dotted.key=value` strings applied last.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    includes = cfg.pop("include", []) or []
    merged: dict = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        with inc_path.open("r", encoding="utf-8") as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})
    merged = _deep_merge(merged, cfg)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        node = merged
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _coerce(raw)

    return merged


def dump_config(cfg: dict, path: str | Path) -> None:
    """Write the fully resolved config to disk for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
