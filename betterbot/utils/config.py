"""Utility helpers for loading BetterBot configuration files.

The configuration file is expected to be a YAML document that can be overridden by
setting the environment variable ``BETTERBOT_CONFIG`` to point at an alternative
path. Values inside the YAML file can also be overridden by environment variables
following the ``BETTERBOT__SECTION__KEY`` naming convention.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency resolution guard
    raise ImportError(
        "PyYAML is required to load BetterBot configuration. Install it via 'pip install pyyaml'."
    ) from exc


CONFIG_ENV_VAR = "BETTERBOT_CONFIG"
ENV_OVERRIDE_PREFIX = "BETTERBOT__"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "betterbot.yaml"


@dataclass
class Config:
    """Container object that allows attribute-style access to nested dictionaries."""

    data: Mapping[str, Any]

    def get(self, path: str, default: Any = None) -> Any:
        """Retrieve a value using dotted-path notation (e.g. ``api.key``)."""

        keys: Iterable[str] = path.split(".") if path else []
        current: Any = self.data
        for key in keys:
            if isinstance(current, Mapping) and key in current:
                current = current[key]
            else:
                return default
        return current

    def __getitem__(self, item: str) -> Any:
        return self.data[item]

    def __contains__(self, item: object) -> bool:
        return item in self.data


def _apply_env_overrides(config: MutableMapping[str, Any]) -> None:
    """Apply overrides from environment variables using the prefix convention."""

    for env_key, value in os.environ.items():
        if not env_key.startswith(ENV_OVERRIDE_PREFIX):
            continue
        path = env_key[len(ENV_OVERRIDE_PREFIX) :].lower().split("__")
        cursor: MutableMapping[str, Any] = config
        for key in path[:-1]:
            if key not in cursor or not isinstance(cursor[key], MutableMapping):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = _parse_env_value(value)


def _parse_env_value(raw: str) -> Any:
    """Parse primitive values from strings for convenience."""

    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_config(path: str | Path | None = None) -> Config:
    """Load the BetterBot configuration file and return a :class:`Config` object."""

    config_path = Path(path or os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(
            f"BetterBot configuration file not found at '{config_path}'. "
            "Create the file or point BETTERBOT_CONFIG at an alternative path."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config: Dict[str, Any] = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, MutableMapping):
        raise TypeError("Configuration root must be a mapping/dictionary.")

    _apply_env_overrides(raw_config)
    return Config(raw_config)


__all__ = ["Config", "load_config", "CONFIG_ENV_VAR", "DEFAULT_CONFIG_PATH"]
