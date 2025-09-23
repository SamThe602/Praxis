"""Utility helpers for loading YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config"]


def load_config(path: str | Path) -> Any:
    """Return the parsed YAML document located at ``path``.

    The helper keeps behaviour intentionally small: it performs basic existence
    checks, parses the document via :func:`yaml.safe_load`, and raises a
    :class:`ValueError` if the result is not a mapping-like object.  The unit
    tests only rely on dictionaries, keeping the return type flexible avoids
    locking in a heavier configuration framework.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive parsing guard
        raise ValueError(f"failed to parse configuration: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data
