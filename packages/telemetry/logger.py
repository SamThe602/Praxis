"""Logging helpers used across the telemetry stack."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import yaml

_CONFIG_LOCK = RLock()
_CONFIGURED = False

_DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "standard",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
    "loggers": {
        "praxis": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        }
    },
}


def _load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "logging.yaml"
    if not config_path.exists():
        return dict(_DEFAULT_CONFIG)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - guard rails for broken configs
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("praxis.telemetry").warning("failed to parse logging.yaml: %s", exc)
        return dict(_DEFAULT_CONFIG)
    if not isinstance(data, Mapping):
        return dict(_DEFAULT_CONFIG)
    merged = dict(_DEFAULT_CONFIG)
    merged.update(
        {
            k: v
            for k, v in data.items()
            if k
            in ("version", "disable_existing_loggers", "formatters", "handlers", "root", "loggers")
        }
    )
    return merged


def configure() -> None:
    """Ensure the logging subsystem is configured exactly once."""

    global _CONFIGURED
    with _CONFIG_LOCK:
        if _CONFIGURED:
            return
        config = _load_config()
        logging.config.dictConfig(config)
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured via ``configs/logging.yaml``."""

    if not isinstance(name, str) or not name:
        raise ValueError("logger name must be a non-empty string")
    configure()
    return logging.getLogger(name)


__all__ = ["configure", "get_logger"]
