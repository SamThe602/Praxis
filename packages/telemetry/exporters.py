"""Metric exporters for Praxis telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - imported only for typing
    from .metrics import MetricSample


class Exporter(Protocol):
    """Protocol implemented by all metric exporters."""

    def export(self, sample: "MetricSample") -> None:  # pragma: no cover - interface definition
        ...


class JsonlExporter:
    """Append metric samples to a JSONL file on disk."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def export(self, sample: "MetricSample") -> None:
        record = sample.to_dict()
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")


class PrometheusExporter:
    """In-memory stub that exposes metrics in Prometheus exposition format."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._latest: dict[str, "MetricSample"] = {}

    def export(self, sample: "MetricSample") -> None:
        with self._lock:
            self._latest[sample.name] = sample

    def render(self) -> str:
        """Return the current snapshot as a text exposition."""

        with self._lock:
            lines: list[str] = []
            for name in sorted(self._latest.keys()):
                sample = self._latest[name]
                if sample.extra.get("description"):
                    lines.append(f"# HELP {name} {sample.extra['description']}")
                if sample.extra.get("unit"):
                    lines.append(f"# UNIT {name} {sample.extra['unit']}")
                label_str = _render_labels(sample)
                lines.append(f"{name}{label_str} {sample.value} {int(sample.timestamp * 1000)}")
            return "\n".join(lines) + ("\n" if lines else "")


_DEFAULT_EXPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "telemetry.jsonl"
_EXPORTER: Exporter = JsonlExporter(_DEFAULT_EXPORT_PATH)


def configure(exporter: Exporter) -> None:
    """Override the global exporter used by :func:`export`."""

    global _EXPORTER
    _EXPORTER = exporter


def export(sample: "MetricSample") -> None:
    """Forward ``sample`` to the active exporter."""

    if _EXPORTER is None:  # pragma: no cover - defensive guard
        return
    _EXPORTER.export(sample)


def _render_labels(sample: "MetricSample") -> str:
    if not sample.tags:
        return ""
    pairs = ",".join(f"{key}={_quote_label(value)}" for key, value in sorted(sample.tags.items()))
    return f"{{{pairs}}}"


def _quote_label(value: object) -> str:
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return f'"{escaped}"'


__all__ = ["Exporter", "JsonlExporter", "PrometheusExporter", "configure", "export"]
