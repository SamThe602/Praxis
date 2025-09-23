"""Minimal yet test-friendly metrics registry for Praxis telemetry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from . import exporters


@dataclass(frozen=True)
class MetricSample:
    """Immutable record representing a single metric observation."""

    name: str
    value: float
    timestamp: float
    kind: str
    tags: Mapping[str, str]
    extra: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "kind": self.kind,
        }
        if self.tags:
            payload["tags"] = dict(self.tags)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass
class MetricSeries:
    """Collection of metric samples exposed with aggregate helpers."""

    name: str
    kind: str
    description: str | None = None
    unit: str | None = None
    samples: list[MetricSample] = field(default_factory=list)

    def add_sample(self, sample: MetricSample) -> None:
        self.samples.append(sample)

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                "name": self.name,
                "kind": self.kind,
                "count": 0,
            }
        values = [item.value for item in self.samples]
        total = sum(values)
        summary: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "count": len(values),
            "avg": total / len(values),
            "min": min(values),
            "max": max(values),
            "last": self.samples[-1].to_dict(),
        }
        if self.description:
            summary["description"] = self.description
        if self.unit:
            summary["unit"] = self.unit
        return summary


class MetricsRegistry:
    """Thread-safe registry storing metric series in-memory."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._series: Dict[str, MetricSeries] = {}

    def emit(
        self,
        name: str,
        value: Any,
        *,
        kind: str | None = None,
        tags: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> MetricSample:
        if not isinstance(name, str) or not name:
            raise ValueError("metric name must be a non-empty string")
        numeric_value = _coerce_value(value)
        inferred = _METRIC_CATALOG.get(name, {})
        metric_kind = kind or inferred.get("kind", "gauge")
        description = inferred.get("description")
        unit = inferred.get("unit")
        timestamp = time.time()
        metadata = dict(extra or {})
        if description and "description" not in metadata:
            metadata["description"] = description
        if unit and "unit" not in metadata:
            metadata["unit"] = unit
        sample = MetricSample(
            name=name,
            value=numeric_value,
            timestamp=timestamp,
            kind=metric_kind,
            tags=dict(tags or {}),
            extra=metadata,
        )
        with self._lock:
            series = self._series.get(name)
            if series is None:
                series = MetricSeries(
                    name=name,
                    kind=metric_kind,
                    description=description,
                    unit=unit,
                )
                self._series[name] = series
            series.add_sample(sample)
        exporters.export(sample)
        return sample

    def get_series(self, name: str) -> MetricSeries | None:
        with self._lock:
            series = self._series.get(name)
            if series is None:
                return None
            clone = MetricSeries(
                name=series.name,
                kind=series.kind,
                description=series.description,
                unit=series.unit,
                samples=list(series.samples),
            )
            return clone

    def snapshot(self) -> Dict[str, list[Dict[str, Any]]]:
        with self._lock:
            return {
                name: [sample.to_dict() for sample in series.samples]
                for name, series in self._series.items()
            }

    def summaries(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {name: series.summary() for name, series in self._series.items()}

    def reset(self) -> None:
        with self._lock:
            self._series.clear()


_METRIC_CATALOG: Dict[str, Dict[str, Any]] = {
    "praxis.translator.accuracy": {
        "kind": "gauge",
        "description": "Confidence score returned by the translator",
        "unit": "ratio",
    },
    "praxis.synth.search_nodes": {
        "kind": "counter",
        "description": "Number of nodes explored during synthesis search",
        "unit": "count",
    },
    "praxis.vm.latency_ms": {
        "kind": "gauge",
        "description": "Wall clock latency for VM executions",
        "unit": "milliseconds",
    },
    "praxis.verifier.smt_calls": {
        "kind": "counter",
        "description": "Number of SMT solver invocations performed by the verifier",
        "unit": "count",
    },
    "praxis.feedback.pending": {
        "kind": "gauge",
        "description": "Pending items in the human feedback queue",
        "unit": "count",
    },
    "praxis.rl.reward": {
        "kind": "gauge",
        "description": "Rolling reward signal emitted during RL updates",
        "unit": "score",
    },
    "praxis.pref.loss": {
        "kind": "gauge",
        "description": "Preference model loss computed on the latest batch",
        "unit": "loss",
    },
    "praxis.sandbox.execution_ms": {
        "kind": "gauge",
        "description": "Wall clock latency for sandboxed VM executions",
        "unit": "milliseconds",
    },
    "praxis.sandbox.memory_peak_kb": {
        "kind": "gauge",
        "description": "Peak RSS observed during sandboxed execution",
        "unit": "kilobytes",
    },
    "praxis.sandbox.violation": {
        "kind": "counter",
        "description": "Number of sandbox policy violations prevented",
        "unit": "count",
    },
}


_REGISTRY = MetricsRegistry()


def emit(
    metric: str,
    value: Any,
    *,
    kind: str | None = None,
    tags: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> MetricSample:
    """Record ``value`` for ``metric`` and forward to configured exporters."""

    return _REGISTRY.emit(metric, value, kind=kind, tags=tags, extra=extra)


def get_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry."""

    return _REGISTRY


def _coerce_value(value: Any) -> float:
    candidate = value
    if isinstance(value, Mapping):
        for key in ("value", "count", "total"):
            if key in value:
                candidate = value[key]
                break
    if isinstance(candidate, bool):
        return 1.0 if candidate else 0.0
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        return float(candidate)
    if isinstance(candidate, str):
        try:
            return float(candidate)
        except ValueError as exc:
            raise TypeError(f"metric value for {candidate!r} must be numeric") from exc
    raise TypeError(f"metric value for {candidate!r} must be numeric")


__all__ = [
    "MetricSample",
    "MetricSeries",
    "MetricsRegistry",
    "emit",
    "get_registry",
]
