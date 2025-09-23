"""Utility helpers that compute roll-ups over recorded telemetry metrics."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from . import metrics as metric_module


def analyze(registry: metric_module.MetricsRegistry | None = None) -> Dict[str, Any]:
    """Return aggregated telemetry insights from ``registry``.

    The return payload includes raw metric summaries as well as a small derived
    section with the most common KPIs used by local dashboards.
    """

    registry = registry or metric_module.get_registry()
    if not isinstance(registry, metric_module.MetricsRegistry):
        raise TypeError("registry must be a MetricsRegistry instance")

    summaries = registry.summaries()
    derived = {
        "translator_confidence": _confidence_section(summaries.get("praxis.translator.accuracy")),
        "search_nodes": _simple_section(summaries.get("praxis.synth.search_nodes")),
        "vm_latency": _latency_section(summaries.get("praxis.vm.latency_ms")),
    }
    return {
        "summaries": summaries,
        "derived": {key: value for key, value in derived.items() if value is not None},
    }


def _confidence_section(summary: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not summary:
        return None
    return {
        "average": round(summary.get("avg", 0.0), 4),
        "latest": summary.get("last", {}).get("value"),
        "samples": summary.get("count", 0),
    }


def _simple_section(summary: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not summary:
        return None
    return {
        "total": summary.get("avg", 0.0) * summary.get("count", 0),
        "latest": summary.get("last", {}).get("value"),
        "samples": summary.get("count", 0),
    }


def _latency_section(summary: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not summary:
        return None
    return {
        "pseudomean_ms": round(summary.get("avg", 0.0), 3),
        "latest_ms": summary.get("last", {}).get("value"),
        "samples": summary.get("count", 0),
        "peak_ms": summary.get("max"),
    }


__all__ = ["analyze"]
