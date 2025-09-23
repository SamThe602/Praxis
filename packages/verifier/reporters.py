"""Telemetry helpers for static analysis reporting."""

from __future__ import annotations

from typing import Any, Dict

from .static_analysis import AbstractValue, StaticAnalysisResult

__all__ = ["build_report"]


def build_report(result: StaticAnalysisResult) -> Dict[str, Any]:
    """Convert ``result`` into a telemetry-friendly dictionary."""

    payload: Dict[str, Any] = {
        "status": "ok" if result.ok else "failed",
        "failure_count": len(result.failures),
    }
    if result.failures:
        payload["failures"] = [failure.to_dict() for failure in result.failures]
    if result.summaries:
        payload["functions"] = {
            name: {
                "registers": {
                    f"r{reg}": _describe_value(value)
                    for reg, value in summary.final_registers.items()
                }
            }
            for name, summary in result.summaries.items()
        }
    return payload


def _describe_value(value: AbstractValue) -> Dict[str, Any]:
    if value.kind == "int" and value.interval is not None:
        return {
            "kind": "int",
            "interval": _format_interval(value.interval.lower, value.interval.upper),
        }
    if value.kind == "bool":
        return {"kind": "bool", "value": value.bool_value}
    if value.kind == "string":
        return {"kind": "string", "value": value.string_value}
    if value.kind == "list" and value.list_value is not None:
        interval = value.list_value.length
        return {
            "kind": "list",
            "length": _format_interval(interval.lower, interval.upper),
        }
    if value.kind == "map" and value.map_value is not None:
        summary = {
            "kind": "map",
            "known_keys": sorted(value.map_value.known_keys),
        }
        if value.map_value.may_have_unknown:
            summary["may_have_unknown"] = True
        return summary
    if value.kind == "unknown":
        return {"kind": "unknown"}
    return {"kind": value.kind}


def _format_interval(lower: Any, upper: Any) -> str:
    lo = "-inf" if lower is None else str(int(lower) if isinstance(lower, bool) else lower)
    hi = "+inf" if upper is None else str(int(upper) if isinstance(upper, bool) else upper)
    if lo == hi:
        return lo
    return f"[{lo}, {hi}]"
