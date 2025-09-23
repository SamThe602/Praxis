"""Text-only dashboards for inspecting telemetry while developing locally."""

from __future__ import annotations

from datetime import datetime

from . import analyzers
from . import metrics as metric_module


def build_dashboard(registry: metric_module.MetricsRegistry | None = None) -> str:
    """Return a human-readable dashboard snapshot of the current telemetry."""

    report = analyzers.analyze(registry)
    derived = report.get("derived", {})
    summaries = report.get("summaries", {})
    lines = [
        "Praxis Telemetry Dashboard",
        "Generated at: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "",
    ]
    if derived:
        lines.append("Key Indicators:")
        for name, section in derived.items():
            pretty = _humanise_name(name)
            lines.append(f"  - {pretty}:")
            for key, value in section.items():
                lines.append(f"      {key}: {value}")
        lines.append("")
    if summaries:
        lines.append("Tracked Metrics:")
        for name in sorted(summaries.keys()):
            summary = summaries[name]
            lines.append(f"  * {name} ({summary.get('kind', 'unknown')}):")
            lines.append(
                f"      samples={summary.get('count', 0)} avg={summary.get('avg')} min={summary.get('min')} max={summary.get('max')}"
            )
            last = summary.get("last") or {}
            if last:
                lines.append(f"      latest={last.get('value')} at {last.get('timestamp')}")
        lines.append("")
    if len(lines) == 3:
        lines.append(
            "No telemetry recorded yet. Trigger a translator or synthesis run to populate the dashboard."
        )
    return "\n".join(lines)


def _humanise_name(name: str) -> str:
    return name.replace("_", " ").title()


__all__ = ["build_dashboard"]
