"""Render structured explanation sections into Markdown."""

from __future__ import annotations

from typing import Iterable

from .templates import ExplanationSections


def render_markdown(sections: ExplanationSections) -> str:
    """Render *sections* into a compact Markdown document."""

    lines: list[str] = ["### Summary", sections.summary.strip(), ""]

    _append_block(lines, "Trace Highlights", sections.reasoning_steps)
    _append_block(lines, "Verification", sections.verification)
    _append_block(lines, "Telemetry", sections.telemetry)
    _append_block(lines, "Caveats", sections.caveats)

    while lines and lines[-1] == "":  # trim trailing blanks
        lines.pop()
    lines.append("")
    return "\n".join(lines)


def _append_block(lines: list[str], title: str, entries: Iterable[str]) -> None:
    entries = list(entry for entry in entries if str(entry).strip())
    if not entries:
        return
    lines.append(f"### {title}")
    for entry in entries:
        lines.append(f"- {entry}")
    lines.append("")


__all__ = ["render_markdown"]
