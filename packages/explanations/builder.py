"""High-level API for turning synthesis reports into explanations."""

from __future__ import annotations

from typing import Any, Mapping

from . import renderers, templates


def build_explanation(report: Mapping[str, Any] | None) -> str:
    """Return a Markdown explanation for the provided *report*."""

    payload: Mapping[str, Any]
    if report is None:
        payload = {}
    elif isinstance(report, Mapping):
        payload = report
    else:
        raise TypeError("report must be a mapping or None")

    sections = templates.build_sections(payload)
    return renderers.render_markdown(sections)


__all__ = ["build_explanation"]
