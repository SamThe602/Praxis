"""Synthesiser pruning hooks backed by static analysis results."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

from packages.verifier import static_analysis

__all__ = ["should_prune"]


def should_prune(state: Any) -> bool:
    """Return ``True`` when the static analyzer rejects the candidate state."""

    cache = _analysis_cache(state)
    if cache is not None and "static" in cache:
        result = cache["static"]
        return not result.ok

    program = _extract_program(state)
    if program is None:
        return False

    result = static_analysis.analyze(program)
    if cache is not None:
        cache["static"] = result
    return not result.ok


def _analysis_cache(state: Any) -> Optional[MutableMapping[str, Any]]:
    if isinstance(state, MutableMapping):
        return state.setdefault("analysis", {})
    if hasattr(state, "analysis"):
        existing = getattr(state, "analysis")
        if existing is None:
            existing = {}
            setattr(state, "analysis", existing)
        if isinstance(existing, MutableMapping):
            return existing
    return None


def _extract_program(state: Any) -> Any:
    if isinstance(state, Mapping):
        program = state.get("program") or state.get("bytecode")
        if program is not None:
            return program
    if hasattr(state, "program"):
        return getattr(state, "program")
    if hasattr(state, "bytecode"):
        return getattr(state, "bytecode")
    return None
