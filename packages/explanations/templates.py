"""Template helpers for generating human-readable explanations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ExplanationSections:
    """Structured building blocks used by the renderer."""

    summary: str
    reasoning_steps: Sequence[str] = field(default_factory=tuple)
    verification: Sequence[str] = field(default_factory=tuple)
    telemetry: Sequence[str] = field(default_factory=tuple)
    caveats: Sequence[str] = field(default_factory=tuple)


def build_sections(payload: Mapping[str, Any]) -> ExplanationSections:
    """Convert a synthesis report into templated explanation sections."""

    program = payload.get("program") or {}
    trace = payload.get("trace") or {}
    verifier = payload.get("verifier") or {}
    telemetry = payload.get("telemetry") or {}

    summary = _summarise(program, verifier)
    reasoning = tuple(_render_reasoning(trace))
    verification = tuple(_render_verification(verifier))
    telemetry_lines = tuple(_render_telemetry(telemetry))
    caveats = tuple(_render_caveats(program, verifier))

    return ExplanationSections(
        summary=summary,
        reasoning_steps=reasoning,
        verification=verification,
        telemetry=telemetry_lines,
        caveats=caveats,
    )


def _summarise(program: Mapping[str, Any], verifier: Mapping[str, Any]) -> str:
    entry = program.get("entry") or "solve"
    spec_id = program.get("spec_id") or program.get("id") or "unknown_spec"
    status = verifier.get("status") or verifier.get("outcome") or "unknown"
    result = verifier.get("result")
    if isinstance(result, Mapping) and "value" in result:
        result = result.get("value")
    if result is None:
        result_text = "no return value was recorded"
    else:
        result_text = f"returned {result!r}"
    return f"Program `{entry}` for `{spec_id}` {result_text}; verifier status: {status}."


def _render_reasoning(trace: Mapping[str, Any]) -> Iterable[str]:
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    if isinstance(steps, Iterable) and not isinstance(steps, (str, bytes)):
        for index, item in enumerate(steps, start=1):
            if isinstance(item, Mapping):
                op = item.get("op") or item.get("instruction") or "step"
                detail = item.get("detail") or item.get("comment")
                if detail:
                    yield f"Step {index}: {op} – {detail}."
                else:
                    yield f"Step {index}: {op}."
            else:
                yield f"Step {index}: {item}."
    elif trace:
        highlight = trace.get("summary") if isinstance(trace, Mapping) else None
        if highlight:
            yield str(highlight)


def _render_verification(verifier: Mapping[str, Any]) -> Iterable[str]:
    checks = verifier.get("checks") if isinstance(verifier, Mapping) else None
    if isinstance(checks, Iterable) and not isinstance(checks, (str, bytes)):
        for check in checks:
            if isinstance(check, Mapping):
                name = check.get("name") or "check"
                status = check.get("status") or check.get("result") or "unknown"
                detail = check.get("detail")
                if detail:
                    yield f"{name}: {status} ({detail})."
                else:
                    yield f"{name}: {status}."
            else:
                yield str(check)
    failures = verifier.get("failures") if isinstance(verifier, Mapping) else None
    if failures:
        for failure in failures:
            yield f"Failure: {failure}."


def _render_telemetry(telemetry: Mapping[str, Any]) -> Iterable[str]:
    if not isinstance(telemetry, Mapping):
        return ()
    metrics = telemetry.get("metrics")
    if isinstance(metrics, Mapping):
        for name, value in sorted(metrics.items()):
            yield f"{name}: {value}."
    extras = telemetry.get("extras")
    if isinstance(extras, Mapping):
        for key, value in sorted(extras.items()):
            yield f"{key}: {value}."


def _render_caveats(program: Mapping[str, Any], verifier: Mapping[str, Any]) -> Iterable[str]:
    caveats = []
    if isinstance(verifier, Mapping) and not verifier.get("status", "ok") == "ok":
        caveats.append("Verification reported issues; review recommended.")
    if program.get("heuristics") == "experimental":
        caveats.append("Program relies on experimental heuristics.")
    return caveats


__all__ = ["ExplanationSections", "build_sections"]
