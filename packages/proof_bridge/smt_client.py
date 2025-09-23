"""Thin wrapper around an SMT solver invocation."""

from __future__ import annotations

import time
from typing import Any, Callable

from . import obligations

Runner = Callable[
    [obligations.ProofObligation, float | None], obligations.ProofResult | dict[str, Any]
]

__all__ = ["SmtClient", "prove"]


def _default_runner(
    obligation: obligations.ProofObligation, timeout: float | None
) -> obligations.ProofResult:
    """Fallback used in tests when no solver is configured."""

    return obligations.ProofResult(
        obligation_id=obligation.identifier,
        backend=obligation.backend,
        status=obligations.ProofStatus.UNKNOWN,
        diagnostics="SMT solver not configured",
    )


class SmtClient:
    """Client abstraction around an SMT backend.

    The implementation is intentionally lightweight: we accept a ``runner``
    callable (easy to monkeypatch in tests) that receives a
    :class:`ProofObligation` and returns a :class:`ProofResult`.  The default
    runner simply marks the obligation as ``unknown`` so callers can decide
    whether to escalate to a stronger prover.
    """

    def __init__(self, *, runner: Runner | None = None, timeout: float | None = 10.0) -> None:
        self._runner = runner or _default_runner
        self._timeout = timeout

    def prove(self, obligation: obligations.ProofObligation) -> obligations.ProofResult:
        start = time.perf_counter()
        raw = self._runner(obligation, self._timeout)
        result = _normalise_result(obligation, raw)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return result.with_duration(duration_ms)


def prove(
    obligation: obligations.ProofObligation,
    *,
    runner: Runner | None = None,
    timeout: float | None = 10.0,
) -> obligations.ProofResult:
    """Convenience wrapper that mirrors the historical functional API."""

    client = SmtClient(runner=runner, timeout=timeout)
    return client.prove(obligation)


def _normalise_result(
    obligation: obligations.ProofObligation,
    response: obligations.ProofResult | dict[str, Any],
) -> obligations.ProofResult:
    if isinstance(response, obligations.ProofResult):
        return response
    if not isinstance(response, dict):
        raise TypeError("SMT runner must return ProofResult or mapping")
    data = dict(response)
    data.setdefault("obligation_id", obligation.identifier)
    data.setdefault("backend", obligation.backend)
    return obligations.ProofResult.from_mapping(data)
