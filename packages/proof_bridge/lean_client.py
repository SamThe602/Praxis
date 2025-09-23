"""Lean prover bridge client."""

from __future__ import annotations

import time
from typing import Any, Callable

from . import obligations

Runner = Callable[
    [obligations.ProofObligation, float | None], obligations.ProofResult | dict[str, Any]
]

__all__ = ["LeanClient", "prove"]


def _default_runner(
    obligation: obligations.ProofObligation, timeout: float | None
) -> obligations.ProofResult:
    return obligations.ProofResult(
        obligation_id=obligation.identifier,
        backend=obligation.backend,
        status=obligations.ProofStatus.UNKNOWN,
        diagnostics="Lean prover not configured",
    )


class LeanClient:
    """Lightweight wrapper with injectable execution hook for tests."""

    def __init__(self, *, runner: Runner | None = None, timeout: float | None = 20.0) -> None:
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
    timeout: float | None = 20.0,
) -> obligations.ProofResult:
    client = LeanClient(runner=runner, timeout=timeout)
    return client.prove(obligation)


def _normalise_result(
    obligation: obligations.ProofObligation,
    response: obligations.ProofResult | dict[str, Any],
) -> obligations.ProofResult:
    if isinstance(response, obligations.ProofResult):
        return response
    if not isinstance(response, dict):
        raise TypeError("Lean runner must return ProofResult or mapping")
    data = dict(response)
    data.setdefault("obligation_id", obligation.identifier)
    data.setdefault("backend", obligation.backend)
    return obligations.ProofResult.from_mapping(data)
