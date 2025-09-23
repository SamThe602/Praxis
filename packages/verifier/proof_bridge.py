"""Proof bridge orchestrator tying bytecode obligations to provers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from packages.proof_bridge import cache as proof_cache
from packages.proof_bridge import coq_client, lean_client, smt_client
from packages.proof_bridge.obligations import (
    ObligationQueue,
    ProofObligation,
    ProofResult,
    ProofStatus,
    collect_from_program,
)

ClientLike = Any

__all__ = ["ProofBridgeResult", "discharge_obligations"]


@dataclass(slots=True)
class ProofBridgeResult:
    """Aggregate outcome returned by :func:`discharge_obligations`."""

    obligations: tuple[ProofObligation, ...]
    results: tuple[ProofResult, ...]
    summary: dict[str, int | str]

    @property
    def ok(self) -> bool:
        return self.summary.get("status") == "ok"


def discharge_obligations(
    program: Mapping[str, Any] | Any,
    *,
    cache_dir: str | Path | None = None,
    smt: ClientLike | None = None,
    coq: ClientLike | None = None,
    lean: ClientLike | None = None,
) -> ProofBridgeResult:
    """Send proof obligations attached to ``program`` to the appropriate provers."""

    obligations_from_program = collect_from_program(program)
    if not obligations_from_program:
        summary: dict[str, int | str] = {
            "proved": 0,
            "refuted": 0,
            "unknown": 0,
            "error": 0,
            "cached": 0,
            "status": "ok",
        }
        return ProofBridgeResult(obligations=(), results=(), summary=summary)

    queue = ObligationQueue()
    for item in obligations_from_program:
        queue.add(item)

    resolved: list[ProofResult] = []
    cached_hits = 0

    for obligation in queue:
        cached_result = proof_cache.load_result(obligation, cache_dir=cache_dir)
        if cached_result is not None:
            resolved.append(cached_result)
            cached_hits += 1
            continue

        client = _select_client(obligation.backend, smt=smt, coq=coq, lean=lean)
        result = _invoke_client(client, obligation)
        proof_cache.store_result(obligation, result, cache_dir=cache_dir)
        resolved.append(result)

    summary = _summarise(resolved, cached_hits)
    return ProofBridgeResult(obligations=tuple(queue), results=tuple(resolved), summary=summary)


def _select_client(
    backend: str,
    *,
    smt: ClientLike | None,
    coq: ClientLike | None,
    lean: ClientLike | None,
) -> ClientLike:
    backend = backend.lower()
    if backend == "smt":
        return smt if smt is not None else smt_client.SmtClient()
    if backend == "coq":
        return coq if coq is not None else coq_client.CoqClient()
    if backend == "lean":
        return lean if lean is not None else lean_client.LeanClient()
    raise ValueError(f"Unknown proof backend '{backend}'")


def _invoke_client(client: ClientLike, obligation: ProofObligation) -> ProofResult:
    if hasattr(client, "prove"):
        response = client.prove(obligation)
    elif callable(client):
        response = client(obligation)
    else:
        raise TypeError("Client must expose a 'prove' method or be callable")

    if isinstance(response, ProofResult):
        return response
    if not isinstance(response, Mapping):
        raise TypeError("Prover response must be ProofResult or mapping")
    payload = dict(response)
    payload.setdefault("obligation_id", obligation.identifier)
    payload.setdefault("backend", obligation.backend)
    return ProofResult.from_mapping(payload)


def _summarise(results: Iterable[ProofResult], cached_hits: int) -> dict[str, int | str]:
    counters: dict[str, int | str] = {status.value: 0 for status in ProofStatus}
    for result in results:
        key = result.status.value
        current = cast(int, counters.get(key, 0))
        counters[key] = current + 1
    counters["cached"] = cached_hits
    counters["status"] = (
        "ok"
        if cast(int, counters.get("refuted", 0)) == 0 and cast(int, counters.get("error", 0)) == 0
        else "failed"
    )
    return counters
