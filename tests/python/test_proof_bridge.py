"""Unit tests for the proof bridge orchestrator and cache handling."""

from __future__ import annotations

from pathlib import Path

from packages.proof_bridge import coq_client, obligations, smt_client
from packages.verifier import proof_bridge


def test_discharge_obligations_dedupes_and_uses_cache(tmp_path: Path) -> None:
    program = {
        "proof_obligations": [
            {"id": "commutativity", "backend": "smt", "payload": {"goal": "a + b = b + a"}},
            {"id": "commutativity_dup", "backend": "smt", "payload": {"goal": "a + b = b + a"}},
            {"id": "totality", "backend": "coq", "payload": "forall x, f x >= 0"},
        ]
    }

    smt_calls: list[str] = []
    coq_calls: list[str] = []

    def smt_runner(
        obligation: obligations.ProofObligation, _: float | None
    ) -> obligations.ProofResult:
        smt_calls.append(obligation.identifier)
        return obligations.ProofResult(
            obligation_id=obligation.identifier,
            backend=obligation.backend,
            status=obligations.ProofStatus.PROVED,
            diagnostics="proved via stub",
        )

    def coq_runner(
        obligation: obligations.ProofObligation, _: float | None
    ) -> obligations.ProofResult:
        coq_calls.append(obligation.identifier)
        return obligations.ProofResult(
            obligation_id=obligation.identifier,
            backend=obligation.backend,
            status=obligations.ProofStatus.REFUTED,
            counterexample={"x": 0},
            diagnostics="counterexample found",
        )

    first = proof_bridge.discharge_obligations(
        program,
        cache_dir=tmp_path,
        smt=smt_client.SmtClient(runner=smt_runner),
        coq=coq_client.CoqClient(runner=coq_runner),
    )

    assert len(first.obligations) == 2  # duplicate goal collapses
    assert len(first.results) == 2
    assert first.summary["proved"] == 1
    assert first.summary["refuted"] == 1
    assert first.summary["cached"] == 0
    assert first.summary["status"] == "failed"
    assert smt_calls == ["commutativity"]
    assert coq_calls == ["totality"]

    smt_calls.clear()
    coq_calls.clear()

    second = proof_bridge.discharge_obligations(
        program,
        cache_dir=tmp_path,
        smt=smt_client.SmtClient(runner=smt_runner),
        coq=coq_client.CoqClient(runner=coq_runner),
    )

    assert smt_calls == []
    assert coq_calls == []
    assert second.summary["cached"] == 2
    assert all(result.cached for result in second.results)
    counterexamples = [result.counterexample for result in second.results if result.counterexample]
    assert counterexamples == [{"x": 0}]
