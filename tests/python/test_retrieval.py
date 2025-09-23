"""End-to-end tests for the lightweight retrieval pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from packages.guidance import embeddings
from packages.orchestrator.spec_loader import ExampleSpec, IOSpec, StructuredSpec
from packages.synthesizer import expansions, retrieval, scoring
from packages.synthesizer.state import SynthState


def _make_spec(identifier: str, prompt: str, constraint: str) -> StructuredSpec:
    inputs = (IOSpec(name="data", type="list[int]"),)
    outputs = (IOSpec(name="result", type="list[int]"),)
    constraints = (constraint,)
    goal_tokens = (
        "use_input:data",
        "produce_output:result",
        f"constraint:{constraint}",
    )
    input_types = {"data": "list[int]"}
    examples = (ExampleSpec(inputs={"data": [1, 2]}, outputs={"result": [1, 2]}),)
    raw = {
        "id": identifier,
        "natural_prompt": prompt,
        "constraints": [constraint],
    }
    return StructuredSpec(
        identifier=identifier,
        metadata={},
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
        natural_prompt=prompt,
        latency_target_ms=None,
        operators=("InlineSnippet", "Sort"),
        reuse_pool=(),
        examples=examples,
        goal_tokens=goal_tokens,
        input_types=input_types,
        raw=raw,
    )


def test_embed_spec_is_deterministic() -> None:
    spec = _make_spec("demo-sort", "Sort numbers in ascending order", "ensure ascending")
    first = embeddings.embed_spec(spec)
    second = embeddings.embed_spec(spec)
    assert np.allclose(first.vector, second.vector)
    assert first.tokens == second.tokens
    norm = float(np.linalg.norm(first.vector))
    assert norm == pytest.approx(1.0) or norm == 0.0


def test_retrieval_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PRAXIS_KB_PATH", str(tmp_path))
    kb = retrieval.KnowledgeBase.load()

    sort_spec = _make_spec("spec-sort", "Sort an integer list ascending", "ascending order")
    fact_spec = _make_spec("spec-factorial", "Compute factorial of n", "factorial recurrence")

    kb.record(spec=sort_spec, snippet="return sorted(data)")
    kb.record(spec=fact_spec, snippet="return factorial(data[0])")
    kb.save()

    kb_reloaded = retrieval.KnowledgeBase.load()
    query_spec = _make_spec("query-sort", "Sort integer array ascending", "ascending order")

    results = retrieval.retrieve(
        query_spec, kb=kb_reloaded, config=retrieval.RetrievalConfig(top_k=2)
    )
    assert [item.spec_id for item in results] == ["spec-sort", "spec-factorial"]
    assert results[0].score >= results[1].score

    enriched_spec = retrieval.apply_retrieval(
        query_spec, kb=kb_reloaded, config=retrieval.RetrievalConfig(top_k=2)
    )
    assert enriched_spec.reuse_pool[0] == "return sorted(data)"
    retrieval_meta = enriched_spec.metadata.get("retrieval", {})
    assert len(retrieval_meta.get("results", ())) == 2
    assert retrieval_meta["results"][0]["spec_id"] == "spec-sort"

    root_state = SynthState(
        steps=(),
        covered_requirements=frozenset(),
        pending_requirements=("constraint:ascending order",),
        metadata={},
    )
    children = tuple(expansions.expand(root_state, spec=enriched_spec))
    inline_children = [
        child
        for child in children
        if any(step.startswith("InlineSnippet::") for step in child.steps)
    ]
    assert inline_children, "expected InlineSnippet expansion"

    best_inline = inline_children[0]
    score = scoring.composite_score(best_inline, spec=enriched_spec)
    assert score.reuse_bonus > 0.0
    assert best_inline.metadata.get("reuse_rank") == 0
    assert best_inline.metadata.get("snippet_reuse", 0.0) >= results[0].score - 1e-6
