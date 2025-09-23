"""Property and metamorphic verifier test suite."""

from __future__ import annotations

from typing import Any, Mapping, cast

from packages.synthesizer import feedback as synth_feedback
from packages.synthesizer.state import SynthState
from packages.verifier import metamorphic, property_tests


def test_property_tests_collect_failures_and_feedback() -> None:
    program = {
        "property_tests": [
            {
                "name": "square_non_negative",
                "function": lambda value: value * value,
                "check": lambda actual=None, **_: actual >= 0,
                "cases": [
                    {"name": "positive", "inputs": {"value": 3}},
                    {"name": "negative", "inputs": {"value": -2}},
                ],
            },
            {
                "name": "reject_zero",
                "function": lambda value: value,
                "cases": [
                    {
                        "name": "zero",
                        "inputs": {"value": 0},
                        "check": lambda actual=None, **_: (False, "value should not be zero"),
                    },
                    {
                        "name": "one",
                        "inputs": {"value": 1},
                        "check": lambda actual=None, **_: True,
                    },
                ],
            },
        ]
    }

    run = property_tests.run_property_tests(program)

    assert len(run.results) == 4
    assert run.summary["passed"] == 3
    assert run.summary["failed"] == 1
    assert run.summary["status"] == "failed"

    failing = [result for result in run.results if result.status == "failed"][0]
    assert failing.case == "zero"
    assert failing.message == "value should not be zero"
    assert failing.counterexample == {"inputs": {"value": 0}, "expected": None, "actual": 0}

    assert len(run.feedback) == 1
    feedback_entry = run.feedback[0]
    assert feedback_entry["kind"] == "property_failure"
    assert feedback_entry["case"] == "zero"


def test_metamorphic_relations_detect_regressions() -> None:
    program = {
        "metamorphic_tests": [
            {
                "name": "commutativity",
                "function": lambda a, b: a + b,
                "relation": lambda base=None, candidate=None, **_: base["output"]
                == candidate["output"],
                "cases": [
                    {
                        "base": {"a": 2, "b": 3},
                        "variants": [
                            {"name": "swap", "inputs": {"a": 3, "b": 2}},
                            {
                                "name": "perturb",
                                "inputs": {"a": 5, "b": 1},
                                "relation": lambda base=None, candidate=None, **_: (
                                    candidate["output"] == base["output"],
                                    "sum should stay invariant",
                                    {"base": base, "variant": candidate},
                                ),
                            },
                        ],
                    }
                ],
            }
        ]
    }

    run = metamorphic.run_metamorphic_tests(program)

    assert len(run.results) == 2
    assert run.summary["passed"] == 1
    assert run.summary["failed"] == 1
    assert run.summary["status"] == "failed"

    failing = [result for result in run.results if result.status == "failed"][0]
    assert failing.variant == "perturb"
    assert failing.message == "sum should stay invariant"
    counterexample = cast(Mapping[str, Any] | None, failing.counterexample)
    assert counterexample is not None
    base_inputs = cast(Mapping[str, Any], counterexample["base"])["inputs"]
    assert base_inputs == {"a": 2, "b": 3}
    assert len(run.feedback) == 1
    assert run.feedback[0]["kind"] == "metamorphic_failure"


def test_synth_feedback_marks_state_invalid() -> None:
    state = SynthState()
    feedback_entries: list[dict[str, Any]] = [
        {"kind": "property_failure", "counterexample": {"inputs": {"x": 1}}, "message": "bad"},
        {"kind": "annotation", "severity": "info"},
    ]

    updated = synth_feedback.incorporate(state, feedback_entries)

    assert updated is not state
    assert updated.metadata["invalid"] is True
    assert len(updated.metadata["feedback"]) == 2
    assert updated.analysis["counterexamples"] == (feedback_entries[0]["counterexample"],)

    untouched = synth_feedback.incorporate(state, None)
    assert untouched is state
