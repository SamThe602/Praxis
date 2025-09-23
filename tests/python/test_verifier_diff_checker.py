"""Tests for the diff checker covering structural and behavioural modes."""

from __future__ import annotations

from typing import Any, Sequence

from packages.verifier.diff_checker import DiffConfig, compare
from packages.verifier.runner import VmExecution


def _stub_runner(
    *, module: dict[str, Any], entry: str, args: Sequence[Any], limits: Any | None
) -> VmExecution:
    value = module.get("value", 0)
    if args:
        value = value + sum(int(arg) for arg in args if isinstance(arg, (int, float)))
    return VmExecution(return_value=value, trace={"entry": entry})


def test_structural_equivalence_whitespace_and_constants() -> None:
    candidate = "def foo():\n    return 1 + 2\n"
    baseline = "def foo():\n    return 3\n"
    report = compare(candidate, baseline, config=DiffConfig(behavioural_checks=0))
    assert report.structural.equivalent


def test_structural_difference_reports_diff() -> None:
    candidate = "def foo():\n    return 4\n"
    baseline = "def foo():\n    return 5\n"
    report = compare(candidate, baseline, config=DiffConfig(behavioural_checks=0))
    assert not report.structural.equivalent
    assert report.structural.differences


def test_behavioural_equivalence_with_stub_runner() -> None:
    candidate = {"module": {"value": 5}}
    baseline = {"module": {"value": 5}}
    report = compare(
        candidate,
        baseline,
        config=DiffConfig(
            behavioural_checks=2,
            behavioural_runner=_stub_runner,
            behavioural_inputs=[(), (1, 2)],
        ),
    )
    assert report.behavioural is not None
    assert report.behavioural.equivalent


def test_behavioural_mismatch_detected() -> None:
    candidate = {"module": {"value": 2}}
    baseline = {"module": {"value": 1}}
    report = compare(
        candidate,
        baseline,
        config=DiffConfig(
            behavioural_checks=2,
            behavioural_runner=_stub_runner,
            behavioural_inputs=[(), (1,)],
        ),
    )
    assert report.behavioural is not None
    assert not report.behavioural.equivalent
    assert report.behavioural.mismatches
