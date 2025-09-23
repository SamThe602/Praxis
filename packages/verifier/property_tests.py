"""Property-based testing harness for Praxis verifier artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import native_bridge


@dataclass(slots=True, frozen=True)
class PropertyCaseResult:
    """Outcome for a single property test case."""

    test: str
    case: str
    status: str
    message: str | None = None
    counterexample: Mapping[str, Any] | None = None
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "test": self.test,
            "case": self.case,
            "status": self.status,
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.counterexample is not None:
            payload["counterexample"] = dict(self.counterexample)
        if self.details is not None:
            payload["details"] = dict(self.details)
        return payload


@dataclass(slots=True)
class PropertyTestRun:
    """Aggregate summary for property testing."""

    results: tuple[PropertyCaseResult, ...]
    summary: dict[str, int | str]
    feedback: tuple[dict[str, Any], ...]

    @property
    def ok(self) -> bool:
        return self.summary.get("status") == "ok"


__all__ = ["PropertyCaseResult", "PropertyTestRun", "run_property_tests"]


def run_property_tests(
    program: Mapping[str, Any] | Any,
    *,
    evaluator: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
) -> PropertyTestRun:
    """Execute property tests attached to ``program``.

    The expected schema mirrors the orchestrator contract:
    ``program['property_tests']`` is a list of dictionaries with at least a
    ``name`` and ``cases`` array.  Each case can specify:

    * ``inputs`` – mapping or sequence passed to the function under test
    * ``function`` – callable producing the value under test (optional)
    * ``expected`` – literal or callable yielding expected value (optional)
    * ``check``/``predicate`` – callable receiving the evaluation context and
      returning ``bool`` or ``(bool, message)``

    Callers may also pass a custom ``evaluator`` which receives the test and
    case dictionaries and returns an outcome in the same format as ``check``.
    """

    tests = _collect_definitions(program, "property_tests", fallback="properties")
    results = []
    for test_index, definition in enumerate(tests):
        name = str(definition.get("name") or f"property_{test_index}")
        cases = definition.get("cases") or ()
        for case_index, raw_case in enumerate(cases):
            case_name = str(raw_case.get("name") or f"{name}[{case_index}]")
            try:
                outcome = _execute_case(name, case_name, definition, raw_case, evaluator)
            except Exception as exc:  # defensive: never crash the verifier
                outcome = PropertyCaseResult(
                    test=name,
                    case=case_name,
                    status="error",
                    message=str(exc),
                    counterexample={"inputs": raw_case.get("inputs"), "exception": repr(exc)},
                    details={"case": raw_case},
                )
            results.append(outcome)

    summary = _summarise(results)
    feedback = tuple(
        {
            "kind": "property_failure",
            "test": result.test,
            "case": result.case,
            "message": result.message,
            "counterexample": result.counterexample,
        }
        for result in results
        if result.status != "passed"
    )
    return PropertyTestRun(results=tuple(results), summary=summary, feedback=feedback)


def _collect_definitions(
    program: Mapping[str, Any] | Any, key: str, *, fallback: str | None = None
) -> Sequence[Mapping[str, Any]]:
    if not isinstance(program, Mapping):
        return ()
    value = program.get(key)
    if not value and fallback:
        value = program.get(fallback)
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    raise TypeError(f"{key} must be a mapping or sequence of mappings")


def _execute_case(
    test_name: str,
    case_name: str,
    definition: Mapping[str, Any],
    raw_case: Mapping[str, Any],
    evaluator: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None,
) -> PropertyCaseResult:
    inputs = raw_case.get("inputs")
    actual = _maybe_call(raw_case.get("function") or definition.get("function"), inputs)
    expected = _resolve_expected(raw_case, definition, inputs, actual)

    checker = _resolve_callable(
        raw_case,
        definition,
        ("check", "predicate", "validator"),
    )
    if checker is None and evaluator is not None:

        def checker(ctx: Mapping[str, Any]) -> Any:
            return evaluator(definition, raw_case)

    context = {
        "test": definition,
        "case": raw_case,
        "inputs": inputs,
        "actual": actual,
        "expected": expected,
    }

    if checker is None and native_bridge.should_use("checkers"):
        native_spec = _resolve_native_checker(raw_case, definition)
        if native_spec is not None:
            payload = _build_native_checker_payload(native_spec, context)
            if payload is not None:
                try:
                    native_response = native_bridge.run_checker(payload)
                except native_bridge.NativeBridgeError:
                    native_bridge.record_fallback("checkers")
                else:
                    ok = native_response.get("ok")
                    if isinstance(ok, bool):
                        message = native_response.get("message")
                        counterexample = None
                        if not ok:
                            counterexample = {
                                "inputs": context.get("inputs"),
                                "actual": context.get("actual"),
                                "expected": context.get("expected"),
                            }
                        return PropertyCaseResult(
                            test=test_name,
                            case=case_name,
                            status="passed" if ok else "failed",
                            message=message,
                            counterexample=counterexample,
                            details=context,
                        )
                    else:
                        native_bridge.record_fallback("checkers")
            else:
                native_bridge.record_fallback("checkers")

    if checker is None:
        if raw_case.get("expected") is None and definition.get("expected") is None:
            raise ValueError(
                f"Property '{test_name}' case '{case_name}' is missing a checker or expected value"
            )
        passed = actual == expected
        message = None if passed else f"expected {expected!r} but received {actual!r}"
        counterexample = (
            None if passed else {"inputs": inputs, "expected": expected, "actual": actual}
        )
        return PropertyCaseResult(
            test=test_name,
            case=case_name,
            status="passed" if passed else "failed",
            message=message,
            counterexample=counterexample,
            details=context,
        )

    outcome = _call_checker(checker, context)
    passed, message, counterexample = _interpret_outcome(outcome)
    if counterexample is None and not passed:
        counterexample = {"inputs": inputs, "expected": expected, "actual": context.get("actual")}
    return PropertyCaseResult(
        test=test_name,
        case=case_name,
        status="passed" if passed else "failed",
        message=message,
        counterexample=counterexample,
        details=context,
    )


def _maybe_call(function: Any, inputs: Any) -> Any:
    if function is None:
        return None
    if not callable(function):
        return function
    try:
        if isinstance(inputs, Mapping):
            return function(**inputs)
        if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
            return function(*inputs)
        return function(inputs)
    except TypeError:
        return function(inputs)


def _resolve_expected(
    raw_case: Mapping[str, Any],
    definition: Mapping[str, Any],
    inputs: Any,
    actual: Any,
) -> Any:
    if "expected" not in raw_case and "expected" not in definition:
        return raw_case.get("expected")
    source = (
        raw_case.get("expected")
        if raw_case.get("expected") is not None
        else definition.get("expected")
    )
    if callable(source):
        try:
            if isinstance(inputs, Mapping):
                return source(actual=actual, **inputs)
        except TypeError:
            pass
        return source(actual)
    return source


def _resolve_callable(
    raw_case: Mapping[str, Any],
    definition: Mapping[str, Any],
    keys: Sequence[str],
) -> Callable[[Mapping[str, Any]], Any] | None:
    for key in keys:
        if key in raw_case and raw_case[key] is not None:
            candidate = raw_case[key]
            if callable(candidate):
                return candidate
        if key in definition and definition[key] is not None:
            candidate = definition[key]
            if callable(candidate):
                return candidate
    return None


def _resolve_native_checker(
    raw_case: Mapping[str, Any], definition: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    for source in (raw_case, definition):
        candidate = source.get("native_checker")
        if candidate is None:
            candidate = source.get("check") or source.get("predicate")
        spec = _normalise_checker_spec(candidate)
        if spec is not None:
            return spec
    return None


def _normalise_checker_spec(candidate: Any) -> Mapping[str, Any] | None:
    if isinstance(candidate, str):
        return {"name": candidate}
    if isinstance(candidate, Mapping):
        name = candidate.get("name") or candidate.get("checker")
        if isinstance(name, str):
            spec = dict(candidate)
            spec["name"] = name
            spec.pop("checker", None)
            return spec
    return None


def _build_native_checker_payload(
    spec: Mapping[str, Any], context: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    name = spec.get("name")
    if not isinstance(name, str):
        return None
    values_path = spec.get("values", "actual")
    values = _resolve_context_path(context, values_path)
    if values is None:
        return None
    payload: dict[str, Any] = {
        "checker": name,
        "values": values,
    }
    if name == "histogram_matches":
        expected_path = spec.get("expected", "expected")
        expected = _resolve_context_path(context, expected_path)
        if expected is None:
            return None
        payload["expected"] = expected
    return payload


def _resolve_context_path(context: Mapping[str, Any], path: Any) -> Any:
    if not isinstance(path, str):
        return None
    value: Any = context
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return None
    return value


def _call_checker(checker: Callable[[Mapping[str, Any]], Any], context: Mapping[str, Any]) -> Any:
    try:
        return checker(**context)
    except TypeError:
        return checker(context)


def _interpret_outcome(outcome: Any) -> tuple[bool, str | None, Mapping[str, Any] | None]:
    if isinstance(outcome, PropertyCaseResult):
        return (
            outcome.status == "passed",
            outcome.message,
            outcome.counterexample,
        )
    if isinstance(outcome, bool):
        return outcome, None, None
    if isinstance(outcome, tuple):
        if len(outcome) == 0:
            return False, "empty outcome", None
        if len(outcome) == 1:
            return bool(outcome[0]), None, None
        counterexample = outcome[2] if len(outcome) > 2 else None
        return bool(outcome[0]), outcome[1], counterexample
    if isinstance(outcome, Mapping):
        status = outcome.get("status")
        if isinstance(status, str):
            passed = status.lower() in {"passed", "ok", "success"}
        else:
            passed = bool(outcome.get("passed") or outcome.get("ok"))
        message = outcome.get("message") or outcome.get("detail")
        counterexample = outcome.get("counterexample")
        return passed, message, counterexample
    return bool(outcome), None, None


def _summarise(results: Iterable[PropertyCaseResult]) -> dict[str, int | str]:
    counts = {"passed": 0, "failed": 0, "error": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    status = "ok" if counts.get("failed", 0) == 0 and counts.get("error", 0) == 0 else "failed"
    counts["status"] = status
    return counts
