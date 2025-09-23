"""Metamorphic testing helpers for the verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import native_bridge


@dataclass(slots=True, frozen=True)
class MetamorphicCaseResult:
    """Single metamorphic relation evaluation."""

    test: str
    variant: str
    status: str
    message: str | None = None
    counterexample: Mapping[str, Any] | None = None
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "test": self.test,
            "variant": self.variant,
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
class MetamorphicTestRun:
    results: tuple[MetamorphicCaseResult, ...]
    summary: dict[str, int | str]
    feedback: tuple[dict[str, Any], ...]

    @property
    def ok(self) -> bool:
        return self.summary.get("status") == "ok"


__all__ = ["MetamorphicCaseResult", "MetamorphicTestRun", "run_metamorphic_tests"]


def run_metamorphic_tests(
    program: Mapping[str, Any] | Any,
    *,
    evaluator: (
        Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Any] | None
    ) = None,
) -> MetamorphicTestRun:
    """Execute metamorphic relations declared on ``program``."""

    definitions = _collect_definitions(program, "metamorphic_tests", fallback="metamorphic")
    results: list[MetamorphicCaseResult] = []
    for test_index, definition in enumerate(definitions):
        name = str(definition.get("name") or f"metamorphic_{test_index}")
        base_function = definition.get("function")
        cases = definition.get("cases") or ()
        for case_index, case in enumerate(cases):
            try:
                base_inputs = case.get("base") or case.get("inputs")
                function = case.get("function") or base_function
                base_output = _maybe_call(function, base_inputs)
                variants = _normalise_variants(case.get("variants") or ())
                if not variants and evaluator is None:
                    raise ValueError(
                        f"Metamorphic test '{name}' case {case_index} declares no variants"
                    )
                if not variants:
                    variants = evaluator(definition, case)  # type: ignore[assignment]
                    variants = _normalise_variants(variants)
                for variant_index, raw_variant in enumerate(variants):
                    variant_name = str(raw_variant.get("name") or f"{name}.variant_{variant_index}")
                    result = _execute_variant(
                        name,
                        variant_name,
                        definition,
                        case,
                        raw_variant,
                        base_inputs,
                        base_output,
                        function,
                        evaluator,
                    )
                    results.append(result)
            except Exception as exc:
                results.append(
                    MetamorphicCaseResult(
                        test=name,
                        variant=f"{name}.case_{case_index}",
                        status="error",
                        message=str(exc),
                        counterexample={"exception": repr(exc)},
                        details={"case": case},
                    )
                )

    summary = _summarise(results)
    feedback = tuple(
        {
            "kind": "metamorphic_failure",
            "test": result.test,
            "variant": result.variant,
            "message": result.message,
            "counterexample": result.counterexample,
        }
        for result in results
        if result.status != "passed"
    )
    return MetamorphicTestRun(results=tuple(results), summary=summary, feedback=feedback)


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


def _execute_variant(
    test_name: str,
    variant_name: str,
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    raw_variant: Mapping[str, Any],
    base_inputs: Any,
    base_output: Any,
    function: Any,
    evaluator: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Any] | None,
) -> MetamorphicCaseResult:
    variant_inputs = raw_variant.get("inputs")
    if variant_inputs is None:
        transform = (
            raw_variant.get("transform")
            or raw_variant.get("mutate")
            or case.get("transform")
            or definition.get("transform")
        )
        if callable(transform):
            variant_inputs = transform(base_inputs)
        else:
            variant_inputs = base_inputs
    variant_function = (
        raw_variant.get("function")
        or case.get("function")
        or definition.get("function")
        or function
    )
    variant_output = _maybe_call(variant_function, variant_inputs)

    relation = _resolve_callable(
        raw_variant,
        case,
        definition,
        ("relation", "check", "predicate"),
    )
    context = {
        "test": definition,
        "case": case,
        "variant": raw_variant,
        "base": {"inputs": base_inputs, "output": base_output},
        "candidate": {"inputs": variant_inputs, "output": variant_output},
    }
    if relation is None and native_bridge.should_use("metamorphic"):
        native_spec = _resolve_native_relation(raw_variant, case, definition)
        if native_spec is not None:
            name, spec = native_spec
            payload = _build_native_relation_payload(
                name,
                spec,
                base_inputs=base_inputs,
                base_output=base_output,
                variant_inputs=variant_inputs,
                variant_output=variant_output,
            )
            if payload is not None:
                try:
                    native_response = native_bridge.evaluate_relation(payload)
                except native_bridge.NativeBridgeError:
                    native_bridge.record_fallback("metamorphic")
                else:
                    ok = native_response.get("ok")
                    if isinstance(ok, bool):
                        message = native_response.get("message")
                        counterexample = None
                        if not ok:
                            counterexample = {
                                "base": {"inputs": base_inputs, "output": base_output},
                                "variant": {
                                    "inputs": variant_inputs,
                                    "output": variant_output,
                                },
                            }
                        return MetamorphicCaseResult(
                            test=test_name,
                            variant=variant_name,
                            status="passed" if ok else "failed",
                            message=message,
                            counterexample=counterexample,
                            details=context,
                        )
                    else:
                        native_bridge.record_fallback("metamorphic")
            else:
                native_bridge.record_fallback("metamorphic")
    if relation is None and evaluator is not None:

        def relation(ctx: Mapping[str, Any]) -> Any:
            return evaluator(definition, case, raw_variant)

    if relation is None:
        passed = variant_output == base_output
        message = (
            None
            if passed
            else f"expected invariant output {base_output!r}, received {variant_output!r}"
        )
        counterexample = (
            None
            if passed
            else {
                "base": {"inputs": base_inputs, "output": base_output},
                "variant": {"inputs": variant_inputs, "output": variant_output},
            }
        )
        return MetamorphicCaseResult(
            test=test_name,
            variant=variant_name,
            status="passed" if passed else "failed",
            message=message,
            counterexample=counterexample,
            details=context,
        )

    outcome = _call_relation(relation, context)
    passed, message, counterexample = _interpret_outcome(outcome)
    if counterexample is None and not passed:
        counterexample = {
            "base": {"inputs": base_inputs, "output": base_output},
            "variant": {"inputs": variant_inputs, "output": variant_output},
        }
    return MetamorphicCaseResult(
        test=test_name,
        variant=variant_name,
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


def _resolve_callable(
    variant: Mapping[str, Any],
    case: Mapping[str, Any],
    definition: Mapping[str, Any],
    keys: Sequence[str],
) -> Callable[[Mapping[str, Any]], Any] | None:
    for key in keys:
        for source in (variant, case, definition):
            if key in source and source[key] is not None:
                candidate = source[key]
                if callable(candidate):
                    return candidate
    return None


def _resolve_native_relation(
    variant: Mapping[str, Any],
    case: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]] | None:
    for source in (variant, case, definition):
        candidate = source.get("native_relation")
        if candidate is None:
            candidate = source.get("relation")
        spec = _normalise_relation_spec(candidate)
        if spec is not None:
            return spec
    return None


def _normalise_relation_spec(candidate: Any) -> tuple[str, Mapping[str, Any]] | None:
    if isinstance(candidate, str):
        return candidate, {}
    if isinstance(candidate, Mapping):
        name = candidate.get("name") or candidate.get("relation") or candidate.get("type")
        if isinstance(name, str):
            spec = dict(candidate)
            spec.setdefault("name", name)
            spec.pop("relation", None)
            spec.pop("type", None)
            return name, spec
    return None


def _build_native_relation_payload(
    name: str,
    spec: Mapping[str, Any],
    *,
    base_inputs: Any,
    base_output: Any,
    variant_inputs: Any,
    variant_output: Any,
) -> Mapping[str, Any] | None:
    if name == "permutation_invariance":
        variants = spec.get("variants")
        if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
            variants_payload = list(variants)
        else:
            variants_payload = [
                {
                    "inputs": variant_inputs,
                    "output": variant_output,
                }
            ]
        return {
            "relation": "permutation_invariance",
            "base_inputs": spec.get("base_inputs", base_inputs),
            "base_output": spec.get("base_output", base_output),
            "variants": variants_payload,
        }
    if name == "monotonicity":
        values = spec.get("values")
        if values is None:
            return None
        return {
            "relation": "monotonicity",
            "values": values,
            "strict": bool(spec.get("strict", False)),
        }
    if name == "idempotence":
        twice = spec.get("twice")
        if twice is None:
            return None
        return {
            "relation": "idempotence",
            "value": spec.get("value", base_output),
            "once": spec.get("once", variant_output),
            "twice": twice,
        }
    if name == "inverse_property":
        samples = spec.get("samples")
        if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
            return {"relation": "inverse_property", "samples": list(samples)}
        return None
    return None


def _call_relation(relation: Callable[[Mapping[str, Any]], Any], context: Mapping[str, Any]) -> Any:
    try:
        return relation(**context)
    except TypeError:
        return relation(context)


def _interpret_outcome(outcome: Any) -> tuple[bool, str | None, Mapping[str, Any] | None]:
    if isinstance(outcome, MetamorphicCaseResult):
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


def _summarise(results: Iterable[MetamorphicCaseResult]) -> dict[str, int | str]:
    counts = {"passed": 0, "failed": 0, "error": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    status = "ok" if counts.get("failed", 0) == 0 and counts.get("error", 0) == 0 else "failed"
    counts["status"] = status
    return counts


def _normalise_variants(variants: Any) -> tuple[Mapping[str, Any], ...]:
    if variants is None:
        return ()
    if isinstance(variants, Mapping):
        return (variants,)
    if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
        result: list[Mapping[str, Any]] = []
        for entry in variants:
            if isinstance(entry, Mapping):
                result.append(entry)
            else:
                result.append({"inputs": entry})
        return tuple(result)
    return ({"inputs": variants},)
