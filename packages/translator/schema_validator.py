"""Schema validator for translator outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableSequence

SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class ValidationResult:
    """Validation outcome returned by :func:`validate`."""

    ok: bool
    errors: list[str]

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


_REQUIRED_TOP_LEVEL = ("id", "natural_prompt", "inputs", "outputs", "metadata")


def validate(candidate: Mapping[str, Any]) -> ValidationResult:
    """Validate a translated specification against the expected schema."""

    errors: MutableSequence[str] = []

    if not isinstance(candidate, Mapping):
        errors.append("candidate must be a mapping")
        return ValidationResult(ok=False, errors=list(errors))

    for field in _REQUIRED_TOP_LEVEL:
        if field not in candidate:
            errors.append(f"missing top level field: {field}")

    if "inputs" in candidate and not isinstance(candidate["inputs"], list):
        errors.append("inputs must be a list")
    else:
        for idx, item in enumerate(candidate.get("inputs", [])):
            if not isinstance(item, Mapping):
                errors.append(f"inputs[{idx}] must be a mapping")
                continue
            if "name" not in item or "type" not in item:
                errors.append(f"inputs[{idx}] missing name/type")

    outputs = candidate.get("outputs")
    if outputs is None or not isinstance(outputs, Mapping):
        errors.append("outputs must be a mapping")
    elif "type" not in outputs:
        errors.append("outputs must include type")

    metadata = candidate.get("metadata", {})
    if not isinstance(metadata, Mapping):
        errors.append("metadata must be a mapping")
    elif "difficulty" not in metadata:
        errors.append("metadata must include difficulty")

    constraints = candidate.get("constraints", [])
    if constraints is not None:
        if not isinstance(constraints, list):
            errors.append("constraints must be a list when provided")
        else:
            for idx, item in enumerate(constraints):
                if not isinstance(item, str):
                    errors.append(f"constraints[{idx}] must be a string")

    proof_required = candidate.get("proof_required")
    if proof_required is not None and not isinstance(proof_required, bool):
        errors.append("proof_required must be a boolean when provided")

    latency_target = candidate.get("latency_target_ms")
    if latency_target is not None:
        if not isinstance(latency_target, int) or latency_target <= 0:
            errors.append("latency_target_ms must be a positive integer")

    examples = candidate.get("examples", [])
    if not isinstance(examples, list):
        errors.append("examples must be a list when provided")
    else:
        for idx, example in enumerate(examples):
            if not isinstance(example, Mapping):
                errors.append(f"examples[{idx}] must be a mapping")
                continue
            for field in ("input", "output"):
                if field not in example:
                    errors.append(f"examples[{idx}] missing field: {field}")

    ok = not errors
    return ValidationResult(ok=ok, errors=list(errors))
