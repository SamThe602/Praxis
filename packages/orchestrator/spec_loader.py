"""Structured spec loader for Praxis synthesiser tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml


@dataclass(slots=True, frozen=True)
class IOSpec:
    name: str
    type: str
    description: str | None = None
    shape: tuple[int, ...] | None = None
    optional: bool = False


@dataclass(slots=True, frozen=True)
class ExampleSpec:
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    description: str | None = None


@dataclass(slots=True, frozen=True)
class StructuredSpec:
    identifier: str
    metadata: Mapping[str, Any]
    inputs: tuple[IOSpec, ...]
    outputs: tuple[IOSpec, ...]
    constraints: tuple[str, ...]
    natural_prompt: str | None
    latency_target_ms: float | None
    operators: tuple[str, ...]
    reuse_pool: tuple[str, ...]
    examples: tuple[ExampleSpec, ...]
    goal_tokens: tuple[str, ...]
    input_types: Mapping[str, str]
    raw: Mapping[str, Any]


def load_spec(path: str | Path) -> StructuredSpec:
    """Load a structured YAML spec and enforce a minimal schema."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"spec file not found: {source_path}")

    data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, MutableMapping):
        raise TypeError("structured spec must be a mapping")
    return _build_structured_spec(dict(data))


def parse_structured_spec(document: Mapping[str, Any]) -> StructuredSpec:
    """Normalise an in-memory mapping into :class:`StructuredSpec`."""

    if not isinstance(document, Mapping):
        raise TypeError("structured spec payload must be a mapping")
    return _build_structured_spec(dict(document))


def _parse_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise TypeError(f"expected a sequence of strings, received {type(value).__name__}")


def _parse_io(value: Any, *, require_name: bool, default_prefix: str) -> tuple[IOSpec, ...]:
    if value is None:
        raise ValueError("spec must declare at least one input/output")
    items: list[Mapping[str, Any]]
    if isinstance(value, Mapping):
        items = [value]
    elif isinstance(value, Sequence):
        items = []
        for entry in value:
            if not isinstance(entry, Mapping):
                raise TypeError("IO entries must be mappings")
            items.append(entry)
    else:
        raise TypeError("inputs/outputs must be a mapping or a list of mappings")

    result: list[IOSpec] = []
    for index, entry in enumerate(items):
        name = entry.get("name")
        if require_name and not name:
            raise ValueError("input entries require a 'name'")
        if not name:
            name = f"{default_prefix}_{index}"
        type_value = entry.get("type")
        if not isinstance(type_value, str):
            raise ValueError("inputs/outputs must declare a string 'type'")
        description = entry.get("description")
        shape_value = entry.get("shape")
        shape = None
        if shape_value is not None:
            if isinstance(shape_value, Sequence) and not isinstance(shape_value, str):
                shape = tuple(int(dim) for dim in shape_value)
            else:
                raise TypeError("shape must be a sequence of integers")
        optional = bool(entry.get("optional", False))
        result.append(
            IOSpec(
                name=str(name),
                type=type_value,
                description=str(description) if isinstance(description, str) else None,
                shape=shape,
                optional=optional,
            )
        )
    return tuple(result)


def _parse_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("user_text", "text", "prompt"):
            text = value.get(key)
            if isinstance(text, str):
                return text
        return str(value)
    raise TypeError("natural_prompt must be a string or mapping")


def _parse_latency(data: Mapping[str, Any]) -> float | None:
    for key in ("latency_target_ms", "latency_target", "latency_budget"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid latency target: {value!r}") from exc
    return None


def _parse_examples(value: Any) -> tuple[ExampleSpec, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence):
        raise TypeError("examples must be a list")
    examples: list[ExampleSpec] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise TypeError("example entries must be mappings")
        inputs = entry.get("input") or entry.get("inputs") or {}
        outputs = entry.get("output") or entry.get("outputs") or {}
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            raise TypeError("example inputs/outputs must be mappings")
        description = entry.get("description")
        examples.append(
            ExampleSpec(
                inputs=inputs,
                outputs=outputs,
                description=str(description) if isinstance(description, str) else None,
            )
        )
    return tuple(examples)


def _build_goals(
    inputs: tuple[IOSpec, ...],
    outputs: tuple[IOSpec, ...],
    constraints: tuple[str, ...],
) -> tuple[str, ...]:
    goals: list[str] = []
    for spec in inputs:
        goals.append(f"use_input:{spec.name}")
    for index, spec in enumerate(outputs):
        label = spec.name or f"result_{index}"
        goals.append(f"produce_output:{label}")
    for item in constraints:
        goals.append(f"constraint:{item}")
    return tuple(goals)


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"spec missing required string field '{key}'")
    return value


def _get_mapping(
    data: Mapping[str, Any], key: str, *, default: Mapping[str, Any]
) -> Mapping[str, Any]:
    value = data.get(key, default)
    if value is default:
        return default
    if not isinstance(value, Mapping):
        raise TypeError(f"field '{key}' must be a mapping")
    return value


def _build_structured_spec(document: MutableMapping[str, Any]) -> StructuredSpec:
    identifier = _require_str(document, "id")
    metadata = dict(_get_mapping(document, "metadata", default={}))
    inputs = _parse_io(document.get("inputs"), require_name=True, default_prefix="arg")
    outputs = _parse_io(document.get("outputs"), require_name=False, default_prefix="result")
    constraints = _parse_sequence(document.get("constraints"))
    natural_prompt = _parse_prompt(document.get("natural_prompt"))
    latency_target = _parse_latency(document)
    operators = tuple(_parse_sequence(document.get("operators")))
    reuse_pool = tuple(
        _parse_sequence(document.get("reuse")) or _parse_sequence(document.get("snippets"))
    )
    examples = tuple(_parse_examples(document.get("examples")))

    goal_tokens = _build_goals(inputs, outputs, constraints)
    input_types = {io.name: io.type for io in inputs}

    return StructuredSpec(
        identifier=identifier,
        metadata=metadata,
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
        natural_prompt=natural_prompt,
        latency_target_ms=latency_target,
        operators=operators,
        reuse_pool=reuse_pool,
        examples=examples,
        goal_tokens=goal_tokens,
        input_types=input_types,
        raw=document,
    )


__all__ = ["IOSpec", "ExampleSpec", "StructuredSpec", "load_spec", "parse_structured_spec"]
