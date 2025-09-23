"""Natural language translator decoder.

The decoder exposes a deterministic rule-based path (used in tests and when no
model credentials are configured) while still supporting grammar hooks that a
future neural backend could honour. Outputs include both the structured spec
and a YAML serialisation so downstream components can consume whichever they
prefer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from packages.telemetry import hooks as telemetry_hooks
from packages.telemetry import metrics as telemetry_metrics

from . import cache, confidence, glossary, schema_validator

Hook = Callable[[str, Dict[str, Any]], Dict[str, Any] | None]


@dataclass(slots=True)
class TranslationResult:
    """Container for decoder outputs."""

    prompt: str
    locale: str
    spec: Dict[str, Any]
    yaml: str
    confidence: float
    validation: schema_validator.ValidationResult
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cache_payload(self) -> Dict[str, Any]:
        return {
            "spec": self.spec,
            "yaml": self.yaml,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


_RULE_BUILDERS: Dict[str, Callable[[str, glossary.Glossary, str], Dict[str, Any]]] = {}


def _register_rule(
    name: str,
) -> Callable[
    [Callable[[str, glossary.Glossary, str], Dict[str, Any]]],
    Callable[[str, glossary.Glossary, str], Dict[str, Any]],
]:
    def decorator(
        func: Callable[[str, glossary.Glossary, str], Dict[str, Any]],
    ) -> Callable[[str, glossary.Glossary, str], Dict[str, Any]]:
        _RULE_BUILDERS[name] = func
        return func

    return decorator


@_register_rule("array_sort")
def _build_sort_spec(prompt: str, terms: glossary.Glossary, locale: str) -> Dict[str, Any]:
    return {
        "id": "array_sort",
        "natural_prompt": prompt,
        "description": terms.definition_for("array_sort"),
        "inputs": [
            {"name": "arr", "type": "vector<int>", "description": "Input array"},
        ],
        "outputs": {"type": "vector<int>", "description": "Sorted array"},
        "constraints": ["result must be ascending"],
        "examples": [
            {"input": {"arr": [3, 1, 2]}, "output": {"result": [1, 2, 3]}},
        ],
        "metadata": {"difficulty": "easy", "tags": ["arrays", "sorting"], "locale": locale},
        "proof_required": False,
        "latency_target_ms": 100,
    }


@_register_rule("array_reverse")
def _build_reverse_spec(prompt: str, terms: glossary.Glossary, locale: str) -> Dict[str, Any]:
    return {
        "id": "array_reverse",
        "natural_prompt": prompt,
        "description": terms.definition_for("array_reverse"),
        "inputs": [
            {"name": "arr", "type": "vector<int>", "description": "Input array"},
        ],
        "outputs": {"type": "vector<int>", "description": "Reversed array"},
        "constraints": [],
        "examples": [
            {"input": {"arr": [1, 2, 3]}, "output": {"result": [3, 2, 1]}},
        ],
        "metadata": {"difficulty": "easy", "tags": ["arrays"], "locale": locale},
        "proof_required": False,
        "latency_target_ms": 100,
    }


@_register_rule("graph_path")
def _build_graph_spec(prompt: str, terms: glossary.Glossary, locale: str) -> Dict[str, Any]:
    return {
        "id": "graph_path",
        "natural_prompt": prompt,
        "description": terms.definition_for("graph_path"),
        "inputs": [
            {"name": "graph", "type": "struct", "description": "Graph adjacency description"},
            {"name": "start", "type": "string", "description": "Source node"},
            {"name": "goal", "type": "string", "description": "Destination node"},
        ],
        "outputs": {"type": "vector<string>", "description": "Path node sequence"},
        "constraints": ["path must exist", "return empty if no path"],
        "examples": [
            {
                "input": {
                    "graph": {"A": ["B"], "B": ["C"], "C": []},
                    "start": "A",
                    "goal": "C",
                },
                "output": {"result": ["A", "B", "C"]},
            }
        ],
        "metadata": {"difficulty": "medium", "tags": ["graphs"], "locale": locale},
        "proof_required": True,
        "latency_target_ms": 200,
    }


@_register_rule("histogram")
def _build_histogram_spec(prompt: str, terms: glossary.Glossary, locale: str) -> Dict[str, Any]:
    return {
        "id": "histogram",
        "natural_prompt": prompt,
        "description": terms.definition_for("histogram"),
        "inputs": [
            {"name": "values", "type": "vector<int>", "description": "Values to count"},
        ],
        "outputs": {"type": "map<int,int>", "description": "Frequency map"},
        "constraints": ["must cover all seen values"],
        "examples": [
            {"input": {"values": [1, 1, 2]}, "output": {"result": {"1": 2, "2": 1}}},
        ],
        "metadata": {"difficulty": "easy", "tags": ["statistics"], "locale": locale},
        "proof_required": False,
        "latency_target_ms": 100,
    }


@_register_rule("matrix_mult")
def _build_matrix_mult_spec(prompt: str, terms: glossary.Glossary, locale: str) -> Dict[str, Any]:
    return {
        "id": "matrix_mult",
        "natural_prompt": prompt,
        "description": terms.definition_for("matrix_mult"),
        "inputs": [
            {"name": "a", "type": "matrix<int>", "description": "Left matrix"},
            {"name": "b", "type": "matrix<int>", "description": "Right matrix"},
        ],
        "outputs": {"type": "matrix<int>", "description": "Product matrix"},
        "constraints": ["matrices must be conformable"],
        "examples": [
            {
                "input": {"a": [[1, 0], [0, 1]], "b": [[2, 3], [4, 5]]},
                "output": {"result": [[2, 3], [4, 5]]},
            }
        ],
        "metadata": {"difficulty": "medium", "tags": ["matrices"], "locale": locale},
        "proof_required": True,
        "latency_target_ms": 250,
    }


def _build_generic_spec(prompt: str, terms: glossary.Glossary, locale: str) -> Dict[str, Any]:
    return {
        "id": "generic_task",
        "natural_prompt": prompt,
        "description": "Generic task inferred from natural prompt.",
        "inputs": [
            {"name": "input", "type": "struct", "description": "User-provided input"},
        ],
        "outputs": {"type": "struct", "description": "Expected output"},
        "constraints": [],
        "examples": [],
        "metadata": {"difficulty": "unknown", "tags": [], "locale": locale},
        "proof_required": False,
        "latency_target_ms": 300,
    }


_KEYWORD_RULES: Dict[str, str] = {
    "sort": "array_sort",
    "order": "array_sort",
    "ascending": "array_sort",
    "reverse": "array_reverse",
    "flip": "array_reverse",
    "graph": "graph_path",
    "path": "graph_path",
    "histogram": "histogram",
    "frequency": "histogram",
    "matrix": "matrix_mult",
    "product": "matrix_mult",
}


def decode(
    prompt: str,
    *,
    locale: str = "en",
    config: Mapping[str, Any] | None = None,
) -> TranslationResult:
    """Translate a natural language ``prompt`` into a structured spec.

    Parameters
    ----------
    prompt:
        Natural language task description.
    locale:
        Language tag for the prompt; currently only ``"en"`` is supported but we
        keep the argument to avoid churn later.
    config:
        Optional mapping configuring cache usage, glossary overrides, grammar
        restrictions, custom hooks, or a ``model_key`` for future neural models.
    """

    config = dict(config or {})
    hooks = _normalise_hooks(config.get("hooks"))
    grammar = config.get("grammar") if isinstance(config.get("grammar"), Mapping) else None
    glossary_overrides = config.get("glossary_overrides")
    model_key = config.get("model_key")

    glossary_obj = glossary.build_glossary(
        glossary_overrides if isinstance(glossary_overrides, Mapping) else None
    )

    if config.get("cache_enabled", True):
        cached_payload = cache.get(prompt, locale)
        if cached_payload:
            spec = dict(cached_payload.get("spec", {}))
            yaml_text = str(cached_payload.get("yaml", ""))
            confidence_value = float(cached_payload.get("confidence", 0.0))
            metadata = dict(cached_payload.get("metadata", {}))
            validation = schema_validator.validate(spec)
            result = TranslationResult(
                prompt=prompt,
                locale=locale,
                spec=spec,
                yaml=yaml_text,
                confidence=confidence_value,
                validation=validation,
                cache_hit=True,
                metadata=metadata,
            )
            _record_translator_telemetry(result)
            return result

    spec = _deterministic_translation(prompt, glossary_obj, locale)
    if grammar:
        spec = _apply_grammar(spec, grammar)

    for hook in hooks:
        maybe_updated = hook(prompt, dict(spec))
        if maybe_updated is not None:
            spec = dict(maybe_updated)

    validation = schema_validator.validate(spec)
    validation.raise_for_errors()

    confidence_value = confidence.score(spec, prompt)
    yaml_text = _dict_to_yaml(spec)

    metadata = {
        "inference_mode": "rule_based" if not model_key else "mocked-model",
        "glossary_terms": [term.canonical for term in glossary_obj.terms()],
    }

    result = TranslationResult(
        prompt=prompt,
        locale=locale,
        spec=spec,
        yaml=yaml_text,
        confidence=confidence_value,
        validation=validation,
        metadata=metadata,
    )

    if config.get("cache_enabled", True):
        cache.set(prompt, locale, result.to_cache_payload())

    _record_translator_telemetry(result)
    return result


def _normalise_hooks(raw: Any) -> Sequence[Hook]:
    if raw is None:
        return ()
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        hooks: list[Hook] = []
        for item in raw:
            if callable(item):
                hooks.append(_wrap_hook(item))
        return tuple(hooks)
    if callable(raw):
        return (_wrap_hook(raw),)
    return ()


def _wrap_hook(func: Callable[[str, Dict[str, Any]], Dict[str, Any] | None]) -> Hook:
    def wrapped(prompt: str, draft: Dict[str, Any]) -> Dict[str, Any] | None:
        return func(prompt, draft)

    return wrapped


def _deterministic_translation(
    prompt: str, terms: glossary.Glossary, locale: str
) -> Dict[str, Any]:
    lowered = prompt.lower()
    matched = terms.match(lowered)
    if not matched:
        for keyword, rule_name in _KEYWORD_RULES.items():
            if keyword in lowered:
                matched = rule_name
                break
    builder = _RULE_BUILDERS.get(matched or "") if matched else None
    if not builder:
        builder = _build_generic_spec
    return builder(prompt, terms, locale)


def _apply_grammar(spec: Dict[str, Any], grammar: Mapping[str, Any]) -> Dict[str, Any]:
    updated: Dict[str, Any] = dict(spec)
    allowed_ids = grammar.get("allowed_ids")
    if isinstance(allowed_ids, Iterable) and not isinstance(allowed_ids, (str, bytes)):
        allowed = [str(item) for item in allowed_ids if item]
        if allowed:
            if updated.get("id") not in allowed:
                updated["id"] = allowed[0]
    defaults = grammar.get("defaults")
    if isinstance(defaults, Mapping):
        for key, value in defaults.items():
            updated.setdefault(str(key), value)
    required = grammar.get("required_fields")
    if isinstance(required, Iterable) and not isinstance(required, (str, bytes)):
        missing = [field for field in required if field not in updated]
        if missing:
            raise ValueError(f"Grammar requires missing fields: {', '.join(missing)}")
    return updated


def _dict_to_yaml(data: Dict[str, Any]) -> str:
    lines = _emit_yaml(data, indent=0)
    return "\n".join(lines) + "\n"


def _emit_yaml(value: Any, indent: int) -> list[str]:
    indent_str = "  " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key in sorted(value.keys()):
            item = value[key]
            if isinstance(item, (Mapping, list)):
                lines.append(f"{indent_str}{key}:")
                lines.extend(_emit_yaml(item, indent + 1))
            else:
                lines.append(f"{indent_str}{key}: {_format_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (Mapping, list)):
                lines.append(f"{indent_str}-")
                lines.extend(_emit_yaml(item, indent + 1))
            else:
                lines.append(f"{indent_str}- {_format_scalar(item)}")
        if not value:
            lines.append(f"{indent_str}[]")
        return lines
    return [f"{indent_str}{_format_scalar(value)}"]


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ":#\n\"'"):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _record_translator_telemetry(result: TranslationResult) -> None:
    """Emit metrics and hooks for translator completions."""

    try:
        spec_id = "unknown"
        if isinstance(result.spec, Mapping):
            spec_id = str(result.spec.get("id", "unknown"))
        telemetry_metrics.emit(
            "praxis.translator.accuracy",
            result.confidence,
            tags={
                "locale": result.locale,
                "cache": "hit" if result.cache_hit else "miss",
            },
            extra={"spec_id": spec_id},
        )
        telemetry_hooks.dispatch(
            telemetry_hooks.TRANSLATOR_DECODE_COMPLETED,
            {
                "prompt": result.prompt,
                "spec_id": spec_id,
                "locale": result.locale,
                "confidence": result.confidence,
                "cache_hit": result.cache_hit,
                "validation_ok": bool(result.validation.ok),
            },
        )
    except Exception:  # pragma: no cover - telemetry must not break decoding
        telemetry_hooks.dispatch(
            "translator.telemetry_error",
            {"locale": result.locale, "cache_hit": result.cache_hit},
        )
