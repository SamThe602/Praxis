"""Canonical JSON serializer for Praxis DSL AST nodes.

The serializer produces deterministic output so that golden files, cache keys,
and retrieval indices remain stable across runs.  Each node receives a
content-addressed identifier derived from its structural JSON encoding; the
deserializer recomputes the hash to guarantee integrity.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from . import ast


def to_json(node: ast.Node, *, ensure_ascii: bool = True) -> str:
    """Serialize ``node`` into canonical JSON."""

    payload = _serialize_node(node)
    return json.dumps(payload, indent=2, separators=(",", ": "), ensure_ascii=ensure_ascii)


def from_json(payload: str) -> ast.Node:
    """Deserialize JSON back into an AST node, validating all node hashes."""

    raw = json.loads(payload)
    return _deserialize_node(raw)


# ---------------------------------------------------------------------------
# Serialization helpers


def _serialize_node(node: ast.Node) -> OrderedDict[str, Any]:
    data: OrderedDict[str, Any] = OrderedDict()
    data["type"] = node.node_type
    if node.span is not None:
        data["span"] = node.span.to_tuple()
    for field_info in fields(node):
        if field_info.name in {"span", "metadata"}:
            continue
        value = getattr(node, field_info.name)
        data[field_info.name] = _serialize_value(node, field_info.name, value)
    data["id"] = _hash_payload(data)
    return data


def _serialize_value(node: ast.Node, field_name: str, value: Any) -> Any:
    if isinstance(node, ast.Conditional) and field_name == "branches":
        return [
            OrderedDict(
                [
                    ("condition", _serialize_optional(condition)),
                    ("body", _serialize_generic(body)),
                ]
            )
            for condition, body in value
        ]
    if isinstance(node, ast.Conditional) and field_name == "arms":
        return [
            OrderedDict(
                [
                    ("pattern", _serialize_pattern(arm.pattern)),
                    ("guard", _serialize_optional(arm.guard)),
                    ("body", _serialize_generic(arm.body)),
                ]
            )
            for arm in value
        ]
    if isinstance(node, ast.Literal) and field_name == "value":
        if node.literal_type == "map":
            return [
                OrderedDict(
                    [
                        ("key", _serialize_generic(key)),
                        ("value", _serialize_generic(val)),
                    ]
                )
                for key, val in value
            ]
        return _serialize_generic(value)
    return _serialize_generic(value)


def _serialize_generic(value: Any) -> Any:
    if isinstance(value, ast.Node):
        return _serialize_node(value)
    if isinstance(value, ast.Parameter):
        payload = OrderedDict(
            [
                ("name", value.name),
                ("type_annotation", value.type_annotation),
                ("default", _serialize_optional(value.default)),
            ]
        )
        payload["id"] = _hash_payload(payload)
        return payload
    if isinstance(value, ast.Pattern):
        payload = OrderedDict([("text", value.text)])
        payload["id"] = _hash_payload(payload)
        return payload
    if isinstance(value, ast.Span):
        return value.to_tuple()
    if isinstance(value, list):
        return [_serialize_generic(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_generic(item) for item in value]
    if isinstance(value, Mapping):
        return OrderedDict((key, _serialize_generic(val)) for key, val in value.items())
    return value


def _serialize_optional(value: Any) -> Any:
    if value is None:
        return None
    return _serialize_generic(value)


def _serialize_pattern(pattern: ast.Pattern) -> OrderedDict[str, Any]:
    payload = OrderedDict([("text", pattern.text)])
    payload["id"] = _hash_payload(payload)
    return payload


def _hash_payload(data: Mapping[str, Any]) -> str:
    normalized = json.dumps(
        _strip_ids(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _strip_ids(data: Any) -> Any:
    if isinstance(data, Mapping):
        return {key: _strip_ids(value) for key, value in data.items() if key != "id"}
    if isinstance(data, list):
        return [_strip_ids(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# Deserialization helpers


NODE_TYPES: dict[str, type[ast.Node]] = {
    cls.__name__: cls
    for cls in (
        ast.Module,
        ast.FunctionDecl,
        ast.Contract,
        ast.Block,
        ast.Let,
        ast.Assign,
        ast.Loop,
        ast.Conditional,
        ast.MatchArm,
        ast.Call,
        ast.BuiltinCall,
        ast.Lambda,
        ast.Literal,
        ast.BinaryOp,
        ast.UnaryOp,
        ast.Comprehension,
    )
}


def _deserialize_node(data: Mapping[str, Any]) -> ast.Node:
    _verify_hash(data)
    node_type = data["type"]
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unknown node type '{node_type}'")
    cls = NODE_TYPES[node_type]
    kwargs: dict[str, Any] = {}
    span_data = data.get("span")
    if span_data is not None:
        kwargs["span"] = ast.Span(*span_data)
    for field_info in fields(cls):
        if field_info.name in {"span", "metadata"}:
            continue
        raw_value = data.get(field_info.name)
        kwargs[field_info.name] = _deserialize_field(cls, field_info.name, raw_value, data)
    return cls(**kwargs)


def _deserialize_field(
    cls: type[ast.Node], field_name: str, raw_value: Any, payload: Mapping[str, Any]
) -> Any:
    if raw_value is None:
        return None
    if cls is ast.Conditional and field_name == "branches":
        return [
            (
                _deserialize_optional(entry.get("condition")),
                _require_block(_deserialize_node(entry["body"])),
            )
            for entry in raw_value
        ]
    if cls is ast.Conditional and field_name == "arms":
        return [
            ast.MatchArm(
                pattern=_deserialize_pattern(entry["pattern"]),
                guard=_deserialize_optional(entry.get("guard")),
                body=_require_block(_deserialize_node(entry["body"])),
            )
            for entry in raw_value
        ]
    if cls is ast.Literal and field_name == "value":
        literal_type = payload.get("literal_type")
        if literal_type == "map":
            return [
                (
                    _deserialize_generic(item["key"]),
                    _deserialize_generic(item["value"]),
                )
                for item in raw_value
            ]
        return _deserialize_generic(raw_value)
    return _deserialize_generic(raw_value)


def _deserialize_generic(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "type" in value:
            return _deserialize_node(value)
        if "name" in value and "type_annotation" in value:
            _verify_hash(value)
            default = _deserialize_optional(value.get("default"))
            return ast.Parameter(
                name=value["name"],
                type_annotation=value.get("type_annotation"),
                default=default,
            )
        if "text" in value and len(value) <= 2:
            _verify_hash(value)
            return ast.Pattern(text=value["text"])
        return {key: _deserialize_generic(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_deserialize_generic(item) for item in value]
    if is_dataclass(value):
        raise TypeError("Unexpected dataclass instance in serialized payload")
    return value


def _deserialize_optional(value: Any) -> Any:
    if value is None:
        return None
    return _deserialize_generic(value)


def _deserialize_pattern(data: Mapping[str, Any]) -> ast.Pattern:
    _verify_hash(data)
    return ast.Pattern(text=data["text"])


def _verify_hash(data: Mapping[str, Any]) -> None:
    stored = data.get("id")
    if stored is None:
        raise ValueError("Serialized node is missing 'id'")
    computed = _hash_payload(data)
    if stored != computed:
        raise ValueError("Serialized node failed integrity check")


def _require_block(node: ast.Node) -> ast.Block:
    if not isinstance(node, ast.Block):
        raise TypeError("Expected block node during deserialization")
    return node


__all__ = ["to_json", "from_json"]
