"""AST serialization scaffold."""

from __future__ import annotations

from typing import Any


def to_json(node: Any) -> str:
    raise NotImplementedError("Serializer scaffold.")


def from_json(payload: str) -> Any:
    raise NotImplementedError("Deserializer scaffold.")
