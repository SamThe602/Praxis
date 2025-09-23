"""Serialization helpers scaffold."""

from __future__ import annotations

import json
from typing import Any


def dumps(payload: Any) -> str:
    return json.dumps(payload)


def loads(data: str) -> Any:
    return json.loads(data)
