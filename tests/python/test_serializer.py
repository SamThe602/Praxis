"""Tests for canonical JSON serialisation of DSL ASTs."""

import json

import pytest

from packages.dsl import generators, grammar, serializer


@pytest.mark.parametrize(
    "program",
    list(generators.generate_curriculum_tasks()),
)
def test_round_trip_is_canonical(program):
    payload = serializer.to_json(program.module)
    restored = serializer.from_json(payload)
    assert serializer.to_json(restored) == payload


def test_corrupted_hash_is_detected():
    module = grammar.parse_module("fn fail() { return; }")
    payload = serializer.to_json(module)
    data = json.loads(payload)
    data["id"] = "0000000000000000"
    tampered = json.dumps(data)
    with pytest.raises(ValueError):
        serializer.from_json(tampered)
