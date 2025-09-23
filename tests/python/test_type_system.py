"""Tests for the Praxis DSL Hindley–Milner type system."""

from __future__ import annotations

from typing import Any, cast

import pytest

from packages.dsl import grammar, type_system


def _binding_type(stmt: Any) -> str:
    binding = cast(type_system.Type, stmt.metadata["binding_type"])
    return type_system.format_type(binding)


def test_infer_function_signature() -> None:
    module = grammar.parse_module("fn id(x: int) -> int { return x; }")
    fn = module.functions[0]

    fn_type = cast(type_system.Type, fn.metadata["type"])
    return_type = cast(type_system.Type, fn.metadata["return_type_inferred"])
    param_types = cast(dict[str, type_system.Type], fn.metadata["parameter_types"])

    assert type_system.format_type(fn_type) == "(int) -> int"
    assert type_system.format_type(return_type) == "int"
    assert type_system.format_type(param_types["x"]) == "int"


def test_polymorphic_let_generalisation() -> None:
    source = """
    fn poly() {
        let id = |x| x;
        let a = id(1);
        let b = id(true);
    }
    """

    module = grammar.parse_module(source)
    block = module.functions[0].body
    id_stmt, a_stmt, b_stmt = block.statements

    assert _binding_type(id_stmt) == "('a) -> 'a"
    assert _binding_type(a_stmt) == "int"
    assert _binding_type(b_stmt) == "bool"


def test_contract_metadata_harvest() -> None:
    source = """
    @requires("cpu", "network")
    @ensures("cpu")
    fn task(x: int) -> int {
        return x;
    }
    """

    module = grammar.parse_module(source)
    fn = module.functions[0]
    contracts = cast(dict[str, list[list[str]]], fn.metadata["contracts"])

    assert contracts["requires"] == [["cpu", "network"]]
    assert contracts["ensures"] == [["cpu"]]
    assert contracts["pre"] == []
    assert contracts["post"] == []


def test_annotation_mismatch_raises() -> None:
    source = """
    fn boom(x: int) -> int {
        let y: bool = x;
        return x;
    }
    """

    with pytest.raises(type_system.TypeSystemError):
        grammar.parse_module(source)
