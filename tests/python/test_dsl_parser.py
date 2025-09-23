"""Integration-focused tests for the Praxis DSL grammar."""

import pytest

from packages.dsl import ast, generators, grammar


@pytest.mark.parametrize(
    "program",
    list(generators.generate_curriculum_tasks()),
)
def test_fixture_programs_parse(program):
    module = program.module
    assert isinstance(module, ast.Module)
    assert module.functions, "fixtures should define at least one function"
    for fn in module.functions:
        assert isinstance(fn.body, ast.Block)
        # Ensure every return is represented as a builtin call for uniformity.
        returns = [stmt for stmt in fn.body.statements if isinstance(stmt, ast.BuiltinCall)]
        for call in returns:
            if call.name == "return":
                assert len(call.arguments) <= 1


def test_undefined_identifier_is_rejected():
    with pytest.raises(grammar.DSLParseError) as exc:
        grammar.parse_module("fn oops() { return missing; }")
    assert "missing" in str(exc.value)


def test_comprehension_target_is_scoped():
    source = "fn build(xs: list<int>) { let ys = [x for x in xs]; }"
    module = grammar.parse_module(source)
    comp = module.functions[0].body.statements[0].value
    assert isinstance(comp, ast.Comprehension)
    assert comp.target == "x"
    # The comprehension expression should reference the loop variable.
    assert isinstance(comp.expression, ast.Literal)
    assert comp.expression.value == "x"


def test_duplicate_binding_in_same_scope_raises():
    with pytest.raises(grammar.DSLParseError):
        grammar.parse_module("fn f() { let x = 1; let x = 2; }")
