"""Helpers for parsing and representing Praxis DSL contracts.

The type checker attaches raw :class:`packages.dsl.ast.Contract` nodes to
function definitions.  Later passes – static analysis today, runtime checks in
the future – need those annotations in a normalised, well-typed form.  This
module provides lightweight dataclasses for that representation along with
convenience utilities to harvest contracts from AST declarations.

Parsing happens eagerly during verification so we surface informative errors as
soon as a contract is malformed (e.g. non-string ``@requires`` arguments).  The
resulting dataclasses are intentionally simple and serialisable so they can be
cached alongside other synthesis artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from packages.dsl import ast

__all__ = [
    "ContractError",
    "ContractCondition",
    "FunctionContracts",
    "ResourceClause",
    "collect_function_contracts",
]


class ContractError(ValueError):
    """Raised when a contract annotation cannot be interpreted."""


@dataclass(slots=True, frozen=True)
class ContractCondition:
    """Boolean contract clause, typically produced by ``@pre`` or ``@post``."""

    kind: str
    expression: object
    description: str
    span: Optional[ast.Span]


@dataclass(slots=True, frozen=True)
class ResourceClause:
    """Resource-style contract (``@requires``/``@ensures``)."""

    kind: str
    resources: tuple[str, ...]
    span: Optional[ast.Span]

    def __bool__(self) -> bool:  # pragma: no cover - ergonomic helper
        return bool(self.resources)


@dataclass(slots=True, frozen=True)
class FunctionContracts:
    """Normalised set of contract annotations for a function."""

    function: str
    preconditions: tuple[ContractCondition, ...]
    postconditions: tuple[ContractCondition, ...]
    requires: tuple[ResourceClause, ...]
    ensures: tuple[ResourceClause, ...]

    def any_conditions(self) -> bool:
        """Return ``True`` when at least one clause is present."""

        return bool(self.preconditions or self.postconditions or self.requires or self.ensures)


ALLOWED_KINDS: frozenset[str] = frozenset({"pre", "post", "requires", "ensures"})


def collect_function_contracts(function: ast.FunctionDecl) -> FunctionContracts:
    """Harvest contract annotations from ``function``.

    Parameters
    ----------
    function:
        A fully parsed Praxis DSL function declaration.  The node must still
        contain its original :class:`ast.Contract` entries.

    Returns
    -------
    FunctionContracts
        Structured representation consumable by later passes.

    Raises
    ------
    ContractError
        If an annotation has an unknown decorator name or malformed arguments.
    """

    preconditions: list[ContractCondition] = []
    postconditions: list[ContractCondition] = []
    requires: list[ResourceClause] = []
    ensures: list[ResourceClause] = []

    for contract in function.contracts:
        if contract.name not in ALLOWED_KINDS:
            raise ContractError(f"Unknown contract @{contract.name}")

        if contract.name in {"pre", "post"}:
            for argument in _normalise_condition_arguments(
                contract.name, contract.arguments, contract.span
            ):
                condition = ContractCondition(
                    kind=contract.name,
                    expression=argument,
                    description=_format_expression(argument),
                    span=_extract_span(argument) or contract.span,
                )
                if contract.name == "pre":
                    preconditions.append(condition)
                else:
                    postconditions.append(condition)
        else:  # requires / ensures
            clause = ResourceClause(
                kind=contract.name,
                resources=_collect_resources(contract.arguments, contract.span),
                span=contract.span,
            )
            if contract.name == "requires":
                requires.append(clause)
            else:
                ensures.append(clause)

    return FunctionContracts(
        function=function.name,
        preconditions=tuple(preconditions),
        postconditions=tuple(postconditions),
        requires=tuple(requires),
        ensures=tuple(ensures),
    )


def _normalise_condition_arguments(
    kind: str,
    arguments: Iterable[ast.Expression],
    span: Optional[ast.Span],
) -> Iterable[ast.Expression]:
    if not arguments:
        raise ContractError(f"@{kind} requires at least one boolean expression")
    for argument in arguments:
        if not _looks_boolean(argument):
            raise ContractError(
                f"@{kind} expects boolean expressions, got {_format_expression(argument)!r}"
            )
        yield argument


def _collect_resources(
    arguments: Iterable[ast.Expression],
    span: Optional[ast.Span],
) -> tuple[str, ...]:
    resources: list[str] = []
    for argument in arguments:
        literal = _extract_literal(argument)
        if literal is None or not isinstance(literal, str):
            raise ContractError("resource contracts expect string literals")
        resources.append(literal)
    if not resources:
        raise ContractError("resource contracts must list at least one resource")
    return tuple(resources)


def _extract_literal(node: ast.Expression) -> object | None:
    if isinstance(node, ast.Literal):
        return node.value
    return None


def _looks_boolean(node: ast.Expression) -> bool:
    if isinstance(node, ast.Literal):
        return node.literal_type == "bool"
    # Non-literal expressions are assumed to be valid boolean expressions – the
    # type checker guarantees the annotation's typing, so we only guard against
    # obviously wrong literals such as strings.
    return True


def _format_expression(node: object) -> str:
    if isinstance(node, ast.Literal):
        if node.literal_type == "string":
            return repr(node.value)
        return str(node.value)
    if isinstance(node, ast.BinaryOp):
        return f"{_format_expression(node.left)} {node.operator} {_format_expression(node.right)}"
    if isinstance(node, ast.UnaryOp):
        return f"{node.operator} {_format_expression(node.operand)}"
    if isinstance(node, ast.BuiltinCall):
        args = ", ".join(_format_expression(arg) for arg in node.arguments)
        return f"{node.name}({args})"
    if isinstance(node, ast.Call):
        args = ", ".join(_format_expression(arg) for arg in node.arguments)
        return f"{node.function}({args})"
    return type(node).__name__


def _extract_span(node: object) -> Optional[ast.Span]:
    if isinstance(node, ast.Node):
        return node.span
    return None
