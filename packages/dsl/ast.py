"""Core abstract syntax tree definitions for the Praxis DSL.

The goal of this module is to provide a small, explicit set of node types that the
grammar can target and the rest of the system (serializer, generators, transpiler)
can reason about.  The set of nodes mirrors the structures outlined in
``Spec.md`` and favours simple data-only dataclasses so they remain easy to
serialize, compare, and transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Iterable, Iterator, Optional, Sequence

# ---------------------------------------------------------------------------
# Shared utilities


@dataclass(slots=True)
class Span:
    """Represents the start/end position of a token or node in the source file."""

    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return a tuple form used by serializers."""

        return (self.start_line, self.start_column, self.end_line, self.end_column)


@dataclass(slots=True, kw_only=True)
class Node:
    """Base class for all AST nodes.

    The ``span`` attribute is optional because we occasionally synthesise nodes
    that do not come directly from user-authored source (e.g. in tests or
    generators).  The ``metadata`` dictionary is available for passes that want
    to attach additional information without mutating the structural fields.
    """

    span: Optional[Span] = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def node_type(self) -> str:
        """Expose a stable node type string used by the serializer."""

        return self.__class__.__name__

    def children(self) -> Iterator[Node]:
        """Yield child nodes in declaration order.

        ``dataclasses.fields`` is used so the traversal automatically stays in
        sync when new attributes are added.
        """

        for spec in fields(self):
            if spec.name in {"span", "metadata"}:
                continue
            value = getattr(self, spec.name)
            yield from _iter_possible_children(value)

    def walk(self) -> Iterator[Node]:
        """Depth-first traversal starting at this node."""

        yield self
        for child in self.children():
            yield from child.walk()


# ---------------------------------------------------------------------------
# Supporting lightweight dataclasses


@dataclass(slots=True)
class Parameter:
    """Function or lambda parameter definition."""

    name: str
    type_annotation: Optional[str] = None
    default: Optional["Expression"] = None


@dataclass(slots=True)
class Pattern:
    """Simple pattern used inside match arms."""

    text: str


# ---------------------------------------------------------------------------
# AST node hierarchy (statements + expressions)


@dataclass(slots=True)
class Module(Node):
    """Top-level compilation unit containing a set of functions."""

    functions: list["FunctionDecl"]


@dataclass(slots=True)
class Contract(Node):
    """Decorator-style contract attached to functions or blocks."""

    name: str
    arguments: list["Expression"]


@dataclass(slots=True)
class Block(Node):
    """A lexical block consisting of an ordered list of statements."""

    statements: list["Statement"]


Expression = Node
Statement = Node


@dataclass(slots=True)
class FunctionDecl(Node):
    """Function declaration consisting of signature, contracts, and body."""

    name: str
    parameters: list[Parameter]
    return_type: Optional[str]
    body: Block
    contracts: list[Contract] = field(default_factory=list)


@dataclass(slots=True)
class Let(Node):
    """Variable declaration with optional type annotation."""

    name: str
    value: Expression
    type_annotation: Optional[str] = None
    mutable: bool = False


@dataclass(slots=True)
class Assign(Node):
    """Assignment to an existing binding."""

    target: str
    value: Expression


@dataclass(slots=True)
class Loop(Node):
    """Represents either a ``for`` or ``while`` loop."""

    kind: str  # "for" | "while"
    target: Optional[str]
    iterable: Optional[Expression]
    condition: Optional[Expression]
    body: Block


@dataclass(slots=True)
class Conditional(Node):
    """If/elif/else chains and match expressions."""

    kind: str  # "if" | "match"
    test: Expression
    branches: list[tuple[Optional[Expression], Block]] = field(default_factory=list)
    arms: list[MatchArm] = field(default_factory=list)


@dataclass(slots=True)
class MatchArm(Node):
    """Single arm of a match expression or statement."""

    pattern: Pattern
    body: Block
    guard: Optional[Expression] = None


@dataclass(slots=True)
class Call(Node):
    """Invocation of user-defined function or callable expression."""

    function: str
    arguments: list[Expression]


@dataclass(slots=True)
class BuiltinCall(Node):
    """Invocation of a recognised builtin (e.g. ``len`` or ``return``)."""

    name: str
    arguments: list[Expression]


@dataclass(slots=True)
class Lambda(Node):
    """Lambda expression capturing an inline block or expression."""

    parameters: list[Parameter]
    body: Expression


@dataclass(slots=True)
class Literal(Node):
    """Primitive literal value.

    ``literal_type`` differentiates between identifiers, numbers, strings, and
    container literals.  Identifiers are represented as literals rather than a
    dedicated ``Identifier`` node to keep the public node set aligned with the
    spec; the ``value`` field stores the textual identifier name in that case.
    """

    literal_type: str
    value: object


@dataclass(slots=True)
class BinaryOp(Node):
    """Binary operation (``lhs <op> rhs``)."""

    operator: str
    left: Expression
    right: Expression


@dataclass(slots=True)
class UnaryOp(Node):
    """Unary operation (``op rhs``)."""

    operator: str
    operand: Expression


@dataclass(slots=True)
class Comprehension(Node):
    """List/map comprehension expression."""

    kind: str  # "list" | "set" | "map"
    target: str
    iterable: Expression
    expression: Expression
    condition: Optional[Expression] = None


# ---------------------------------------------------------------------------
# Helper functions


def is_node(value: object) -> bool:
    """Return True when ``value`` is an AST node instance."""

    return is_dataclass(value) and isinstance(value, Node)


def iter_nodes(root: Node) -> Iterable[Node]:
    """Convenience wrapper to iterate depth-first over a subtree."""

    return root.walk()


def _iter_possible_children(value: object) -> Iterator[Node]:
    if isinstance(value, Node):
        yield value
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, Node):
                yield item
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                yield from _iter_possible_children(item)
