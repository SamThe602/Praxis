# Praxis DSL

The Praxis DSL is a small, expression-oriented language used to represent
verification-friendly programs.  It borrows familiar syntax from Rust and
Python, prioritising readability over exotic features so that curriculum tasks
and automated tooling remain approachable.

## Core Syntax

- **Functions** – declared with `fn name(param: type) -> return_type { ... }`.  A
  function may carry decorators such as `@pre` or `@ensures` which are captured
  as contract nodes in the AST.
- **Bindings** – `let` introduces new immutable bindings, `let mut` allows
  reassignment, and plain `target = expr;` updates existing bindings.
- **Control flow** – familiar `if`/`elif`/`else` blocks, `for` loops over
  iterables, `while` loops, and `match` expressions that dispatch on pattern
  arms with optional guards.
- **Expressions** – arithmetic and comparison operators, boolean connectives,
  function calls, inline lambdas (`|x, y| expr`), list/map literals, and list
  comprehensions (`[expr for item in items if predicate]`).
- **Return values** – `return expr;` is encoded as a builtin call so the grammar
  stays uniform across statements and expressions.

## AST Overview

Parsing produces the node types defined in `packages/dsl/ast.py`:

- `Module` contains a list of `FunctionDecl` nodes.
- `FunctionDecl` stores parameters (`Parameter` helper records), optional
  `Contract` decorators, an optional return type string, and a `Block` body.
- `Block` contains ordered statements (other node types such as `Let`, `Assign`,
  `Loop`, `Conditional`, `Call`, `BuiltinCall`).
- Expression nodes include `Literal`, `BinaryOp`, `UnaryOp`, `Call`,
  `BuiltinCall`, `Lambda`, and `Comprehension`.  Literals cover identifiers,
  primitives, tuples, lists, and maps.
- `Conditional` captures both `if` chains (via `branches`) and `match`
  statements (via `arms` made of `MatchArm` records).

Lexical scope is enforced during parsing.  Identifiers must be declared before
use, shadowing inside nested blocks is allowed, and comprehension targets are
validated so stray variables are rejected early.

## Canonical Serialization

`packages/dsl/serializer.py` produces canonical JSON for any AST node:

- Fields are emitted in dataclass order using an `OrderedDict`, ensuring stable
  formatting for golden tests and caching.
- Every node (including parameters and match patterns) receives a
  content-addressed `id` computed from its JSON representation.  The hash is the
  first 16 hex characters of a SHA-256 digest.
- Deserialization recomputes and verifies each hash.  A mismatch raises a
  `ValueError`, preventing stale or tampered payloads from entering the system.

Example round-trip:

```python
from packages.dsl import grammar, serializer

module = grammar.parse_module('fn id(x: int) -> int { return x; }')
payload = serializer.to_json(module)
restored = serializer.from_json(payload)
assert serializer.to_json(restored) == payload
```

## Debugging Helpers

- `packages/dsl/transpiler.py` can render an AST back into readable Python code
  for quick inspection while developing fixtures.
- `packages/dsl/generators.py` loads the canonical `.dsl` examples shipped under
  `tests/python/fixtures/programs`, parses them using the grammar, and yields
  structured `GeneratedProgram` records for reuse in tests.

The DSL type system is implemented in subsequent prompts; for now the parser
performs structural checks, leaving type inference to dedicated passes.
