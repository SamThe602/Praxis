"""Parser implementation for the Praxis DSL.

The grammar is intentionally small and pragmatic: we aim to support the
constructs required by the fixtures and foreseeable curriculum tasks while also
providing strong validation (lexical scope checks, structured errors) so later
pipeline stages receive well-formed ASTs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Set, cast

from . import ast, type_system


class DSLParseError(RuntimeError):
    """Structured parse error that includes source location information."""

    def __init__(self, message: str, line: int, column: int, filename: str = "<dsl>"):
        super().__init__(f"{filename}:{line}:{column}: {message}")
        self.message = message
        self.line = line
        self.column = column
        self.filename = filename


@dataclass(slots=True)
class Token:
    """Single lexical token."""

    kind: str
    value: str
    line: int
    column: int
    end_line: int
    end_column: int


KEYWORDS = {
    "fn",
    "let",
    "mut",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "in",
    "match",
    "return",
    "true",
    "false",
    "and",
    "or",
    "not",
}

BUILTINS = {"len", "range", "sum", "min", "max", "sorted", "enumerate", "return"}

OPERATORS = {
    "==",
    "!=",
    "<=",
    ">=",
    "=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "%",
}


class Tokenizer:
    """Simple hand-written tokenizer.

    The DSL is deliberately compact; a hand-rolled lexer keeps dependencies
    minimal while still giving precise error reporting.
    """

    def __init__(self, source: str, filename: str = "<dsl>") -> None:
        self.source = source
        self.filename = filename
        self.length = len(source)
        self.index = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while not self._eof:
            ch = self._peek()
            if ch.isspace():
                self._consume_whitespace()
                continue
            if ch == "#":
                self._consume_comment()
                continue
            if ch.isalpha() or ch == "_":
                tokens.append(self._consume_identifier())
                continue
            if ch.isdigit():
                tokens.append(self._consume_number())
                continue
            if ch == '"':
                tokens.append(self._consume_string())
                continue
            tokens.append(self._consume_punctuation())
        tokens.append(Token("EOF", "", self.line, self.column, self.line, self.column))
        return tokens

    @property
    def _eof(self) -> bool:
        return self.index >= self.length

    def _peek(self, offset: int = 0) -> str:
        if self.index + offset >= self.length:
            return "\0"
        return self.source[self.index + offset]

    def _advance(self, count: int = 1) -> str:
        value = ""
        for _ in range(count):
            if self._eof:
                break
            ch = self.source[self.index]
            value += ch
            self.index += 1
            if ch == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return value

    def _consume_whitespace(self) -> None:
        while not self._eof and self._peek().isspace():
            self._advance()

    def _consume_comment(self) -> None:
        while not self._eof and self._peek() != "\n":
            self._advance()

    def _consume_identifier(self) -> Token:
        start_line, start_column = self.line, self.column
        value = self._advance()
        while self._peek().isalnum() or self._peek() == "_":
            value += self._advance()
        kind = value if value in KEYWORDS else "IDENT"
        return Token(kind, value, start_line, start_column, self.line, self.column)

    def _consume_number(self) -> Token:
        start_line, start_column = self.line, self.column
        value = self._advance()
        kind = "INT"
        while self._peek().isdigit():
            value += self._advance()
        if self._peek() == "." and self._peek(1).isdigit():
            kind = "FLOAT"
            value += self._advance()
            while self._peek().isdigit():
                value += self._advance()
        return Token(kind, value, start_line, start_column, self.line, self.column)

    def _consume_string(self) -> Token:
        start_line, start_column = self.line, self.column
        self._advance()  # opening quote
        value_chars: list[str] = []
        while not self._eof:
            ch = self._advance()
            if ch == '"':
                break
            if ch == "\\":
                escaped = self._advance()
                escape_map = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
                value_chars.append(escape_map.get(escaped, escaped))
            else:
                value_chars.append(ch)
        else:
            raise DSLParseError(
                "Unterminated string literal", start_line, start_column, self.filename
            )
        literal = "".join(value_chars)
        return Token("STRING", literal, start_line, start_column, self.line, self.column)

    def _consume_punctuation(self) -> Token:
        start_line, start_column = self.line, self.column
        ch = self._advance()
        two_char = ch + self._peek()
        if ch == "-" and self._peek() == ">":
            self._advance()
            return Token("ARROW", "->", start_line, start_column, self.line, self.column)
        if ch == "=" and self._peek() == ">":
            self._advance()
            return Token("FATARROW", "=>", start_line, start_column, self.line, self.column)
        if two_char in {"==", "!=", "<=", ">=", "&&", "||"}:
            self._advance()
            kind = {
                "&&": "AND_OP",
                "||": "OR_OP",
            }.get(two_char, two_char)
            return Token(kind, two_char, start_line, start_column, self.line, self.column)
        punct_map = {
            "(": "LPAREN",
            ")": "RPAREN",
            "{": "LBRACE",
            "}": "RBRACE",
            "[": "LBRACKET",
            "]": "RBRACKET",
            ":": "COLON",
            ",": "COMMA",
            ";": "SEMICOLON",
            "@": "AT",
            "|": "PIPE",
        }
        if ch in punct_map:
            kind = punct_map[ch]
        elif ch in OPERATORS:
            kind = ch
        else:
            raise DSLParseError(
                f"Unexpected character '{ch}'", start_line, start_column, self.filename
            )
        return Token(kind, ch, start_line, start_column, self.line, self.column)


# ---------------------------------------------------------------------------
# Parser


class Scope:
    """Lexical scope tracking declared identifiers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.bindings: set[str] = set()

    def declare(self, name: str) -> None:
        self.bindings.add(name)

    def contains(self, name: str) -> bool:
        return name in self.bindings


class Parser:
    """Recursive-descent parser paired with lexical scope validation."""

    def __init__(self, tokens: Sequence[Token], filename: str = "<dsl>") -> None:
        self.tokens = tokens
        self.index = 0
        self.filename = filename
        self.scopes: list[Scope] = [Scope("module")]
        self.functions: set[str] = {
            tokens[i + 1].value
            for i in range(len(tokens) - 1)
            if tokens[i].kind == "fn" and tokens[i + 1].kind == "IDENT"
        }
        self.current_contracts: list[ast.Contract] = []
        self._allow_unbound = 0

    # ------------------------------------------------------------------
    # Scope helpers

    def _push_scope(self, name: str) -> None:
        self.scopes.append(Scope(name))

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _declare(self, name: str, token: Token) -> None:
        if not name:
            raise DSLParseError("Empty identifier", token.line, token.column, self.filename)
        if self.scopes[-1].contains(name):
            raise DSLParseError(
                f"Identifier '{name}' already declared in this scope",
                token.line,
                token.column,
                self.filename,
            )
        self.scopes[-1].declare(name)

    def _ensure_defined(self, name: str, token: Token) -> None:
        if self._allow_unbound > 0:
            return
        if name in BUILTINS or name in self.functions:
            return
        for scope in reversed(self.scopes):
            if scope.contains(name):
                return
        raise DSLParseError(
            f"Use of undefined identifier '{name}'", token.line, token.column, self.filename
        )

    def _validate_identifiers(
        self, node: Optional[ast.Node], extra_allowed: Optional[set[str]] = None
    ) -> None:
        if node is None:
            return
        allowed: Set[str] = set(extra_allowed or set())
        for literal in self._iter_identifier_literals(node):
            name = cast(str, literal.value)
            token_obj = (
                literal.metadata.get("token") if isinstance(literal.metadata, dict) else None
            )
            token = cast(Optional[Token], token_obj)
            if name in allowed:
                continue
            if token is None:
                token = Token("IDENT", name, 0, 0, 0, 0)
            self._ensure_defined(name, token)

    def _iter_identifier_literals(self, node: ast.Node) -> Iterator[ast.Literal]:
        if isinstance(node, ast.Literal) and node.literal_type == "identifier":
            yield node
        for child in node.children():
            yield from self._iter_identifier_literals(child)

    def _permit_unbound(self) -> None:
        self._allow_unbound += 1

    def _release_unbound(self) -> None:
        if self._allow_unbound == 0:
            raise RuntimeError("Unbalanced _permit_unbound/_release_unbound calls")
        self._allow_unbound -= 1

    @contextmanager
    def _unbound_context(self) -> Iterator[None]:
        self._permit_unbound()
        try:
            yield
        finally:
            self._release_unbound()

    # ------------------------------------------------------------------
    # Token helpers

    def _peek(self, offset: int = 0) -> Token:
        target = self.index + offset
        if target < 0:
            target = 0
        if target >= len(self.tokens):
            target = len(self.tokens) - 1
        return self.tokens[target]

    def _advance(self) -> Token:
        token = self._peek()
        if token.kind != "EOF":
            self.index += 1
        return token

    def _match(self, *kinds: str) -> Optional[Token]:
        token = self._peek()
        if token.kind in kinds:
            self.index += 1
            return token
        return None

    def _expect(self, *kinds: str) -> Token:
        token = self._peek()
        if token.kind not in kinds:
            expected = " or ".join(kinds)
            raise DSLParseError(
                f"Expected {expected}, found {token.kind}", token.line, token.column, self.filename
            )
        self.index += 1
        return token

    def _check(self, *kinds: str) -> bool:
        return self._peek().kind in kinds

    # ------------------------------------------------------------------
    # Entry points

    def parse_module(self) -> ast.Module:
        functions: list[ast.FunctionDecl] = []
        seen: set[str] = set()
        while not self._check("EOF"):
            contracts = self._parse_contracts()
            fn_token = self._expect("fn")
            name_token = self._expect("IDENT")
            if name_token.value in seen:
                raise DSLParseError(
                    f"Duplicate function '{name_token.value}'",
                    name_token.line,
                    name_token.column,
                    self.filename,
                )
            seen.add(name_token.value)
            function = self._parse_function_body(fn_token, name_token, contracts)
            functions.append(function)
        return ast.Module(functions=functions)

    def _parse_contracts(self) -> list[ast.Contract]:
        contracts: list[ast.Contract] = []
        while self._match("AT"):
            name_token = self._expect("IDENT")
            self._expect("LPAREN")
            args: list[ast.Expression] = []
            if not self._check("RPAREN"):
                args.append(self._parse_expression())
                while self._match("COMMA"):
                    args.append(self._parse_expression())
            self._expect("RPAREN")
            contracts.append(ast.Contract(name=name_token.value, arguments=args))
        return contracts

    def _parse_function_body(
        self, fn_token: Token, name_token: Token, contracts: list[ast.Contract]
    ) -> ast.FunctionDecl:
        self._expect("LPAREN")
        parameters: list[ast.Parameter] = []
        if not self._check("RPAREN"):
            parameters.append(self._parse_parameter())
            while self._match("COMMA"):
                parameters.append(self._parse_parameter())
        self._expect("RPAREN")
        seen_parameters: list[str] = []
        for param in parameters:
            if param.default is not None:
                self._validate_identifiers(param.default, extra_allowed=set(seen_parameters))
            seen_parameters.append(param.name)
        return_type: Optional[str] = None
        if self._match("ARROW"):
            return_type = self._parse_type_annotation()
        self._push_scope(f"fn:{name_token.value}")
        for param in parameters:
            dummy = Token(
                "IDENT", param.name, fn_token.line, fn_token.column, fn_token.line, fn_token.column
            )
            self._declare(param.name, dummy)
        body = self._parse_block()
        self._pop_scope()
        return ast.FunctionDecl(
            name=name_token.value,
            parameters=parameters,
            return_type=return_type,
            body=body,
            contracts=contracts,
        )

    def _parse_parameter(self) -> ast.Parameter:
        name_token = self._expect("IDENT")
        type_annotation: Optional[str] = None
        if self._match("COLON"):
            type_annotation = self._parse_type_annotation()
        default = None
        if self._match("="):
            with self._unbound_context():
                default = self._parse_expression()
        return ast.Parameter(
            name=name_token.value, type_annotation=type_annotation, default=default
        )

    def _parse_type_annotation(self) -> str:
        parts = [self._expect("IDENT").value]
        while self._match("<"):
            parts.append("<")
            parts.append(self._parse_type_annotation())
            while self._match("COMMA"):
                parts.append(",")
                parts.append(self._parse_type_annotation())
            self._expect(">")
            parts.append(">")
        if self._match("LBRACKET"):
            parts.append("[")
            parts.append(self._parse_type_annotation())
            self._expect("RBRACKET")
            parts.append("]")
        return "".join(parts)

    def _parse_block(self) -> ast.Block:
        self._expect("LBRACE")
        statements: list[ast.Statement] = []
        self._push_scope("block")
        while not self._check("RBRACE"):
            statements.append(self._parse_statement())
        self._pop_scope()
        self._expect("RBRACE")
        return ast.Block(statements=statements)

    def _parse_statement(self) -> ast.Statement:
        if self._match("let"):
            mutable = bool(self._match("mut"))
            name_token = self._expect("IDENT")
            type_annotation: Optional[str] = None
            if self._match("COLON"):
                type_annotation = self._parse_type_annotation()
            self._expect("=")
            value = self._parse_expression()
            self._expect("SEMICOLON")
            self._declare(name_token.value, name_token)
            return ast.Let(
                name=name_token.value,
                value=value,
                type_annotation=type_annotation,
                mutable=mutable,
            )
        if self._match("return"):
            if not self._check("SEMICOLON"):
                argument = self._parse_expression()
                arguments = [argument]
            else:
                arguments = []
            self._expect("SEMICOLON")
            return ast.BuiltinCall(name="return", arguments=arguments)
        if self._match("if"):
            return self._parse_if()
        if self._match("for"):
            return self._parse_for()
        if self._match("while"):
            return self._parse_while()
        if self._match("match"):
            return self._parse_match()

        # Expression or assignment statement
        expr = self._parse_expression()
        if isinstance(expr, ast.Literal) and expr.literal_type == "identifier" and self._match("="):
            target_name = cast(str, expr.value)
            assign_token = Token("IDENT", target_name, 0, 0, 0, 0)
            self._ensure_defined(target_name, assign_token)
            value = self._parse_expression()
            self._expect("SEMICOLON")
            return ast.Assign(target=target_name, value=value)
        self._expect("SEMICOLON")
        return expr

    def _parse_if(self) -> ast.Conditional:
        test = self._parse_expression()
        then_block = self._parse_block()
        branches: list[tuple[Optional[ast.Expression], ast.Block]] = [(test, then_block)]
        while self._match("elif"):
            condition = self._parse_expression()
            block = self._parse_block()
            branches.append((condition, block))
        if self._match("else"):
            else_block = self._parse_block()
            branches.append((None, else_block))
        return ast.Conditional(kind="if", test=test, branches=branches)

    def _parse_for(self) -> ast.Loop:
        target_token = self._expect("IDENT")
        self._expect("in")
        iterable = self._parse_expression()
        self._push_scope("for")
        self._declare(target_token.value, target_token)
        body = self._parse_block()
        self._pop_scope()
        return ast.Loop(
            kind="for", target=target_token.value, iterable=iterable, condition=None, body=body
        )

    def _parse_while(self) -> ast.Loop:
        condition = self._parse_expression()
        body = self._parse_block()
        return ast.Loop(kind="while", target=None, iterable=None, condition=condition, body=body)

    def _parse_match(self) -> ast.Conditional:
        target = self._parse_expression()
        self._expect("LBRACE")
        arms: list[ast.MatchArm] = []
        while not self._check("RBRACE"):
            pattern_token = self._expect("IDENT", "STRING", "INT", "FLOAT", "true", "false")
            pattern_text = pattern_token.value
            if pattern_token.kind == "true":
                pattern_text = "true"
            elif pattern_token.kind == "false":
                pattern_text = "false"
            self._push_scope("match-arm")
            if pattern_token.kind == "IDENT" and pattern_token.value != "_":
                self._declare(pattern_token.value, pattern_token)
            guard: Optional[ast.Expression] = None
            if self._match("if"):
                guard = self._parse_expression()
                self._validate_identifiers(
                    guard,
                    extra_allowed={pattern_token.value} if pattern_token.kind == "IDENT" else None,
                )
            self._expect("FATARROW")
            body = self._parse_block()
            self._pop_scope()
            arms.append(ast.MatchArm(pattern=ast.Pattern(pattern_text), body=body, guard=guard))
            self._match("COMMA")
        self._expect("RBRACE")
        return ast.Conditional(kind="match", test=target, arms=arms)

    # ------------------------------------------------------------------
    # Expressions

    def _parse_expression(self) -> ast.Expression:
        return self._parse_binary_expression(0)

    def _parse_binary_expression(self, min_precedence: int) -> ast.Expression:
        expr = self._parse_unary()
        while True:
            operator_token = self._peek()
            operator = self._binary_operator(operator_token)
            if operator is None:
                break
            precedence = _PRECEDENCE[operator]
            if precedence < min_precedence:
                break
            self._advance()
            rhs = self._parse_binary_expression(precedence + 1)
            expr = ast.BinaryOp(operator=operator, left=expr, right=rhs)
        return expr

    def _binary_operator(self, token: Token) -> Optional[str]:
        kind = token.kind
        if kind in {"AND_OP"}:
            return "and"
        if kind in {"OR_OP"}:
            return "or"
        if kind in _PRECEDENCE:
            return kind
        if kind in {"==", "!=", "<=", ">=", "<", ">"}:
            return kind
        return None

    def _parse_unary(self) -> ast.Expression:
        token = self._peek()
        if token.kind in {"-", "not"}:
            self._advance()
            operand = self._parse_unary()
            return ast.UnaryOp(operator=token.kind, operand=operand)
        return self._parse_call()

    def _parse_call(self) -> ast.Expression:
        expr = self._parse_primary()
        while self._match("LPAREN"):
            arguments: list[ast.Expression] = []
            if not self._check("RPAREN"):
                arguments.append(self._parse_expression())
                while self._match("COMMA"):
                    arguments.append(self._parse_expression())
            self._expect("RPAREN")
            if isinstance(expr, ast.Literal) and expr.literal_type == "identifier":
                name = cast(str, expr.value)
                call_node: ast.Node
                if name in BUILTINS:
                    call_node = ast.BuiltinCall(name=name, arguments=arguments)
                else:
                    metadata_token_obj = None
                    if isinstance(expr.metadata, dict):
                        metadata_token_obj = expr.metadata.get("token")
                    metadata_token = cast(Optional[Token], metadata_token_obj)
                    if metadata_token is None:
                        metadata_token = Token("IDENT", name, 0, 0, 0, 0)
                    self._ensure_defined(name, metadata_token)
                    call_node = ast.Call(function=name, arguments=arguments)
                expr = call_node
            else:
                raise DSLParseError(
                    "Calls must target identifiers",
                    self._peek(-1).line,
                    self._peek(-1).column,
                    self.filename,
                )
        return expr

    def _parse_primary(self) -> ast.Expression:
        token = self._peek()
        if token.kind == "LPAREN":
            self._advance()
            expr = self._parse_expression()
            if self._match("COMMA"):
                items = [expr]
                items.append(self._parse_expression())
                while self._match("COMMA"):
                    items.append(self._parse_expression())
                self._expect("RPAREN")
                literal = ast.Literal(literal_type="tuple", value=items)
                literal.metadata["token"] = token
                return literal
            self._expect("RPAREN")
            return expr
        if token.kind == "LBRACKET":
            return self._parse_list_or_comprehension()
        if token.kind == "LBRACE":
            return self._parse_map_literal()
        if token.kind == "PIPE":
            return self._parse_lambda()
        if token.kind in {"IDENT"}:
            self._advance()
            self._ensure_defined(token.value, token)
            literal = ast.Literal(literal_type="identifier", value=token.value)
            literal.metadata["token"] = token
            return literal
        if token.kind == "true":
            self._advance()
            literal = ast.Literal(literal_type="bool", value=True)
            literal.metadata["token"] = token
            return literal
        if token.kind == "false":
            self._advance()
            literal = ast.Literal(literal_type="bool", value=False)
            literal.metadata["token"] = token
            return literal
        if token.kind == "INT":
            self._advance()
            literal = ast.Literal(literal_type="int", value=int(token.value))
            literal.metadata["token"] = token
            return literal
        if token.kind == "FLOAT":
            self._advance()
            literal = ast.Literal(literal_type="float", value=float(token.value))
            literal.metadata["token"] = token
            return literal
        if token.kind == "STRING":
            self._advance()
            literal = ast.Literal(literal_type="string", value=token.value)
            literal.metadata["token"] = token
            return literal
        raise DSLParseError(
            f"Unexpected token {token.kind}", token.line, token.column, self.filename
        )

    def _parse_list_or_comprehension(self) -> ast.Expression:
        open_token = self._expect("LBRACKET")
        if self._check("RBRACKET"):
            closing = self._advance()
            literal = ast.Literal(literal_type="list", value=[])
            literal.metadata["token"] = open_token
            literal.metadata["end_token"] = closing
            return literal
        # Permit unresolved identifiers while we probe for a comprehension; the
        # identifiers will be validated once we know the binding context.
        with self._unbound_context():
            first_expr = self._parse_expression()
        if self._match("for"):
            target_token = self._expect("IDENT")
            self._expect("in")
            iterable = self._parse_expression()
            condition = None
            if self._match("if"):
                with self._unbound_context():
                    condition = self._parse_expression()
            self._expect("RBRACKET")
            self._push_scope("comprehension")
            self._declare(target_token.value, target_token)
            self._validate_identifiers(first_expr, extra_allowed={target_token.value})
            if condition is not None:
                self._validate_identifiers(condition, extra_allowed={target_token.value})
            self._pop_scope()
            return ast.Comprehension(
                kind="list",
                target=target_token.value,
                iterable=iterable,
                expression=first_expr,
                condition=condition,
            )
        self._validate_identifiers(first_expr)
        elements = [first_expr]
        while self._match("COMMA"):
            expr = self._parse_expression()
            self._validate_identifiers(expr)
            elements.append(expr)
        closing = self._expect("RBRACKET")
        literal = ast.Literal(literal_type="list", value=elements)
        literal.metadata["token"] = open_token
        literal.metadata["end_token"] = closing
        return literal

    def _parse_map_literal(self) -> ast.Expression:
        open_token = self._expect("LBRACE")
        entries: list[tuple[ast.Expression, ast.Expression]] = []
        if not self._check("RBRACE"):
            key = self._parse_expression()
            self._expect("COLON")
            value = self._parse_expression()
            self._validate_identifiers(key)
            self._validate_identifiers(value)
            entries.append((key, value))
            while self._match("COMMA"):
                key = self._parse_expression()
                self._expect("COLON")
                value = self._parse_expression()
                self._validate_identifiers(key)
                self._validate_identifiers(value)
                entries.append((key, value))
        closing = self._expect("RBRACE")
        literal = ast.Literal(literal_type="map", value=entries)
        literal.metadata["token"] = open_token
        literal.metadata["end_token"] = closing
        return literal

    def _parse_lambda(self) -> ast.Lambda:
        self._expect("PIPE")
        parameters: list[ast.Parameter] = []
        if not self._check("PIPE"):
            parameters.append(self._parse_parameter())
            while self._match("COMMA"):
                parameters.append(self._parse_parameter())
        self._expect("PIPE")
        seen_params: list[str] = []
        for param in parameters:
            if param.default is not None:
                self._validate_identifiers(param.default, extra_allowed=set(seen_params))
            seen_params.append(param.name)
        self._push_scope("lambda")
        for param in parameters:
            dummy = Token("IDENT", param.name, 0, 0, 0, 0)
            self._declare(param.name, dummy)
        body = self._parse_expression()
        self._pop_scope()
        return ast.Lambda(parameters=parameters, body=body)


_PRECEDENCE: dict[str, int] = {
    "or": 1,
    "and": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "%": 6,
}


def parse_module(source: str, *, filename: str = "<dsl>") -> ast.Module:
    """Parse DSL ``source`` text into an :class:`ast.Module` instance."""

    tokens = Tokenizer(source, filename=filename).tokenize()
    parser = Parser(tokens, filename=filename)
    module = parser.parse_module()
    type_system.infer(module)
    return module


def parse_statements(source: str, *, filename: str = "<dsl>") -> Iterable[ast.Statement]:
    """Parse a brace-wrapped block and return its statements.

    Useful for unit tests where we only care about statement parsing.
    """

    module = parse_module(f"fn __temp__() {{{source}}}", filename=filename)
    if not module.functions:
        return []
    return module.functions[0].body.statements


__all__ = ["parse_module", "parse_statements", "DSLParseError", "Tokenizer", "Token"]
