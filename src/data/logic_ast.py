from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Expr:
    op: str
    value: Optional[object] = None
    children: Tuple["Expr", ...] = ()


def Var(name: str) -> Expr:
    return Expr(op="var", value=name)


def Const(value: bool) -> Expr:
    return Expr(op="const", value=bool(value))


def Not(x: Expr) -> Expr:
    return Expr(op="not", children=(x,))


def And(x: Expr, y: Expr) -> Expr:
    return Expr(op="and", children=(x, y))


def Or(x: Expr, y: Expr) -> Expr:
    return Expr(op="or", children=(x, y))


def Xor(x: Expr, y: Expr) -> Expr:
    return Expr(op="xor", children=(x, y))


def Implies(x: Expr, y: Expr) -> Expr:
    return Expr(op="implies", children=(x, y))


def Iff(x: Expr, y: Expr) -> Expr:
    return Expr(op="iff", children=(x, y))


def eval_expr(expr: Expr, env: Dict[str, bool]) -> bool:
    if expr.op == "const":
        return bool(expr.value)
    if expr.op == "var":
        if expr.value not in env:
            raise KeyError(f"Variable {expr.value!r} missing from environment")
        return bool(env[str(expr.value)])
    if expr.op == "not":
        return not eval_expr(expr.children[0], env)
    left = eval_expr(expr.children[0], env)
    right = eval_expr(expr.children[1], env)
    if expr.op == "and":
        return left and right
    if expr.op == "or":
        return left or right
    if expr.op == "xor":
        return left ^ right
    if expr.op == "implies":
        return (not left) or right
    if expr.op == "iff":
        return left == right
    raise ValueError(f"Unknown operator: {expr.op}")


def collect_vars(expr: Expr) -> List[str]:
    out: List[str] = []

    def walk(node: Expr) -> None:
        if node.op == "var":
            out.append(str(node.value))
            return
        for child in node.children:
            walk(child)

    walk(expr)
    return sorted(set(out))


def rename_vars(expr: Expr, mapping: Dict[str, str]) -> Expr:
    if expr.op == "var":
        old_name = str(expr.value)
        new_name = mapping.get(old_name, old_name)
        return Var(new_name)
    if not expr.children:
        return expr
    return Expr(
        op=expr.op,
        value=expr.value,
        children=tuple(rename_vars(child, mapping) for child in expr.children),
    )


_PRECEDENCE = {
    "iff": 1,
    "implies": 2,
    "or": 3,
    "xor": 3,
    "and": 4,
    "not": 5,
    "var": 6,
    "const": 6,
}


def _maybe_wrap(child: Expr, parent_op: str) -> str:
    if _PRECEDENCE[child.op] < _PRECEDENCE[parent_op]:
        return f"({to_symbolic(child)})"
    return to_symbolic(child)


def to_symbolic(expr: Expr) -> str:
    if expr.op == "const":
        return "True" if expr.value else "False"
    if expr.op == "var":
        return str(expr.value)
    if expr.op == "not":
        inner = expr.children[0]
        rendered = to_symbolic(inner)
        if inner.op in {"var", "const", "not"}:
            return f"not {rendered}"
        return f"not ({rendered})"

    left, right = expr.children
    if expr.op == "and":
        return f"{_maybe_wrap(left, 'and')} and {_maybe_wrap(right, 'and')}"
    if expr.op == "or":
        return f"{_maybe_wrap(left, 'or')} or {_maybe_wrap(right, 'or')}"
    if expr.op == "xor":
        return f"{_maybe_wrap(left, 'xor')} xor {_maybe_wrap(right, 'xor')}"
    if expr.op == "implies":
        return f"{_maybe_wrap(left, 'implies')} -> {_maybe_wrap(right, 'implies')}"
    if expr.op == "iff":
        return f"{_maybe_wrap(left, 'iff')} <-> {_maybe_wrap(right, 'iff')}"
    raise ValueError(f"Unknown operator: {expr.op}")


def to_semi_natural(expr: Expr) -> str:
    text = to_symbolic(expr)
    replacements = [
        ("<->", "EquivalentTo"),
        ("->", "Implies"),
        ("not", "Not"),
        ("and", "And"),
        ("or", "Or"),
        ("xor", "Xor"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


def tokenize(expr_text: str) -> List[Token]:
    text = expr_text.strip()
    tokens: List[Token] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if text.startswith("<->", i):
            tokens.append(Token("OP", "<->"))
            i += 3
            continue
        if text.startswith("->", i):
            tokens.append(Token("OP", "->"))
            i += 2
            continue
        if ch in "()":
            tokens.append(Token(ch, ch))
            i += 1
            continue
        if ch in "~!":
            tokens.append(Token("OP", "not"))
            i += 1
            continue
        if ch == "&":
            tokens.append(Token("OP", "and"))
            i += 1
            continue
        if ch == "|":
            tokens.append(Token("OP", "or"))
            i += 1
            continue

        j = i
        while j < len(text) and (text[j].isalnum() or text[j] == "_"):
            j += 1
        if j == i:
            raise ValueError(f"Unexpected character {ch!r} at offset {i}")
        word = text[i:j]
        lower = word.lower()
        if lower in {"not", "and", "or", "xor"}:
            tokens.append(Token("OP", lower))
        elif lower in {"true", "false"}:
            tokens.append(Token("BOOL", lower.title()))
        else:
            tokens.append(Token("IDENT", word))
        i = j
    return tokens


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        self.idx = 0

    def _peek(self) -> Optional[Token]:
        if self.idx >= len(self.tokens):
            return None
        return self.tokens[self.idx]

    def _take(self) -> Token:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        self.idx += 1
        return tok

    def _accept(self, kind: str, value: Optional[str] = None) -> bool:
        tok = self._peek()
        if tok is None:
            return False
        if tok.kind != kind:
            return False
        if value is not None and tok.value != value:
            return False
        self.idx += 1
        return True

    def parse(self) -> Expr:
        expr = self.parse_iff()
        if self._peek() is not None:
            raise ValueError(f"Unexpected token after parse: {self._peek()}")
        return expr

    def parse_iff(self) -> Expr:
        left = self.parse_implies()
        while self._accept("OP", "<->"):
            right = self.parse_implies()
            left = Iff(left, right)
        return left

    def parse_implies(self) -> Expr:
        left = self.parse_or_xor()
        if self._accept("OP", "->"):
            right = self.parse_implies()
            return Implies(left, right)
        return left

    def parse_or_xor(self) -> Expr:
        left = self.parse_and()
        while True:
            if self._accept("OP", "or"):
                left = Or(left, self.parse_and())
            elif self._accept("OP", "xor"):
                left = Xor(left, self.parse_and())
            else:
                return left

    def parse_and(self) -> Expr:
        left = self.parse_unary()
        while self._accept("OP", "and"):
            left = And(left, self.parse_unary())
        return left

    def parse_unary(self) -> Expr:
        if self._accept("OP", "not"):
            return Not(self.parse_unary())
        tok = self._take()
        if tok.kind == "(":
            inner = self.parse_iff()
            if not self._accept(")"):
                raise ValueError("Missing closing ')' in expression")
            return inner
        if tok.kind == "BOOL":
            return Const(tok.value == "True")
        if tok.kind == "IDENT":
            return Var(tok.value)
        raise ValueError(f"Unexpected token {tok}")


def parse_expression(text: str) -> Expr:
    return Parser(tokenize(text)).parse()
