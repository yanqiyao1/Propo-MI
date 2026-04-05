from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Dict, List

from .logic_ast import And, Const, Expr, Iff, Not, Or, Var


@dataclass(frozen=True)
class RuleTemplate:
    name: str
    category: str
    formal_definition: str
    build: Callable[[Dict[str, Expr], random.Random], Expr]


def _identity(v: Dict[str, Expr], rng: random.Random) -> Expr:
    return And(v["A"], Const(True)) if rng.random() < 0.5 else Or(v["A"], Const(False))


def _domination(v: Dict[str, Expr], rng: random.Random) -> Expr:
    return And(v["A"], Const(False)) if rng.random() < 0.5 else Or(v["A"], Const(True))


def _idempotent(v: Dict[str, Expr], rng: random.Random) -> Expr:
    return And(v["A"], v["A"]) if rng.random() < 0.5 else Or(v["A"], v["A"])


def _double_negation(v: Dict[str, Expr], rng: random.Random) -> Expr:
    del rng
    return Not(Not(v["A"]))


def _excluded_middle(v: Dict[str, Expr], rng: random.Random) -> Expr:
    del rng
    return Or(v["A"], Not(v["A"]))


def _contradiction(v: Dict[str, Expr], rng: random.Random) -> Expr:
    del rng
    return And(v["A"], Not(v["A"]))


def _demorgan(v: Dict[str, Expr], rng: random.Random) -> Expr:
    del rng
    return Or(Not(v["A"]), Not(v["B"]))


def _distributive(v: Dict[str, Expr], rng: random.Random) -> Expr:
    del rng
    return And(v["A"], Or(v["B"], v["C"]))


def _commutative(v: Dict[str, Expr], rng: random.Random) -> Expr:
    del rng
    return And(v["A"], v["B"])


def _associative(v: Dict[str, Expr], rng: random.Random) -> Expr:
    return And(And(v["A"], v["B"]), v["C"]) if rng.random() < 0.5 else And(v["A"], And(v["B"], v["C"]))


def _absorption(v: Dict[str, Expr], rng: random.Random) -> Expr:
    del rng
    return And(v["A"], Or(v["A"], v["B"]))


RULE_TEMPLATES: List[RuleTemplate] = [
    RuleTemplate(
        "identity",
        "basic_laws",
        "P and T <-> P ; P or F <-> P",
        _identity,
    ),
    RuleTemplate(
        "domination",
        "basic_laws",
        "P and F <-> F ; P or T <-> T",
        _domination,
    ),
    RuleTemplate(
        "idempotent",
        "basic_laws",
        "P and P <-> P ; P or P <-> P",
        _idempotent,
    ),
    RuleTemplate(
        "double_negation",
        "negation_laws",
        "not(not P) <-> P",
        _double_negation,
    ),
    RuleTemplate(
        "excluded_middle",
        "negation_laws",
        "P or not P <-> T",
        _excluded_middle,
    ),
    RuleTemplate(
        "contradiction",
        "negation_laws",
        "P and not P <-> F",
        _contradiction,
    ),
    RuleTemplate(
        "commutative",
        "multi_variable_laws",
        "P and Q <-> Q and P",
        _commutative,
    ),
    RuleTemplate(
        "associative",
        "multi_variable_laws",
        "(P and Q) and R <-> P and (Q and R)",
        _associative,
    ),
    RuleTemplate(
        "distributive",
        "multi_variable_laws",
        "P and (Q or R) <-> (P and Q) or (P and R)",
        _distributive,
    ),
    RuleTemplate(
        "demorgan",
        "multi_variable_laws",
        "not(P and Q) <-> (not P) or (not Q)",
        _demorgan,
    ),
    RuleTemplate(
        "absorption",
        "multi_variable_laws",
        "P and (P or Q) <-> P",
        _absorption,
    ),
]


def build_one_hop_expr(rule_name: str, rng: random.Random) -> Expr:
    template = next((x for x in RULE_TEMPLATES if x.name == rule_name), None)
    if template is None:
        raise KeyError(f"Unknown rule name {rule_name}")
    vars_map = {"A": Var("A"), "B": Var("B"), "C": Var("C"), "D": Var("D")}
    return template.build(vars_map, rng)


def build_two_hop_expr(rule_name: str, rng: random.Random) -> Expr:
    expr = build_one_hop_expr(rule_name, rng)
    wrappers = [
        lambda x: Not(Not(x)),
        lambda x: And(x, Const(True)),
        lambda x: Or(x, Const(False)),
        lambda x: Iff(x, Const(True)),
    ]
    expr = rng.choice(wrappers)(expr)
    expr = rng.choice(wrappers)(expr)
    return expr


def all_rule_names() -> List[str]:
    return [x.name for x in RULE_TEMPLATES]


def get_rule_template(rule_name: str) -> RuleTemplate:
    template = next((x for x in RULE_TEMPLATES if x.name == rule_name), None)
    if template is None:
        raise KeyError(f"Unknown rule name {rule_name}")
    return template
