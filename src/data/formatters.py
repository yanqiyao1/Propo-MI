from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Tuple


def _bool_word(value: bool, truth_style: str) -> str:
    if truth_style == "full":
        return "True" if value else "False"
    if truth_style == "short":
        return "T" if value else "F"
    raise ValueError(f"Unknown truth_style {truth_style!r}")


def format_facts(facts: Dict[str, bool], truth_style: str = "full") -> str:
    parts = [f"{k} is {_bool_word(facts[k], truth_style)}" for k in sorted(facts)]
    return ", ".join(parts)


def format_derived_steps(steps: Iterable[Tuple[str, str]], truth_style: str = "full") -> str:
    rendered: List[str] = []
    for var, expr_text in steps:
        rendered.append(f"{var} is {_expr_to_compact(expr_text, truth_style=truth_style)}")
    return ", ".join(rendered)


def _expr_to_compact(expr_text: str, truth_style: str = "short") -> str:
    if truth_style == "short":
        text = re.sub(r"\bTrue\b", "T", expr_text)
        text = re.sub(r"\bFalse\b", "F", text)
        return text
    if truth_style == "full":
        return expr_text
    raise ValueError(f"Unknown truth_style {truth_style!r}")


def _expr_to_verbose(expr_text: str, truth_style: str = "full") -> str:
    if truth_style == "full":
        return expr_text
    if truth_style == "short":
        text = re.sub(r"\bTrue\b", "T", expr_text)
        text = re.sub(r"\bFalse\b", "F", text)
        return text
    raise ValueError(f"Unknown truth_style {truth_style!r}")


def _render_suffix(mode: str) -> str:
    if mode == "cot":
        return " Reason step by step, then end with one final word: True or False."
    if mode == "nocot":
        return " Answer with one word only: True or False."
    raise ValueError(f"Unknown mode {mode!r}")


PROMPT_ENDING_CHOICES = ("answer_suffix", "terminal_is")


def normalize_prompt_ending(prompt_ending: str) -> str:
    if prompt_ending not in PROMPT_ENDING_CHOICES:
        raise ValueError(f"Unknown prompt_ending {prompt_ending!r}")
    return prompt_ending


def resolve_prompt_ending(row: Mapping[str, object], default: str = "answer_suffix") -> str:
    value = str(row.get("prompt_ending", default)).strip()
    return normalize_prompt_ending(value or default)


def resolve_query_expr_text(
    row: Mapping[str, object],
    *,
    prompt_style: str,
    kind: str = "clean",
) -> str:
    if prompt_style not in {"symbolic", "semi_natural"}:
        raise ValueError(f"Unknown prompt_style {prompt_style!r}")
    if kind not in {"clean", "corrupted"}:
        raise ValueError(f"Unknown prompt kind {kind!r}")

    base_field = "expr_symbolic" if prompt_style == "symbolic" else "expr_semi_natural"
    if kind == "corrupted":
        corrupted_field = f"corrupted_{base_field}"
        value = row.get(corrupted_field)
        if value is not None:
            return str(value)
    return str(row[base_field])


def _render_terminal_query(expr_text: str, truth_style: str = "full") -> str:
    return f"{_expr_to_compact(expr_text, truth_style=truth_style)} is"


def build_prompt(
    facts: Dict[str, bool],
    expr_text: str,
    mode: str = "nocot",
    template_style: str = "compact",
    truth_style: str = "full",
    derived_steps: Iterable[Tuple[str, str]] | None = None,
    prompt_order: str = "facts_first",
    prompt_ending: str = "answer_suffix",
) -> str:
    if template_style not in {"compact", "verbose"}:
        raise ValueError(f"Unknown template_style {template_style!r}")
    if prompt_order not in {"facts_first", "expr_first"}:
        raise ValueError(f"Unknown prompt_order {prompt_order!r}")

    prompt_ending = normalize_prompt_ending(prompt_ending)
    if prompt_ending == "terminal_is" and prompt_order != "facts_first":
        raise ValueError("prompt_ending='terminal_is' currently requires prompt_order='facts_first'")

    steps = list(derived_steps or [])
    suffix = _render_suffix(mode)

    if template_style == "compact":
        if prompt_order == "facts_first":
            parts: List[str] = [format_facts(facts, truth_style=truth_style)]
            if steps:
                parts.append(format_derived_steps(steps, truth_style=truth_style))
            parts.append(_render_terminal_query(expr_text, truth_style=truth_style))
            body = ", ".join(parts)
            if prompt_ending == "terminal_is":
                return body
            return body + "." + suffix

        tail_parts: List[str] = [format_facts(facts, truth_style=truth_style)]
        if steps:
            tail_parts.append(format_derived_steps(steps, truth_style=truth_style))
        return f"{_expr_to_compact(expr_text, truth_style=truth_style)} is? " + ", ".join(tail_parts) + "." + suffix

    facts_text = format_facts(facts, truth_style=truth_style)
    expr_rendered = _expr_to_verbose(expr_text, truth_style=truth_style)

    if prompt_order == "facts_first":
        if not steps:
            if prompt_ending == "terminal_is":
                return f"Given the facts: {facts_text}. Evaluate the proposition: {expr_rendered} is"
            return f"Given the facts: {facts_text}. Evaluate the proposition: {expr_rendered}. {suffix.strip()}"
        step_text = format_derived_steps(steps, truth_style=truth_style)
        if prompt_ending == "terminal_is":
            return (
                f"Given the facts: {facts_text}. "
                f"Derived assignments: {step_text}. "
                f"Evaluate the proposition: {expr_rendered} is"
            )
        return (
            f"Given the facts: {facts_text}. "
            f"Derived assignments: {step_text}. "
            f"Evaluate the proposition: {expr_rendered}. "
            f"{suffix.strip()}"
        )

    if not steps:
        return f"Evaluate the proposition: {expr_rendered}. Given the facts: {facts_text}. {suffix.strip()}"
    step_text = format_derived_steps(steps, truth_style=truth_style)
    return (
        f"Evaluate the proposition: {expr_rendered}. "
        f"Given the facts: {facts_text}. "
        f"Derived assignments: {step_text}. "
        f"{suffix.strip()}"
    )


def build_depth_prompt(
    *,
    hop: str,
    facts: Dict[str, bool],
    query_expr_text: str,
    mode: str = "nocot",
    derived_steps: Iterable[Tuple[str, str]] | None = None,
    prompt_order: str = "facts_first",
    prompt_ending: str = "answer_suffix",
) -> str:
    if hop == "one_hop":
        return build_prompt(
            facts=facts,
            expr_text=query_expr_text,
            mode=mode,
            template_style="compact",
            truth_style="full",
            derived_steps=None,
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        )
    if hop == "two_hop":
        return build_prompt(
            facts=facts,
            expr_text=query_expr_text,
            mode=mode,
            template_style="compact",
            truth_style="full",
            derived_steps=derived_steps,
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        )
    raise ValueError(f"Unknown hop {hop!r}")
