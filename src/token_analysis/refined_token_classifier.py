from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple


CATEGORY_ORDER = [
    "facts_value",
    "variable_in_facts",
    "query_token",
    "variable_in_expr",
    "operator",
    "expr_last",
    "derived_assignment",
    "others",
]

CATEGORY_DISPLAY_NAMES = {
    "facts_value": "Facts Value",
    "variable_in_facts": "Variable in Facts",
    "query_token": "Query Token",
    "variable_in_expr": "Variable in Expr",
    "operator": "Operator",
    "expr_last": "Expr Last",
    "derived_assignment": "Derived Assignment",
    "others": "Others",
}

CATEGORY_DESCRIPTIONS = {
    "facts_value": "Truth-value tokens in the facts block, such as True and False.",
    "variable_in_facts": "Variable tokens that appear in factual assignments outside the queried expression.",
    "query_token": "Query-region tokens, including the query 'is' anchor, answer suffix tokens, the final terminal token, and any chat-template tail tokens after the raw prompt.",
    "variable_in_expr": "Variable tokens that participate in the queried expression.",
    "operator": "Logical operators inside an expression, such as and/or/not.",
    "expr_last": "The last non-punctuation token inside the queried expression span.",
    "derived_assignment": "Non-punctuation tokens in the intermediate derived assignment for two-hop prompts.",
    "others": "All remaining tokens, including non-terminal punctuation and uncategorized expression tokens.",
}

_BOOL_WORDS = {"true", "false"}
_OP_WORDS = {"and", "or", "not", "xor", "implies", "equivalentto"}


def _norm(tok: object) -> str:
    return str(tok).strip()


def _strip_token_markers(text: str) -> str:
    # Drop common tokenizer word-boundary prefixes so canonical matching works
    # for Llama/Mistral/Qwen-style string tokens.
    stripped = text.strip()
    while stripped.startswith(("Ġ", "▁", "Ċ", "ĉ", "Ŀ", "##")):
        if stripped.startswith("##"):
            stripped = stripped[2:]
        else:
            stripped = stripped[1:]
    return stripped


def _is_special_token(tok: object) -> bool:
    text = _strip_token_markers(_norm(tok))
    if not text:
        return True
    if re.fullmatch(r"<\|.*?\|>", text):
        return True
    if re.fullmatch(r"</?s>", text):
        return True
    return False


def _canon(tok: object) -> str:
    if _is_special_token(tok):
        return ""
    text = _strip_token_markers(_norm(tok)).lower()
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text


def _is_is_token(tok: object) -> bool:
    return _canon(tok) == "is"


def _is_punct(tok: object) -> bool:
    if _is_special_token(tok):
        return True
    text = _strip_token_markers(_norm(tok))
    if not text:
        return True
    return bool(re.fullmatch(r"[^\w]+", text))


def _detect_prompt_order(str_tokens: Sequence[object], sample: Dict[str, object]) -> str:
    explicit = sample.get("prompt_order")
    if explicit in {"facts_first", "expr_first"}:
        return str(explicit)

    prompt = str(sample.get("clean_prompt_symbolic", ""))
    if " is? " in prompt:
        return "expr_first"

    tokens = [_norm(t) for t in str_tokens]
    for tok in tokens[:12]:
        if "?" in tok:
            return "expr_first"

    return "facts_first"


def _find_body_end(str_tokens: Sequence[object]) -> int:
    for i, tok in enumerate(str_tokens):
        c = _canon(tok)
        if c in {"answer", "reason"}:
            return i
    return len(str_tokens)


def _find_constrain_span(
    str_tokens: Sequence[object],
    body_end: int,
) -> Optional[Tuple[int, int]]:
    n_tokens = len(str_tokens)
    if n_tokens <= 1 or body_end >= n_tokens - 1:
        return None

    start = body_end
    if 0 < body_end < n_tokens and _is_punct(str_tokens[body_end - 1]):
        start = body_end - 1

    end = n_tokens - 2
    if end < start:
        return None
    return (start, end)


def _find_query_idx(str_tokens: Sequence[object], prompt_order: str, body_end: int) -> Optional[int]:
    is_positions = [i for i in range(body_end) if _is_is_token(str_tokens[i])]
    if not is_positions:
        return None
    if prompt_order == "expr_first":
        return is_positions[0]
    return is_positions[-1]


def _last_comma_before(str_tokens: Sequence[object], end_idx: int) -> int:
    for i in range(end_idx - 1, -1, -1):
        if "," in _norm(str_tokens[i]):
            return i
    return -1


def _resolve_expr_span(
    str_tokens: Sequence[object],
    prompt_order: str,
    query_idx: int,
    body_end: int,
) -> Optional[Tuple[int, int]]:
    if query_idx < 0:
        return None

    if prompt_order == "expr_first":
        start = 0
        end = query_idx - 1
        while start <= end and (_is_punct(str_tokens[start]) or not _canon(str_tokens[start])):
            start += 1
        while end >= start and _is_punct(str_tokens[end]):
            end -= 1
        if end < start:
            return None
        return (start, end)

    comma = _last_comma_before(str_tokens, query_idx)
    start = comma + 1 if comma >= 0 else 0
    end = query_idx - 1
    while start <= end and _is_punct(str_tokens[start]):
        start += 1
    while end >= start and _is_punct(str_tokens[end]):
        end -= 1
    if end < start:
        return None
    return (start, end)


def _candidate_vars(sample: Dict[str, object]) -> List[str]:
    vars_set = set()
    facts = sample.get("facts")
    if isinstance(facts, dict):
        vars_set.update(str(k) for k in facts.keys())

    iv = sample.get("intermediate_var")
    if isinstance(iv, str) and iv:
        vars_set.add(iv)

    for field in ("expr_symbolic", "intermediate_expr_symbolic"):
        text = sample.get(field)
        if isinstance(text, str):
            vars_set.update(re.findall(r"\b[A-Z]\b", text))

    return sorted(vars_set)


def _fact_vars(sample: Dict[str, object]) -> List[str]:
    facts = sample.get("facts")
    if not isinstance(facts, dict):
        return []
    return sorted(str(k) for k in facts.keys())


def _find_assignment_span(
    str_tokens: Sequence[object],
    sample: Dict[str, object],
    prompt_order: str,
    query_idx: int,
    body_end: int,
) -> Optional[Tuple[int, int]]:
    iv = sample.get("intermediate_var")
    if not isinstance(iv, str) or not iv:
        return None

    iv_upper = iv.upper()

    if prompt_order == "expr_first":
        start_idx = query_idx + 1
        end_idx = body_end
    else:
        start_idx = 0
        end_idx = query_idx

    for i in range(start_idx, end_idx):
        if _canon(str_tokens[i]).upper() != iv_upper:
            continue
        look_right = min(i + 4, end_idx)
        is_idx = None
        for j in range(i + 1, look_right):
            if _is_is_token(str_tokens[j]):
                is_idx = j
                break
        if is_idx is None:
            continue

        span_end = end_idx - 1
        for k in range(is_idx + 1, end_idx):
            if "," in _norm(str_tokens[k]):
                span_end = k
                break
        return (i, span_end)

    return None


def classify_tokens_refined(
    str_tokens: Sequence[object],
    sample: Dict[str, object],
    *,
    include_derived_assignment: bool = True,
    strict: bool = False,
) -> Tuple[List[str], Dict[str, object]]:
    n = len(str_tokens)
    categories = ["others"] * n
    warnings: List[str] = []

    prompt_order = _detect_prompt_order(str_tokens, sample)
    body_end = _find_body_end(str_tokens)
    query_idx = _find_query_idx(str_tokens, prompt_order=prompt_order, body_end=body_end)
    constrain_span = _find_constrain_span(str_tokens, body_end)

    if query_idx is None:
        msg = "Unable to locate query token 'is'"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)
        return categories, {
            "is_valid": False,
            "prompt_order_detected": prompt_order,
            "query_idx": None,
            "expr_span": None,
            "body_end": body_end,
            "constrain_span": list(constrain_span) if constrain_span is not None else None,
            "assignment_span": None,
            "warnings": warnings,
        }

    expr_span = _resolve_expr_span(
        str_tokens=str_tokens,
        prompt_order=prompt_order,
        query_idx=query_idx,
        body_end=body_end,
    )
    if expr_span is None:
        msg = "Unable to locate expression span"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)
        return categories, {
            "is_valid": False,
            "prompt_order_detected": prompt_order,
            "query_idx": query_idx,
            "expr_span": None,
            "body_end": body_end,
            "constrain_span": list(constrain_span) if constrain_span is not None else None,
            "assignment_span": None,
            "warnings": warnings,
        }

    expr_start, expr_end = expr_span
    assignment_span = _find_assignment_span(
        str_tokens=str_tokens,
        sample=sample,
        prompt_order=prompt_order,
        query_idx=query_idx,
        body_end=body_end,
    )
    var_set = set(_candidate_vars(sample))
    fact_var_set = set(_fact_vars(sample))

    priority = {
        "query_token": 6,
        "expr_last": 5,
        "facts_value": 4,
        "variable_in_facts": 4,
        "variable_in_expr": 3,
        "operator": 2,
        "derived_assignment": 1,
        "others": 0,
    }

    def assign(idx: int, cat: str) -> None:
        if idx < 0 or idx >= n:
            return
        if priority[cat] >= priority[categories[idx]]:
            categories[idx] = cat

    assign(query_idx, "query_token")

    for i in range(body_end):
        if _canon(str_tokens[i]) in _BOOL_WORDS:
            assign(i, "facts_value")

    assignment_start = None if assignment_span is None else assignment_span[0]
    assignment_end = None if assignment_span is None else assignment_span[1]
    for i in range(body_end):
        if expr_start <= i <= expr_end:
            continue
        if assignment_start is not None and assignment_end is not None and assignment_start <= i <= assignment_end:
            continue
        c = _canon(str_tokens[i])
        if not c:
            continue
        if c.upper() in fact_var_set:
            assign(i, "variable_in_facts")

    for i in range(expr_start, expr_end + 1):
        c = _canon(str_tokens[i])
        if not c:
            continue
        if c in _OP_WORDS or "¬" in _norm(str_tokens[i]):
            assign(i, "operator")
            continue
        if c.upper() in var_set:
            assign(i, "variable_in_expr")

    expr_last_idx = None
    for i in range(expr_end, expr_start - 1, -1):
        if _is_punct(str_tokens[i]):
            continue
        if categories[i] == "query_token":
            continue
        expr_last_idx = i
        break
    if expr_last_idx is not None:
        assign(expr_last_idx, "expr_last")

    if constrain_span is not None:
        s_start, s_end = constrain_span
        del s_end
        for i in range(s_start, n):
            assign(i, "query_token")

    raw_prompt_token_count = int(sample.get("raw_prompt_token_count", n))
    if 0 <= raw_prompt_token_count < n:
        for i in range(raw_prompt_token_count, n):
            assign(i, "query_token")

    if include_derived_assignment and assignment_span is not None:
        a_start, a_end = assignment_span
        for i in range(a_start, min(a_end + 1, n)):
            if not _is_punct(str_tokens[i]):
                assign(i, "derived_assignment")

    return categories, {
        "is_valid": True,
        "prompt_order_detected": prompt_order,
        "query_idx": query_idx,
        "expr_span": [expr_start, expr_end],
        "body_end": body_end,
        "constrain_span": list(constrain_span) if constrain_span is not None else None,
        "assignment_span": list(assignment_span) if assignment_span is not None else None,
        "warnings": warnings,
    }
