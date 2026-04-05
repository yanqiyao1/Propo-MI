from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from src.data.formatters import build_depth_prompt, format_derived_steps, format_facts, resolve_prompt_ending, resolve_query_expr_text


def resolve_prompt(row: Dict[str, object], prompt_style: str, kind: str) -> str:
    if kind not in {"clean", "corrupted"}:
        raise ValueError(f"Unknown prompt kind {kind!r}")

    field = f"{kind}_prompt_{prompt_style}"
    val = row.get(field)
    if val is not None:
        return str(val)

    hop = str(row["hop"])

    if kind == "clean":
        facts = row.get("facts")
    else:
        facts = row.get("corrupted_facts")

    if facts is None:
        raise KeyError(
            f"Cannot rebuild {kind} prompt for row id={row.get('id')} because required facts are missing"
        )

    derived_steps = None
    if hop == "two_hop":
        iv = str(row["intermediate_var"])
        ie_field = "intermediate_expr_symbolic" if prompt_style == "symbolic" else "intermediate_expr_semi_natural"
        derived_steps = [(iv, str(row[ie_field]))]

    prompt_order = str(row.get("prompt_order", "facts_first"))
    return build_depth_prompt(
        hop=hop,
        facts=dict(facts),
        query_expr_text=resolve_query_expr_text(row, prompt_style=prompt_style, kind=kind),
        mode="nocot",
        derived_steps=derived_steps,
        prompt_order=prompt_order,
        prompt_ending=resolve_prompt_ending(row),
    )


def find_comma_positions(str_tokens: Sequence[str]) -> List[int]:
    out: List[int] = []
    for i, tok in enumerate(str_tokens):
        if tok.strip() == ",":
            out.append(i)
    return out


def compute_true_false_probs(
    logits: Any,
    true_token_id: int,
    false_token_id: int,
    pos: int = -1,
) -> Tuple[Any, Any]:
    true_logit = logits[:, pos, true_token_id]
    false_logit = logits[:, pos, false_token_id]
    pair = true_logit.new_zeros(true_logit.shape + (2,))
    pair[..., 0] = true_logit
    pair[..., 1] = false_logit
    probs = pair.softmax(dim=-1)
    return probs[..., 0], probs[..., 1]


def compute_margin_for_label(
    logits: Any,
    label: bool,
    true_token_id: int,
    false_token_id: int,
    pos: int = -1,
) -> Any:
    true_prob, false_prob = compute_true_false_probs(
        logits,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        pos=pos,
    )
    return (true_prob - false_prob) if label else (false_prob - true_prob)


def compute_selected_token_prob(
    logits: Any,
    true_token_id: int,
    false_token_id: int,
    pos: int = -1,
) -> Any:
    true_prob, false_prob = compute_true_false_probs(
        logits,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        pos=pos,
    )
    return true_prob.maximum(false_prob)


def pick_terminal_token_indices(seq_len: int) -> List[int]:
    if seq_len <= 0:
        return []
    return [seq_len - 1]


def infer_hop_from_rows(rows: Sequence[Dict[str, object]]) -> str:
    hops = sorted(set(str(r.get("hop", "")) for r in rows))
    if not hops:
        return "unknown"
    if len(hops) == 1:
        return hops[0]
    return "mixed"


def _find_answer_suffix_start(prompt: str) -> int:
    markers = [
        " Answer with one word only: True or False.",
        " Reason step by step, then end with one final word: True or False.",
    ]
    starts = [prompt.find(marker) for marker in markers if prompt.find(marker) >= 0]
    return min(starts) if starts else len(prompt)


def _find_terminal_char_index(prompt: str, answer_start: int) -> int:
    idx = min(answer_start - 1, len(prompt) - 1)
    while idx >= 0 and prompt[idx].isspace():
        idx -= 1
    if idx < 0:
        raise ValueError("Unable to locate terminal character before answer suffix")
    return idx


def _find_query_anchor_start_facts_first(
    prompt: str,
    *,
    query_text: str,
    search_start: int,
    row_id: object,
) -> int:
    terminal_query = f"{query_text} is"
    query_start = prompt.find(terminal_query, search_start)
    if query_start < 0:
        raise ValueError(f"Unable to locate terminal query text in prompt: id={row_id}")
    return query_start + len(query_text) + 1


def _find_query_anchor_span_expr_first(
    prompt: str,
    *,
    query_text: str,
    row_id: object,
) -> Tuple[int, int]:
    query_prefix = f"{query_text} is?"
    query_start = prompt.find(query_prefix)
    if query_start < 0:
        raise ValueError(f"Unable to locate query prefix in prompt: id={row_id}")
    query_anchor_start = query_start + len(query_text) + 1
    query_anchor_end = min(len(prompt), query_start + len(query_prefix))
    return query_anchor_start, query_anchor_end


def build_region_char_spans(
    prompt: str,
    row: Dict[str, object],
    prompt_style: str,
    kind: str = "clean",
) -> Dict[str, List[Tuple[int, int]]]:
    if kind not in {"clean", "corrupted"}:
        raise ValueError(f"Unknown prompt kind {kind!r}")

    hop = str(row["hop"])
    prompt_order = str(row.get("prompt_order", "facts_first"))

    if kind == "clean":
        facts = row.get("facts")
    else:
        facts = row.get("corrupted_facts")
    if facts is None:
        raise KeyError(f"Missing facts for {kind} prompt region parsing: id={row.get('id')}")

    facts_text = format_facts(dict(facts), truth_style="full")
    query_text = resolve_query_expr_text(row, prompt_style=prompt_style, kind=kind)

    derived_text = None
    if hop == "two_hop":
        iv = str(row["intermediate_var"])
        ie_field = "intermediate_expr_symbolic" if prompt_style == "symbolic" else "intermediate_expr_semi_natural"
        derived_text = format_derived_steps([(iv, str(row[ie_field]))], truth_style="full")

    answer_start = _find_answer_suffix_start(prompt)
    prompt_end = len(prompt)

    spans: Dict[str, List[Tuple[int, int]]] = {
        "facts_region": [],
        "expression_region": [],
        "query_region": [],
    }

    if prompt_order == "facts_first":
        facts_start = prompt.find(facts_text)
        if facts_start < 0:
            raise ValueError(f"Unable to locate facts text in prompt: id={row.get('id')}")
        facts_end = facts_start + len(facts_text)
        if facts_end < len(prompt) and prompt[facts_end] == ",":
            facts_end += 1
        spans["facts_region"].append((facts_start, facts_end))

        if hop == "two_hop" and derived_text is not None:
            derived_start = prompt.find(derived_text, facts_end)
            if derived_start < 0:
                raise ValueError(f"Unable to locate derived-step text in prompt: id={row.get('id')}")
            query_start = prompt.find(query_text, derived_start + len(derived_text))
            if query_start < 0:
                raise ValueError(f"Unable to locate query expression text in prompt: id={row.get('id')}")
            expr_end = query_start + len(query_text)
            if derived_start < expr_end:
                spans["expression_region"].append((derived_start, expr_end))
            query_anchor_start = _find_query_anchor_start_facts_first(
                prompt,
                query_text=query_text,
                search_start=derived_start + len(derived_text),
                row_id=row.get("id"),
            )
            if query_anchor_start < prompt_end:
                spans["query_region"].append((query_anchor_start, prompt_end))
            return spans

        query_start = prompt.find(query_text, facts_end)
        if query_start < 0:
            raise ValueError(f"Unable to locate query expression text in prompt: id={row.get('id')}")
        expr_end = query_start + len(query_text)
        if query_start < expr_end:
            spans["expression_region"].append((query_start, expr_end))
        query_anchor_start = _find_query_anchor_start_facts_first(
            prompt,
            query_text=query_text,
            search_start=facts_end,
            row_id=row.get("id"),
        )
        if query_anchor_start < prompt_end:
            spans["query_region"].append((query_anchor_start, prompt_end))
        return spans

    query_start = prompt.find(query_text)
    if query_start < 0:
        raise ValueError(f"Unable to locate query expression text in prompt: id={row.get('id')}")
    spans["expression_region"].append((query_start, query_start + len(query_text)))
    query_anchor_start, query_anchor_end = _find_query_anchor_span_expr_first(
        prompt,
        query_text=query_text,
        row_id=row.get("id"),
    )
    if query_anchor_start < query_anchor_end:
        spans["query_region"].append((query_anchor_start, query_anchor_end))
    if answer_start < prompt_end:
        spans["query_region"].append((answer_start, prompt_end))

    facts_start = prompt.find(facts_text, query_anchor_end)
    if facts_start < 0:
        raise ValueError(f"Unable to locate facts text after query in prompt: id={row.get('id')}")
    facts_end = facts_start + len(facts_text)

    if hop == "two_hop" and derived_text is not None:
        if facts_end < len(prompt) and prompt[facts_end] == ",":
            facts_end += 1
        spans["facts_region"].append((facts_start, facts_end))

        derived_start = prompt.find(derived_text, facts_end)
        if derived_start < 0:
            raise ValueError(f"Unable to locate derived-step text in prompt: id={row.get('id')}")
        derived_end = derived_start + len(derived_text)
        if derived_start < derived_end:
            spans["expression_region"].append((derived_start, derived_end))
        return spans

    spans["facts_region"].append((facts_start, facts_end))
    return spans


def pick_region_token_indices(
    tokenizer,
    prompt: str,
    spans: Sequence[Tuple[int, int]],
    expected_seq_len: int = 0,
) -> List[int]:
    if tokenizer is None:
        raise ValueError("Tokenizer is required for precise region alignment")

    enc = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]

    shift = 0
    if expected_seq_len > 0:
        if len(offsets) == expected_seq_len:
            shift = 0
        elif len(offsets) + 1 == expected_seq_len:
            shift = 1
        else:
            raise ValueError(
                f"Offset length mismatch: offsets={len(offsets)} expected_seq_len={expected_seq_len}"
            )

    out: List[int] = []
    for idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        for span_start, span_end in spans:
            if max(start, span_start) < min(end, span_end):
                out.append(idx + shift)
                break
    return out
