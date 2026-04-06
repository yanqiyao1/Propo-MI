from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from src.eval.io_utils import balanced_sample_by_rule
from src.model_loading import (
    load_hooked_transformer,
    resolve_model_artifact_tags,
    resolve_model_prompt,
    resolve_true_false_token_ids,
    to_tokens,
)
from src.plot_style import apply_paper_style


# =============================================================================
# Plot style
# =============================================================================


# =============================================================================
# IO helpers
# =============================================================================


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(dict(json.loads(line)))
    return rows


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(r) for r in rows]
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys: List[str] = []
        for row in rows_list:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows_list)


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


# =============================================================================
# dPD helpers
# =============================================================================


def _pair_probs_from_logits(
    logits: torch.Tensor,
    token_a_id: int,
    token_b_id: int,
    pos: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logit_a = logits[:, pos, token_a_id]
    logit_b = logits[:, pos, token_b_id]
    pair = torch.stack([logit_a, logit_b], dim=-1)
    probs = torch.softmax(pair, dim=-1)
    return probs[..., 0], probs[..., 1]


def compute_prob_diff(logits: torch.Tensor, token_a_id: int, token_b_id: int, pos: int = -1) -> torch.Tensor:
    p_a, p_b = _pair_probs_from_logits(logits, token_a_id, token_b_id, pos=pos)
    return p_a - p_b


def compute_correct_incorrect_prob_diff(
    logits: torch.Tensor,
    label: bool | torch.Tensor,
    true_token_id: int,
    false_token_id: int,
    pos: int = -1,
) -> torch.Tensor:
    raw = compute_prob_diff(logits, true_token_id, false_token_id, pos=pos)
    if isinstance(label, torch.Tensor):
        sign = torch.where(label.to(dtype=torch.bool), 1.0, -1.0).to(device=raw.device, dtype=raw.dtype)
        return raw * sign
    return raw if bool(label) else -raw


def bool_prediction_from_logits(
    logits: torch.Tensor,
    true_token_id: int,
    false_token_id: int,
    pos: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_prob, false_prob = _pair_probs_from_logits(logits, true_token_id, false_token_id, pos=pos)
    pred = true_prob > false_prob
    return pred, true_prob, false_prob


# =============================================================================
# Prompt formatters (standalone copy)
# =============================================================================


def _bool_word(value: bool, truth_style: str) -> str:
    if truth_style == "full":
        return "True" if value else "False"
    if truth_style == "short":
        return "T" if value else "F"
    raise ValueError(f"Unknown truth_style {truth_style!r}")


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


def normalize_prompt_ending(prompt_ending: str) -> str:
    if prompt_ending not in {"answer_suffix", "terminal_is"}:
        raise ValueError(f"Unknown prompt_ending {prompt_ending!r}")
    return prompt_ending


def resolve_prompt_ending(row: Mapping[str, object], default: str = "answer_suffix") -> str:
    value = str(row.get("prompt_ending", default)).strip()
    return normalize_prompt_ending(value or default)


def resolve_query_expr_text(row: Mapping[str, object], *, prompt_style: str, kind: str = "clean") -> str:
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


def _format_facts(facts: Dict[str, bool], truth_style: str = "full") -> str:
    parts = [f"{k} is {_bool_word(facts[k], truth_style)}" for k in sorted(facts)]
    return ", ".join(parts)


def _format_derived_steps(steps: Iterable[Tuple[str, str]], truth_style: str = "full") -> str:
    rendered: List[str] = []
    for var, expr_text in steps:
        rendered.append(f"{var} is {_expr_to_compact(expr_text, truth_style=truth_style)}")
    return ", ".join(rendered)


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
            parts: List[str] = [_format_facts(facts, truth_style=truth_style)]
            if steps:
                parts.append(_format_derived_steps(steps, truth_style=truth_style))
            parts.append(_render_terminal_query(expr_text, truth_style=truth_style))
            body = ", ".join(parts)
            if prompt_ending == "terminal_is":
                return body
            return body + "." + suffix
        tail_parts: List[str] = [_format_facts(facts, truth_style=truth_style)]
        if steps:
            tail_parts.append(_format_derived_steps(steps, truth_style=truth_style))
        return f"{_expr_to_compact(expr_text, truth_style=truth_style)} is? " + ", ".join(tail_parts) + "." + suffix

    facts_text = _format_facts(facts, truth_style=truth_style)
    expr_rendered = _expr_to_verbose(expr_text, truth_style=truth_style)
    if prompt_order == "facts_first":
        if not steps:
            if prompt_ending == "terminal_is":
                return f"Given the facts: {facts_text}. Evaluate the proposition: {expr_rendered} is"
            return f"Given the facts: {facts_text}. Evaluate the proposition: {expr_rendered}. {suffix.strip()}"
        step_text = _format_derived_steps(steps, truth_style=truth_style)
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
    step_text = _format_derived_steps(steps, truth_style=truth_style)
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


def resolve_prompt(row: Dict[str, object], prompt_style: str, kind: str) -> str:
    if kind not in {"clean", "corrupted"}:
        raise ValueError(f"Unknown prompt kind {kind!r}")
    field = f"{kind}_prompt_{prompt_style}"
    val = row.get(field)
    if val is not None:
        return str(val)

    hop = str(row["hop"])
    facts = row.get("facts") if kind == "clean" else row.get("corrupted_facts")
    if facts is None:
        raise KeyError(f"Cannot rebuild {kind} prompt for row id={row.get('id')}")

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


@dataclass(frozen=True)
class PromptTokenization:
    raw_prompt: str
    model_input_prompt: str
    tokens: torch.Tensor
    raw_str_tokens: List[str]
    body_start: int

    @property
    def answer_pos(self) -> int:
        return int(self.tokens.shape[1] - 1)


def _prompt_body_start_token_idx(
    model,
    model_input_prompt: str,
    raw_prompt: str,
    tokens: torch.Tensor,
) -> int:
    if model_input_prompt == raw_prompt:
        return 0

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return 0

    raw_char_start = model_input_prompt.find(raw_prompt)
    if raw_char_start < 0:
        return 0

    try:
        encoded = tokenizer(model_input_prompt, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, ValueError, NotImplementedError):
        return 0

    offsets = encoded.get("offset_mapping")
    input_ids = encoded.get("input_ids")
    if offsets is None or input_ids is None:
        return 0

    offset_list = list(offsets)
    input_id_list = list(input_ids)
    token_ids = tokens[0].tolist()

    prefix_len = 0
    if len(token_ids) >= len(input_id_list) and token_ids[-len(input_id_list):] == input_id_list:
        prefix_len = len(token_ids) - len(input_id_list)
    else:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if (
            bos_token_id is not None
            and token_ids
            and token_ids[0] == int(bos_token_id)
            and token_ids[1 : 1 + len(input_id_list)] == input_id_list
        ):
            prefix_len = 1

    for idx, span in enumerate(offset_list):
        if span is None or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        if end > raw_char_start:
            return prefix_len + idx

    return prefix_len


def tokenize_prompt_for_eval_alignment(model, prompt: str) -> PromptTokenization:
    model_input_prompt = resolve_model_prompt(model, prompt, enable_thinking=False)
    tokens = to_tokens(model, model_input_prompt)
    raw_tokens = to_tokens(model, prompt, prepend_bos=False)
    return PromptTokenization(
        raw_prompt=prompt,
        model_input_prompt=model_input_prompt,
        tokens=tokens,
        raw_str_tokens=model.to_str_tokens(raw_tokens[0]),
        body_start=_prompt_body_start_token_idx(model, model_input_prompt, prompt, tokens),
    )


# =============================================================================
# Token structure helpers
# =============================================================================


def find_comma_positions(str_tokens: Sequence[str]) -> List[int]:
    body_end = prompt_body_end(str_tokens)
    return [i for i, tok in enumerate(str_tokens[:body_end]) if tok.strip() == ","]


def find_fact_positions(str_tokens: Sequence[str]) -> List[int]:
    body_end = prompt_body_end(str_tokens)
    return [i for i, tok in enumerate(str_tokens[:body_end]) if tok.strip() in {"True", "False"}]


def region_starts_from_commas(str_tokens: Sequence[str]) -> List[int]:
    commas = find_comma_positions(str_tokens)
    starts = [0]
    for c in commas:
        if c + 1 < len(str_tokens):
            starts.append(c + 1)
    return sorted(set(starts))


def infer_prompt_order(row: Mapping[str, object], prompt_text: str) -> str:
    explicit = row.get("prompt_order")
    if explicit in {"facts_first", "expr_first"}:
        return str(explicit)
    return "expr_first" if " is? " in prompt_text else "facts_first"


def prompt_body_end(str_tokens: Sequence[str]) -> int:
    for i, tok in enumerate(str_tokens):
        t = tok.strip().lower()
        if "answer" in t or "reason" in t:
            return i
    return len(str_tokens)


def query_is_pos(str_tokens: Sequence[str], prompt_order: str) -> int:
    body_end = prompt_body_end(str_tokens)
    is_positions = [i for i in range(body_end) if str_tokens[i].strip() == "is"]
    if not is_positions:
        return max(0, len(str_tokens) - 2)
    return int(is_positions[0] if prompt_order == "expr_first" else is_positions[-1])


def tril_mean(attn: Any) -> Any:
    mask = attn.new_ones(attn.shape).tril()
    return (attn * mask).sum() / (mask.sum() + 1e-12)


# =============================================================================
# Dataset filter and stats helpers
# =============================================================================


ROLE_ORDER = ["fact_retrieval", "splitting", "transmission"]
ROLE_LABEL = {
    "fact_retrieval": "Fact-Retrieval",
    "splitting": "Splitting",
    "transmission": "Transmission",
}
ROLE_COLOR = {
    "fact_retrieval": "#D55E00",
    "splitting": "#0072B2",
    "transmission": "#009E73",
}
ROLE_SCORE_COLUMNS = {
    "fact_retrieval": "score_fact",
    "splitting": "score_split",
    "transmission": "score_trans",
}


def _coerce_role_score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")



def resolve_role_label(row: Mapping[str, object], role_col: str = "role_label") -> str | None:
    role_raw = str(row.get(role_col, "")).strip()
    if role_raw in ROLE_ORDER:
        return role_raw

    ranked = []
    for idx, role in enumerate(ROLE_ORDER):
        score = _coerce_role_score(row.get(ROLE_SCORE_COLUMNS[role]))
        if math.isfinite(score):
            ranked.append((score, -idx, role))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return str(ranked[0][2])


def resolve_dual_correct_default(model_id: str, dataset: str) -> str:
    if dataset:
        return dataset

    candidates = []
    for tag in resolve_model_artifact_tags(model_id):
        candidates.append(f"artifacts/filtered_dual_correct_{tag}.jsonl")

    size = "14b" if "14B" in model_id or "14b" in model_id else "8b"
    candidates.extend([
        f"artifacts/filtered_dual_correct_Qwen3-{size}.jsonl",
        f"artifacts/filtered_dual_correct_{size}.jsonl",
    ])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def filter_rows(
    *,
    input_path: str,
    hop: str,
    prompt_order: str,
    max_samples: int,
    require_dual_correct: bool,
    prompt_style: str = "symbolic",
) -> List[Dict[str, object]]:
    rows_all = read_jsonl(Path(input_path))
    if require_dual_correct and rows_all and "correct_clean" not in rows_all[0]:
        raise ValueError("Input rows lack correct_clean/correct_corrupted fields.")
    selected: List[Dict[str, object]] = []
    for row in rows_all:
        if hop != "all" and str(row.get("hop", "")) != hop:
            continue
        clean_prompt = resolve_prompt(row, prompt_style=prompt_style, kind="clean")
        row_order = infer_prompt_order(row, clean_prompt)
        if prompt_order != "all" and row_order != prompt_order:
            continue
        if require_dual_correct:
            if not bool(row.get("correct_clean", False)):
                continue
            if not bool(row.get("correct_corrupted", False)):
                continue
        selected.append(dict(row))
    if max_samples > 0:
        selected = balanced_sample_by_rule(selected, max_samples=max_samples)
    return selected


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        v = float(arr[0]) if arr.size == 1 else 0.0
        return v, v
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = float(arr[rng.integers(0, n, size=n)].mean())
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def paired_sign_test_pvalue(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    pos = int((arr > 0).sum())
    neg = int((arr < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    if n <= 2000:
        tail = sum(math.comb(n, i) / float(2**n) for i in range(k + 1))
        return float(min(1.0, 2.0 * tail))
    mu, sigma = n * 0.5, math.sqrt(n * 0.25)
    z = (k + 0.5 - mu) / sigma
    return float(min(1.0, 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))))


def safe_mean(vals: Sequence[float]) -> float:
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0


def safe_sem(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.std(ddof=0) / np.sqrt(max(1, arr.size))) if arr.size else 0.0


def stable_int_from_text(text: str) -> int:
    val = 0
    for ch in text:
        val = (val * 131 + ord(ch)) % 1_000_000_007
    return int(val)


def serialize_head(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def serialize_heads(heads: Sequence[Tuple[int, int]]) -> str:
    return ";".join(serialize_head(l, h) for l, h in heads)


@dataclass(frozen=True)
class SamplePair:
    sample_id: str
    hop: str
    rule: str
    label_clean: bool
    label_corrupt: bool
    clean_tokens: torch.Tensor
    corrupt_tokens: torch.Tensor
    query_pos_clean: int
    query_pos_corrupt: int
    clean_answer_pos: int
    corrupt_answer_pos: int
    clean_pred: bool
    corrupt_pred: bool
    clean_dpd: float
    corrupt_dpd: float
