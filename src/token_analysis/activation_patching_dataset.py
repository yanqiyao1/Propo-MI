from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import utils

from src.data.formatters import build_depth_prompt, resolve_prompt_ending, resolve_query_expr_text
from src.eval.io_utils import read_jsonl
from src.model_loading import add_model_source_arg, resolve_model_prompt
from src.progress import log_event, make_tqdm, resolve_log_path, setup_file_logger
from src.mech.dld import compute_prob_diff
from src.mech.tl_utils import load_hooked_transformer, resolve_true_false_token_ids, to_tokens
from src.plot_style import apply_paper_style
from src.token_analysis.refined_token_classifier import classify_tokens_refined
from src.token_analysis.result_schema import get_delta_score_matrix


SIMPLE_CATEGORY_ORDER = [
    "facts_value",
    "query_token",
    "expr_last",
    "expression_token",
    "others",
]

SIMPLE_CATEGORY_DISPLAY_NAMES = {
    "facts_value": "Facts Value",
    "query_token": "Query Token",
    "expr_last": "Expr Last",
    "expression_token": "Expression Token",
    "others": "Others",
}

SIMPLE_CATEGORY_DESCRIPTIONS = {
    "facts_value": "Truth-value tokens in the facts block, such as True and False.",
    "query_token": "Query-region tokens, including the query 'is' anchor, answer suffix tokens, the final terminal token, and any chat-template tail tokens after the raw prompt.",
    "expr_last": "The last non-punctuation token inside the queried expression span.",
    "expression_token": "Remaining non-punctuation tokens in the queried expression span.",
    "others": "All remaining tokens, including non-terminal punctuation.",
}

SIMPLE_COLORS = {
    "facts_value": "#E74C3C",
    "query_token": "#3498DB",
    "expr_last": "#27AE60",
    "expression_token": "#8E44AD",
    "others": "#95A5A6",
}
EXPR_FIRST_EXCLUDED_SIMPLE_CATEGORIES = {"expr_last", "expression_token"}


def _bool_from_any(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1"}:
            return True
        if s in {"false", "f", "0"}:
            return False
    raise ValueError(f"Unable to parse bool from {x!r}")


def _resolve_prompt(row: Dict[str, object], prompt_style: str, kind: str) -> str:
    field = f"{kind}_prompt_{prompt_style}"
    val = row.get(field)
    if val is not None:
        return str(val)

    hop = str(row["hop"])
    if kind == "clean":
        facts = row.get("facts")
    else:
        facts = row.get("corrupted_facts")

    if not isinstance(facts, dict):
        raise KeyError(
            f"Cannot rebuild {kind} prompt for id={row.get('id')}: facts field missing"
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


def _detect_prompt_order(row: Dict[str, object], prompt_text: str) -> str:
    explicit = row.get("prompt_order")
    if explicit in {"facts_first", "expr_first"}:
        return str(explicit)
    if " is? " in prompt_text:
        return "expr_first"
    return "facts_first"


def _prompt_body_start_token_idx(
    *,
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
    if len(token_ids) >= len(input_id_list) and token_ids[-len(input_id_list) :] == input_id_list:
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


def _contiguous_body_token_positions(
    *,
    model,
    model_input_prompt: str,
    raw_prompt: str,
    model_tokens: torch.Tensor,
    raw_token_count: int,
) -> List[List[int]]:
    body_start = _prompt_body_start_token_idx(
        model=model,
        model_input_prompt=model_input_prompt,
        raw_prompt=raw_prompt,
        tokens=model_tokens,
    )
    body_end = body_start + raw_token_count
    if body_end > int(model_tokens.shape[1]):
        raise ValueError(
            f"Prompt body does not fit inside model input: start={body_start}, raw_tokens={raw_token_count}, model_len={int(model_tokens.shape[1])}"
        )
    return [[body_start + idx] for idx in range(raw_token_count)]


def _map_raw_token_positions_to_model_input(
    *,
    model,
    raw_prompt: str,
    model_input_prompt: str,
    model_tokens: torch.Tensor,
    raw_tokens: torch.Tensor,
) -> List[List[int]]:
    raw_token_count = int(raw_tokens.shape[1])
    if raw_token_count <= 0:
        return []

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or model_input_prompt == raw_prompt:
        return _contiguous_body_token_positions(
            model=model,
            model_input_prompt=model_input_prompt,
            raw_prompt=raw_prompt,
            model_tokens=model_tokens,
            raw_token_count=raw_token_count,
        )

    raw_char_start = model_input_prompt.find(raw_prompt)
    if raw_char_start < 0:
        return _contiguous_body_token_positions(
            model=model,
            model_input_prompt=model_input_prompt,
            raw_prompt=raw_prompt,
            model_tokens=model_tokens,
            raw_token_count=raw_token_count,
        )

    try:
        raw_encoded = tokenizer(raw_prompt, add_special_tokens=False, return_offsets_mapping=True)
        model_encoded = tokenizer(model_input_prompt, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, ValueError, NotImplementedError):
        return _contiguous_body_token_positions(
            model=model,
            model_input_prompt=model_input_prompt,
            raw_prompt=raw_prompt,
            model_tokens=model_tokens,
            raw_token_count=raw_token_count,
        )

    raw_offsets = raw_encoded.get("offset_mapping")
    model_offsets = model_encoded.get("offset_mapping")
    model_input_ids = model_encoded.get("input_ids")
    if raw_offsets is None or model_offsets is None or model_input_ids is None:
        return _contiguous_body_token_positions(
            model=model,
            model_input_prompt=model_input_prompt,
            raw_prompt=raw_prompt,
            model_tokens=model_tokens,
            raw_token_count=raw_token_count,
        )

    raw_offset_list = list(raw_offsets)
    model_offset_list = list(model_offsets)
    model_input_id_list = list(model_input_ids)
    if len(raw_offset_list) != raw_token_count:
        return _contiguous_body_token_positions(
            model=model,
            model_input_prompt=model_input_prompt,
            raw_prompt=raw_prompt,
            model_tokens=model_tokens,
            raw_token_count=raw_token_count,
        )

    token_ids = model_tokens[0].tolist()
    prefix_len = 0
    if len(token_ids) >= len(model_input_id_list) and token_ids[-len(model_input_id_list) :] == model_input_id_list:
        prefix_len = len(token_ids) - len(model_input_id_list)
    else:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if (
            bos_token_id is not None
            and token_ids
            and token_ids[0] == int(bos_token_id)
            and token_ids[1 : 1 + len(model_input_id_list)] == model_input_id_list
        ):
            prefix_len = 1

    mapped_positions: List[List[int]] = []
    search_start = 0
    for span in raw_offset_list:
        if span is None or len(span) != 2:
            return _contiguous_body_token_positions(
                model=model,
                model_input_prompt=model_input_prompt,
                raw_prompt=raw_prompt,
                model_tokens=model_tokens,
                raw_token_count=raw_token_count,
            )
        raw_start, raw_end = int(span[0]), int(span[1])
        if raw_end <= raw_start:
            return _contiguous_body_token_positions(
                model=model,
                model_input_prompt=model_input_prompt,
                raw_prompt=raw_prompt,
                model_tokens=model_tokens,
                raw_token_count=raw_token_count,
            )

        shifted_start = raw_char_start + raw_start
        shifted_end = raw_char_start + raw_end
        current: List[int] = []
        probe_idx = search_start
        while probe_idx < len(model_offset_list):
            model_span = model_offset_list[probe_idx]
            if model_span is None or len(model_span) != 2:
                probe_idx += 1
                continue
            model_start, model_end = int(model_span[0]), int(model_span[1])
            if model_end <= shifted_start:
                probe_idx += 1
                continue
            if model_start >= shifted_end:
                break
            current.append(prefix_len + probe_idx)
            probe_idx += 1

        if not current:
            return _contiguous_body_token_positions(
                model=model,
                model_input_prompt=model_input_prompt,
                raw_prompt=raw_prompt,
                model_tokens=model_tokens,
                raw_token_count=raw_token_count,
            )

        mapped_positions.append(current)
        search_start = probe_idx

    return mapped_positions


def classify_tokens_simple(
    str_tokens: Sequence[object],
    *,
    sample: Optional[Dict[str, object]] = None,
    prompt_order: str = "facts_first",
) -> List[str]:
    sample_payload: Dict[str, object] = dict(sample or {})
    sample_payload.setdefault("prompt_order", prompt_order)
    refined_categories, _ = classify_tokens_refined(
        str_tokens=str_tokens,
        sample=sample_payload,
        include_derived_assignment=True,
        strict=False,
    )

    simple_categories: List[str] = []
    for cat in refined_categories:
        if cat in {"facts_value", "query_token", "expr_last"}:
            simple_categories.append(cat)
        elif cat in {"variable_in_expr", "operator"}:
            simple_categories.append("expression_token")
        else:
            simple_categories.append("others")
    return simple_categories


def _extend_with_model_input_suffix_positions(
    *,
    clean_position_map: Sequence[Sequence[int]],
    corrupted_position_map: Sequence[Sequence[int]],
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    clean_used_until = max((int(pos) for positions in clean_position_map for pos in positions), default=-1) + 1
    corrupted_used_until = max((int(pos) for positions in corrupted_position_map for pos in positions), default=-1) + 1

    clean_suffix_positions = list(range(clean_used_until, int(clean_tokens.shape[1])))
    corrupted_suffix_positions = list(range(corrupted_used_until, int(corrupted_tokens.shape[1])))
    suffix_count = min(len(clean_suffix_positions), len(corrupted_suffix_positions))

    clean_extended = [list(positions) for positions in clean_position_map]
    corrupted_extended = [list(positions) for positions in corrupted_position_map]
    for clean_pos, corrupted_pos in zip(clean_suffix_positions[:suffix_count], corrupted_suffix_positions[:suffix_count]):
        clean_extended.append([int(clean_pos)])
        corrupted_extended.append([int(corrupted_pos)])

    return clean_extended, corrupted_extended, [int(pos) for pos in clean_suffix_positions[:suffix_count]]

def _choose_diff_token_ids(
    *,
    clean_label: bool,
    corrupted_label: bool,
    true_id: int,
    false_id: int,
) -> Tuple[int, int]:
    correct_id = true_id if clean_label else false_id
    if clean_label != corrupted_label:
        incorrect_id = true_id if corrupted_label else false_id
    else:
        incorrect_id = false_id if clean_label else true_id
    return correct_id, incorrect_id


def _infer_prompt_orders_from_results(results: Sequence[Dict[str, object]]) -> set[str]:
    prompt_orders: set[str] = set()
    for row in results:
        sample = row.get("sample", {})
        if isinstance(sample, dict):
            value = str(sample.get("prompt_order", "")).strip()
            if value:
                prompt_orders.add(value)
    return prompt_orders


def excluded_simple_categories_for_results(results: Sequence[Dict[str, object]]) -> set[str]:
    prompt_orders = _infer_prompt_orders_from_results(results)
    if prompt_orders == {"expr_first"}:
        return set(EXPR_FIRST_EXCLUDED_SIMPLE_CATEGORIES)
    return set()


def run_patching_for_sample(
    *,
    model,
    sample: Dict[str, object],
    prompt_style: str,
    true_id: int,
    false_id: int,
    strict_length_match: bool,
) -> Optional[Dict[str, object]]:
    clean_prompt = _resolve_prompt(sample, prompt_style=prompt_style, kind="clean")
    corrupted_prompt = _resolve_prompt(sample, prompt_style=prompt_style, kind="corrupted")

    clean_label = _bool_from_any(sample["label"])
    corrupted_label = _bool_from_any(sample["label_corrupted"])
    correct_id, incorrect_id = _choose_diff_token_ids(
        clean_label=clean_label,
        corrupted_label=corrupted_label,
        true_id=true_id,
        false_id=false_id,
    )

    clean_model_input_prompt = resolve_model_prompt(model, clean_prompt, enable_thinking=False)
    corrupted_model_input_prompt = resolve_model_prompt(model, corrupted_prompt, enable_thinking=False)
    clean_tokens = to_tokens(model, clean_model_input_prompt)
    corrupted_tokens = to_tokens(model, corrupted_model_input_prompt)
    clean_raw_tokens = to_tokens(model, clean_prompt, prepend_bos=False)
    corrupted_raw_tokens = to_tokens(model, corrupted_prompt, prepend_bos=False)

    clean_len = int(clean_raw_tokens.shape[1])
    corrupted_len = int(corrupted_raw_tokens.shape[1])
    if strict_length_match and clean_len != corrupted_len:
        return None

    n_raw_positions = min(clean_len, corrupted_len)
    if n_raw_positions <= 0:
        return None

    clean_position_map = _map_raw_token_positions_to_model_input(
        model=model,
        raw_prompt=clean_prompt,
        model_input_prompt=clean_model_input_prompt,
        model_tokens=clean_tokens,
        raw_tokens=clean_raw_tokens,
    )[:n_raw_positions]
    corrupted_position_map = _map_raw_token_positions_to_model_input(
        model=model,
        raw_prompt=corrupted_prompt,
        model_input_prompt=corrupted_model_input_prompt,
        model_tokens=corrupted_tokens,
        raw_tokens=corrupted_raw_tokens,
    )[:n_raw_positions]

    if len(clean_position_map) != n_raw_positions or len(corrupted_position_map) != n_raw_positions:
        raise ValueError(
            f"token_alignment_mismatch: expected {n_raw_positions} raw positions, got clean={len(clean_position_map)} corrupted={len(corrupted_position_map)}"
        )

    for pos, (clean_positions, corrupted_positions) in enumerate(zip(clean_position_map, corrupted_position_map)):
        if not clean_positions or not corrupted_positions:
            raise ValueError(f"token_alignment_mismatch: empty mapped token span at raw position {pos}")
        if len(clean_positions) != len(corrupted_positions):
            raise ValueError(
                f"token_alignment_mismatch: span width differs at raw position {pos}: clean={len(clean_positions)} corrupted={len(corrupted_positions)}"
            )

    clean_position_map, corrupted_position_map, clean_suffix_positions = _extend_with_model_input_suffix_positions(
        clean_position_map=clean_position_map,
        corrupted_position_map=corrupted_position_map,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
    )
    n_positions = len(clean_position_map)

    raw_str_tokens = list(model.to_str_tokens(clean_raw_tokens[0])[:n_raw_positions])
    clean_model_str_tokens = list(model.to_str_tokens(clean_tokens[0]))
    suffix_str_tokens = [str(clean_model_str_tokens[pos]) for pos in clean_suffix_positions]
    str_tokens = raw_str_tokens + suffix_str_tokens
    prompt_order = _detect_prompt_order(sample, clean_prompt)
    sample_for_classification = dict(sample)
    sample_for_classification["raw_prompt_token_count"] = int(n_raw_positions)
    categories = classify_tokens_simple(str_tokens, sample=sample_for_classification, prompt_order=prompt_order)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

        baseline = compute_prob_diff(corrupted_logits, correct_id, incorrect_id, pos=-1).item()
        target = compute_prob_diff(clean_logits, correct_id, incorrect_id, pos=-1).item()

        n_layers = model.cfg.n_layers
        patching_results = torch.zeros((n_layers, n_positions), device=model.cfg.device)

        for layer in range(n_layers):
            act_name = utils.get_act_name("resid_pre", layer)
            for pos in range(n_positions):
                clean_positions = tuple(clean_position_map[pos])
                corrupted_positions = tuple(corrupted_position_map[pos])

                def patching_hook(
                    resid_pre,
                    hook,
                    clean_positions_=clean_positions,
                    corrupted_positions_=corrupted_positions,
                    layer_name=act_name,
                ):
                    del hook
                    clean_resid = clean_cache[layer_name]
                    resid_pre[:, corrupted_positions_, :] = clean_resid[:, clean_positions_, :]
                    return resid_pre

                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(act_name, patching_hook)],
                )
                patched_score = compute_prob_diff(patched_logits, correct_id, incorrect_id, pos=-1)
                patching_results[layer, pos] = patched_score

    patch_np = patching_results.detach().float().cpu().numpy()
    delta_dpd = patch_np - baseline

    return {
        "patching_results": patch_np,
        "str_tokens": list(str_tokens),
        "categories": categories,
        "baseline": float(baseline),
        "target": float(target),
        "delta_dpd": delta_dpd,
        "clean_len": clean_len,
        "corrupted_len": corrupted_len,
        "n_positions": int(n_positions),
        "suffix_positions": int(len(clean_suffix_positions)),
        "clean_model_input_len": int(clean_tokens.shape[1]),
        "corrupted_model_input_len": int(corrupted_tokens.shape[1]),
        "sample": sample,
    }


def aggregate_by_category_with_sample_avg(
    all_results: Sequence[Dict[str, object]],
    *,
    n_layers: int,
    early_end: int,
    middle_end: int,
    exclude_categories: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    excluded = set(exclude_categories or [])
    layer_ranges = {
        "all": (0, n_layers),
        "early": (0, early_end),
        "middle": (early_end, middle_end),
        "late": (middle_end, n_layers),
    }

    category_sample_avgs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        delta_matrix = get_delta_score_matrix(result)
        if delta_matrix is None:
            continue
        delta_dpd = np.asarray(delta_matrix)
        categories = list(result["categories"])

        category_positions: Dict[str, List[int]] = defaultdict(list)
        for pos, cat in enumerate(categories):
            if cat in excluded:
                continue
            category_positions[cat].append(pos)

        for cat, positions in category_positions.items():
            for stage_name, (start, end) in layer_ranges.items():
                end_clamped = min(end, delta_dpd.shape[0])
                if start >= end_clamped:
                    continue

                cat_dpds = [float(delta_dpd[start:end_clamped, pos].mean()) for pos in positions]
                sample_avg = float(np.mean(cat_dpds))
                category_sample_avgs[cat][f"{stage_name}_dpd"].append(sample_avg)

            category_sample_avgs[cat]["token_count_per_sample"].append(float(len(positions)))

    rows: List[Dict[str, object]] = []
    for cat, metrics in category_sample_avgs.items():
        row: Dict[str, object] = {
            "category": cat,
            "category_label": SIMPLE_CATEGORY_DISPLAY_NAMES.get(cat, cat),
            "category_description": SIMPLE_CATEGORY_DESCRIPTIONS.get(cat, ""),
        }
        token_counts = np.asarray(metrics.get("token_count_per_sample", []), dtype=np.float64)
        row["avg_tokens_per_sample"] = float(token_counts.mean()) if token_counts.size > 0 else 0.0
        row["n_samples"] = int(token_counts.size)

        for metric_name, values in metrics.items():
            if metric_name == "token_count_per_sample":
                continue
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                row[f"{metric_name}_mean"] = 0.0
                row[f"{metric_name}_std"] = 0.0
                row[f"{metric_name}_sem"] = 0.0
                continue
            row[f"{metric_name}_mean"] = float(arr.mean())
            row[f"{metric_name}_std"] = float(arr.std(ddof=0))
            row[f"{metric_name}_sem"] = float(arr.std(ddof=0) / np.sqrt(arr.size))
        rows.append(row)

    rows.sort(key=lambda r: float(r.get("all_dpd_mean", 0.0)), reverse=True)
    return rows


def _write_stats_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_simple_comparison(
    stats_rows: Sequence[Dict[str, object]],
    *,
    output_path: Path,
) -> None:
    if not stats_rows:
        return

    categories = [str(r["category"]) for r in stats_rows]
    category_labels = [SIMPLE_CATEGORY_DISPLAY_NAMES.get(cat, cat) for cat in categories]
    x = np.arange(len(categories))
    colors = [SIMPLE_COLORS.get(cat, "#95A5A6") for cat in categories]
    means = np.asarray([float(r.get("all_dpd_mean", 0.0)) for r in stats_rows], dtype=np.float64)
    sems = np.asarray([float(r.get("all_dpd_sem", 0.0)) for r in stats_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2, capsize=4)
    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_ylabel("dPD (Mean +/- SEM)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=35, ha="right", fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_stage_simple(
    stats_rows: Sequence[Dict[str, object]],
    *,
    output_path: Path,
    early_end: int,
    middle_end: int,
    n_layers: int,
) -> None:
    if not stats_rows:
        return

    categories = [str(r["category"]) for r in stats_rows]
    category_labels = [SIMPLE_CATEGORY_DISPLAY_NAMES.get(cat, cat) for cat in categories]
    x = np.arange(len(categories))
    width = 0.25

    early_vals = np.asarray([float(r.get("early_dpd_mean", 0.0)) for r in stats_rows], dtype=np.float64)
    middle_vals = np.asarray([float(r.get("middle_dpd_mean", 0.0)) for r in stats_rows], dtype=np.float64)
    late_vals = np.asarray([float(r.get("late_dpd_mean", 0.0)) for r in stats_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, early_vals, width, label=f"Early (L0-{early_end - 1})", color="#3498DB", edgecolor="black", alpha=0.85)
    ax.bar(x, middle_vals, width, label=f"Middle (L{early_end}-{middle_end - 1})", color="#9B59B6", edgecolor="black", alpha=0.85)
    ax.bar(x + width, late_vals, width, label=f"Late (L{middle_end}-{n_layers - 1})", color="#E67E22", edgecolor="black", alpha=0.85)
    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.set_ylabel("dPD Mean")
    ax.set_xlabel("Category")
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_simple_heatmap(
    all_results: Sequence[Dict[str, object]],
    *,
    n_layers: int,
    exclude_categories: Sequence[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    excluded = set(exclude_categories or [])
    category_layer_sample_avgs: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        delta_matrix = get_delta_score_matrix(result)
        if delta_matrix is None:
            continue
        delta_dpd = np.asarray(delta_matrix)
        categories = list(result["categories"])

        category_positions: Dict[str, List[int]] = defaultdict(list)
        for pos, cat in enumerate(categories):
            if cat in excluded:
                continue
            category_positions[cat].append(pos)

        for cat, positions in category_positions.items():
            for layer in range(min(n_layers, delta_dpd.shape[0])):
                vals = [float(delta_dpd[layer, pos]) for pos in positions]
                sample_avg = float(np.mean(vals))
                category_layer_sample_avgs[cat][layer].append(sample_avg)

    present = set(category_layer_sample_avgs.keys())
    category_order = [cat for cat in SIMPLE_CATEGORY_ORDER if cat in present]
    extras = sorted(cat for cat in present if cat not in SIMPLE_CATEGORY_ORDER)
    category_order.extend(extras)
    heatmap = np.zeros((len(category_order), n_layers), dtype=np.float64)
    for i, cat in enumerate(category_order):
        for layer in range(n_layers):
            vals = category_layer_sample_avgs[cat].get(layer, [])
            if vals:
                heatmap[i, layer] = float(np.mean(np.asarray(vals, dtype=np.float64)))
    return heatmap, category_order


def _plot_simple_heatmap(
    heatmap_data: np.ndarray,
    category_order: Sequence[str],
    *,
    output_path: Path,
    early_end: int,
    middle_end: int,
) -> None:
    if heatmap_data.size == 0:
        return

    vmax = float(max(abs(float(heatmap_data.min())), abs(float(heatmap_data.max()))))
    if vmax <= 1e-12:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(category_order)))
    ax.set_yticklabels([SIMPLE_CATEGORY_DISPLAY_NAMES.get(cat, cat) for cat in category_order], fontsize=11)
    ax.set_xticks(range(0, heatmap_data.shape[1], 2))
    ax.set_xticklabels(range(0, heatmap_data.shape[1], 2), fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Category")
    ax.axvline(x=early_end - 0.5, color="white", linewidth=1.8)
    ax.axvline(x=middle_end - 0.5, color="white", linewidth=1.8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean dPD")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _read_dataset(path: Path) -> List[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("examples", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise ValueError(f"Unsupported dataset format in {path}")


def _filter_rows(
    rows: Sequence[Dict[str, object]],
    *,
    hop: str,
    max_samples: int,
    require_dual_correct: bool,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        if hop != "all" and str(r.get("hop", "")) != hop:
            continue
        if require_dual_correct:
            if not bool(r.get("correct_clean", False)):
                continue
            if not bool(r.get("correct_corrupted", False)):
                continue
        out.append(dict(r))
    if max_samples > 0:
        out = out[:max_samples]
    return out


def _print_summary(stats_rows: Sequence[Dict[str, object]], n_results: int, n_failed: int, logger) -> None:
    log_event(logger, "=" * 80)
    log_event(logger, "SUMMARY: Simplified Activation Patching Analysis")
    log_event(logger, "=" * 80)
    log_event(logger, f"Samples analyzed: {n_results}")
    log_event(logger, f"Failed samples: {n_failed}")
    log_event(logger, "--- Results ---")
    for row in stats_rows:
        cat = str(row.get("category", "unknown"))
        label = SIMPLE_CATEGORY_DISPLAY_NAMES.get(cat, cat)
        dpd = float(row.get("all_dpd_mean", 0.0))
        sem = float(row.get("all_dpd_sem", 0.0))
        n = int(row.get("n_samples", 0))
        avg_tok = float(row.get("avg_tokens_per_sample", 0.0))
        log_event(
            logger,
            f"{label:15s}: dPD={dpd:+.3f}+/-{sem:.3f}, n={n} samples ({avg_tok:.1f} tokens/sample)",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-level residual activation patching analysis (simple categories) for downstream refined token analysis"
    )
    parser.add_argument("--model_id", type=str, required=True)
    add_model_source_arg(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_style", choices=["symbolic", "semi_natural"], default="symbolic")
    parser.add_argument("--hop", choices=["one_hop", "two_hop", "all"], default="one_hop")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all selected samples")
    parser.add_argument("--require_dual_correct", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict_length_match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_end", type=int, default=14)
    parser.add_argument("--middle_end", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument(
        "--save_plots",
        "--save-plots",
        dest="save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_dir, filename="activation_patching_dataset.log"))

    apply_paper_style(
        {
            "font.size": 12.5,
            "axes.titlesize": 14.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )

    rows_all = _read_dataset(args.input)
    rows = _filter_rows(
        rows_all,
        hop=args.hop,
        max_samples=args.max_samples,
        require_dual_correct=args.require_dual_correct,
    )

    if len(rows) == 0:
        raise ValueError("No rows selected after hop/max_samples filtering")

    log_event(
        logger,
        {
            "stage": "patching_start",
            "model_id": args.model_id,
            "model_source": args.model_source,
            "input": str(args.input),
            "selected_rows": len(rows),
            "hop": args.hop,
            "prompt_style": args.prompt_style,
            "require_dual_correct": args.require_dual_correct,
        },
    )

    model = load_hooked_transformer(
        args.model_id,
        device=args.device,
        source=args.model_source,
        error_context="mechanistic patching modules",
    )
    true_id, false_id = resolve_true_false_token_ids(model)
    n_layers = model.cfg.n_layers

    all_results: List[Dict[str, object]] = []
    failed_samples: List[Dict[str, object]] = []

    progress = make_tqdm(rows, total=len(rows), desc="token-patching", leave=True, disable=len(rows) <= 1)
    for i, sample in enumerate(progress, start=1):
        try:
            result = run_patching_for_sample(
                model=model,
                sample=sample,
                prompt_style=args.prompt_style,
                true_id=true_id,
                false_id=false_id,
                strict_length_match=args.strict_length_match,
            )
            if result is not None:
                all_results.append(result)
            else:
                failed_samples.append({"index": i - 1, "id": sample.get("id"), "error": "token_length_mismatch"})
        except Exception as exc:
            failed_samples.append({"index": i - 1, "id": sample.get("id"), "error": str(exc)})

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(rows)):
            progress.set_postfix(done=i, ok=len(all_results), failed=len(failed_samples))
            log_event(logger, {"stage": "patching_progress", "done": i, "total": len(rows), "ok": len(all_results), "failed": len(failed_samples)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = args.output_dir / "patching_results.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(all_results, f)

    failed_path = args.output_dir / "failed_samples.json"
    failed_path.write_text(json.dumps(failed_samples, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    stats_rows = aggregate_by_category_with_sample_avg(
        all_results,
        n_layers=n_layers,
        early_end=args.early_end,
        middle_end=args.middle_end,
        exclude_categories=excluded_simple_categories_for_results(all_results),
    )
    _write_stats_csv(args.output_dir / "statistics_simple.csv", stats_rows)

    excluded_categories = excluded_simple_categories_for_results(all_results)
    if args.save_plots:
        _plot_simple_comparison(
            stats_rows,
            output_path=args.output_dir / "category_comparison_simple.png",
        )
        _plot_layer_stage_simple(
            stats_rows,
            output_path=args.output_dir / "layer_stage_simple.png",
            early_end=args.early_end,
            middle_end=args.middle_end,
            n_layers=n_layers,
        )
        heatmap_data, category_order = compute_simple_heatmap(
            all_results,
            n_layers=n_layers,
            exclude_categories=excluded_categories,
        )
        _plot_simple_heatmap(
            heatmap_data,
            category_order,
            output_path=args.output_dir / "heatmap_simple.png",
            early_end=args.early_end,
            middle_end=args.middle_end,
        )
    else:
        heatmap_data, category_order = compute_simple_heatmap(
            all_results,
            n_layers=n_layers,
            exclude_categories=excluded_categories,
        )

    summary = {
        "model_id": args.model_id,
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "hop": args.hop,
        "prompt_style": args.prompt_style,
        "selected_rows": len(rows),
        "successful": len(all_results),
        "failed": len(failed_samples),
        "n_layers": n_layers,
        "early_end": args.early_end,
        "middle_end": args.middle_end,
        "analysis_mode": "raw_patching_plus_simple_plots",
        "next_step": "Run src.token_analysis.refined_token_analysis with patching_results.pkl for refined categorization",
        "category_labels": {cat: SIMPLE_CATEGORY_DISPLAY_NAMES.get(cat, cat) for cat in category_order},
        "category_descriptions": {cat: SIMPLE_CATEGORY_DESCRIPTIONS.get(cat, "") for cat in category_order},
        "excluded_categories": sorted(excluded_categories),
        "categories_in_heatmap": list(category_order),
        "category_labels_in_heatmap": [SIMPLE_CATEGORY_DISPLAY_NAMES.get(cat, cat) for cat in category_order],
        "heatmap_shape": list(heatmap_data.shape),
        "stats_rows": len(stats_rows),
        "files": {
            "patching_results_pkl": str(pkl_path),
            "failed_samples_json": str(failed_path),
            "statistics_simple_csv": str(args.output_dir / "statistics_simple.csv"),
            "category_comparison_png": str(args.output_dir / "category_comparison_simple.png"),
            "layer_stage_png": str(args.output_dir / "layer_stage_simple.png"),
            "heatmap_png": str(args.output_dir / "heatmap_simple.png"),
            "summary_json": str(args.output_dir / "summary.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    _print_summary(stats_rows, len(all_results), len(failed_samples), logger)
    log_event(logger, {"stage": "patching_done", "output_dir": str(args.output_dir), "pkl": str(pkl_path)})


if __name__ == "__main__":
    main()
