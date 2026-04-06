from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import utils

from src.eval.io_utils import balanced_sample_by_rule, count_by_field, read_jsonl, write_jsonl
from src.model_loading import add_model_source_arg, resolve_model_prompt
from src.progress import log_event, make_tqdm, resolve_log_path, setup_file_logger
from src.mech.tl_utils import load_hooked_transformer, resolve_true_false_token_ids, to_tokens
from src.plot_style import apply_paper_style, stylize_axis
from src.mlp_analysis.common import (
    build_region_char_spans,
    compute_margin_for_label,
    compute_selected_token_prob,
    find_comma_positions,
    infer_hop_from_rows,
    pick_region_token_indices,
    resolve_prompt,
)
from src.mlp_analysis.result_records import (
    build_sample_detail_record,
    build_skipped_sample_record,
    summarize_sample_layers,
)


LAYER_BANDS = ("early", "middle", "late")
REGION_MODE_ALIASES = {
    "constrain_region": "query_region",
    "terminal_token": "query_region",
}


def _normalize_region_mode(region_mode: str) -> str:
    return REGION_MODE_ALIASES.get(region_mode, region_mode)


def _find_answer_start(str_tokens: Sequence[str]) -> int:
    for idx, tok in enumerate(str_tokens):
        piece = tok.strip()
        if "Answer" in piece or "Reason" in piece:
            return idx
    return len(str_tokens)


def _find_first_idx_with_substr(str_tokens: Sequence[str], needle: str, hi: int) -> int | None:
    for idx, tok in enumerate(str_tokens):
        if idx >= hi:
            break
        if needle in tok:
            return idx
    return None


def _find_prev_nonempty_idx(str_tokens: Sequence[str], hi: int) -> int:
    idx = min(hi - 1, len(str_tokens) - 1)
    while idx >= 0 and not str_tokens[idx].strip():
        idx -= 1
    return max(0, idx)


def _pick_region_indices(
    region_mode: str,
    str_tokens: Sequence[str],
    prompt_order: str = "facts_first",
    hop: str = "one_hop",
) -> List[int]:
    region_mode = _normalize_region_mode(region_mode)
    seq_len = len(str_tokens)
    if seq_len == 0:
        return []
    if seq_len == 1:
        return [0] if region_mode in {"expression_region", "query_region"} else []

    answer_start = _find_answer_start(str_tokens)
    answer_start = max(0, min(answer_start, seq_len))
    terminal_idx = _find_prev_nonempty_idx(str_tokens, answer_start if answer_start > 0 else seq_len)

    facts_start = 0
    facts_end = terminal_idx
    expr_start = 0
    expr_end = terminal_idx
    extra_expr_start = terminal_idx
    extra_expr_end = terminal_idx

    qmark_idx = _find_first_idx_with_substr(str_tokens, "?", hi=answer_start)
    given_idx = _find_first_idx_with_substr(str_tokens, "Given", hi=answer_start)
    evaluate_idx = _find_first_idx_with_substr(str_tokens, "Evaluate", hi=answer_start)

    comma_idxs = [idx for idx in find_comma_positions(str_tokens) if idx < answer_start]
    last_comma = comma_idxs[-1] if comma_idxs else None

    nonterminal_end = terminal_idx
    is_positions = [idx for idx, tok in enumerate(str_tokens[:answer_start]) if tok.strip().lower().startswith("is")]
    query_anchor_idx = None
    if is_positions:
        query_anchor_idx = is_positions[0] if prompt_order == "expr_first" else is_positions[-1]

    if prompt_order == "expr_first":
        if qmark_idx is not None:
            expr_start = 0
            expr_end = min(qmark_idx + 1, nonterminal_end)
            facts_start = expr_end
            facts_end = nonterminal_end
        elif given_idx is not None:
            expr_start = 0
            expr_end = given_idx
            facts_start = given_idx
            facts_end = nonterminal_end
        else:
            split = max(0, min(nonterminal_end, seq_len // 2))
            expr_start = 0
            expr_end = split
            facts_start = split
            facts_end = nonterminal_end
        if hop == "two_hop" and last_comma is not None:
            facts_end = min(last_comma + 1, nonterminal_end)
            extra_expr_start = facts_end
            extra_expr_end = nonterminal_end
    else:
        if evaluate_idx is not None:
            facts_start = 0
            facts_end = evaluate_idx
            expr_start = evaluate_idx
            expr_end = nonterminal_end
        elif last_comma is not None:
            facts_start = 0
            split_comma = last_comma
            if hop == "two_hop" and len(comma_idxs) >= 2:
                split_comma = comma_idxs[-2]
            facts_end = min(split_comma + 1, nonterminal_end)
            expr_start = facts_end
            expr_end = nonterminal_end
        else:
            split = max(0, min(nonterminal_end, seq_len // 2))
            facts_start = 0
            facts_end = split
            expr_start = split
            expr_end = nonterminal_end

    if region_mode == "query_region":
        out: List[int] = []
        if query_anchor_idx is not None:
            if prompt_order == "expr_first":
                out.append(query_anchor_idx)
            else:
                out.extend(range(query_anchor_idx, seq_len))
        if prompt_order == "expr_first":
            out.extend(range(answer_start, seq_len))
        return sorted(set(idx for idx in out if 0 <= idx < seq_len))
    if region_mode == "facts_region":
        return list(range(max(0, facts_start), max(0, facts_end)))
    if region_mode == "expression_region":
        expr_start = max(0, expr_start)
        expr_end = max(expr_start, nonterminal_end)
        out = list(range(expr_start, expr_end))
        if extra_expr_end > extra_expr_start:
            out.extend(range(max(0, extra_expr_start), max(0, extra_expr_end)))
        return out

    raise ValueError(f"Unknown region_mode {region_mode!r}")


def _prompt_body_start_token_idx(
    *,
    model,
    model_input_prompt: str,
    raw_prompt: str,
    tokens,
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


def _pick_region_indices_precise(
    *,
    model,
    raw_prompt: str,
    model_input_prompt: str,
    row: dict,
    prompt_style: str,
    region_mode: str,
    model_input_seq_len: int,
) -> List[int]:
    region_mode = _normalize_region_mode(region_mode)
    if model_input_seq_len <= 0:
        return []

    spans_by_mode = build_region_char_spans(prompt=raw_prompt, row=row, prompt_style=prompt_style, kind="clean")
    if region_mode not in spans_by_mode:
        raise ValueError(f"Unknown region_mode {region_mode!r}")

    raw_char_start = model_input_prompt.find(raw_prompt)
    if raw_char_start < 0:
        raise ValueError("Unable to locate raw prompt inside model input prompt")

    shifted_spans = [(start + raw_char_start, end + raw_char_start) for start, end in spans_by_mode[region_mode]]
    if region_mode == "query_region":
        raw_prompt_end = raw_char_start + len(raw_prompt)
        if raw_prompt_end < len(model_input_prompt):
            shifted_spans.append((raw_prompt_end, len(model_input_prompt)))
    region_indices = pick_region_token_indices(
        tokenizer=getattr(model, "tokenizer", None),
        prompt=model_input_prompt,
        spans=shifted_spans,
        expected_seq_len=model_input_seq_len,
    )
    if region_mode in {"expression_region", "facts_region"}:
        return [idx for idx in region_indices if idx < model_input_seq_len - 1]
    return region_indices


def _layer_band_indices(n_layers: int) -> Dict[str, List[int]]:
    parts = np.array_split(np.arange(n_layers, dtype=np.int64), len(LAYER_BANDS))
    return {name: [int(x) for x in part.tolist()] for name, part in zip(LAYER_BANDS, parts)}


def _layer_band_assignment(n_layers: int) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for band_name, layers in _layer_band_indices(n_layers).items():
        for layer in layers:
            mapping[int(layer)] = band_name
    return mapping


def _compute_band_metrics(scores: Sequence[float], signed_scores: Sequence[float]) -> Dict[str, object]:
    score_arr = np.asarray(scores, dtype=np.float64)
    signed_arr = np.asarray(signed_scores, dtype=np.float64)
    band_layers = _layer_band_indices(len(score_arr))
    total_abs = float(score_arr.sum())

    payload: Dict[str, object] = {
        "layer_bands": {},
        "BMI": {},
        "BCR": {},
        "SBI": {},
    }

    for band_name, layers in band_layers.items():
        layer_meta = {
            "layers": [int(x) for x in layers],
            "start_layer": int(layers[0]) if layers else None,
            "end_layer": int(layers[-1]) if layers else None,
        }
        cast_layer_bands = payload["layer_bands"]
        assert isinstance(cast_layer_bands, dict)
        cast_layer_bands[band_name] = layer_meta

        if layers:
            band_scores = score_arr[layers]
            band_signed = signed_arr[layers]
            bmi = float(band_scores.mean())
            bcr = float(band_scores.sum() / total_abs) if total_abs > 0 else 0.0
            sbi = float(band_signed.mean())
        else:
            bmi = 0.0
            bcr = 0.0
            sbi = 0.0

        cast_bmi = payload["BMI"]
        cast_bcr = payload["BCR"]
        cast_sbi = payload["SBI"]
        assert isinstance(cast_bmi, dict) and isinstance(cast_bcr, dict) and isinstance(cast_sbi, dict)
        cast_bmi[band_name] = bmi
        cast_bcr[band_name] = bcr
        cast_sbi[band_name] = sbi

    return payload


def _plot_bar(output_png: Path, layers: List[int], scores: List[float]) -> None:
    apply_paper_style(
        {
            "font.size": 14.5,
            "axes.titlesize": 16.5,
            "axes.labelsize": 15.0,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
        }
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.bar(layers, scores, color="#4C78A8", edgecolor="#1F1F1F", linewidth=0.9)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("dPD")
    if layers:
        ax.set_xticks(np.arange(0, max(layers) + 1, 5))
    ax.grid(axis="y", alpha=0.28)
    stylize_axis(ax)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320)
    plt.close(fig)


def _attn_out_hook_name(layer: int) -> str:
    try:
        return str(utils.get_act_name("attn_out", layer))
    except Exception:
        return f"blocks.{layer}.hook_attn_out"


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention region ablation with layout-aware region parsing")
    parser.add_argument("--model_id", required=True)
    add_model_source_arg(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_style", choices=["symbolic", "semi_natural"], default="symbolic")
    parser.add_argument(
        "--region_mode",
        choices=["facts_region", "expression_region", "query_region", "constrain_region", "terminal_token"],
        required=True,
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Balanced per-rule max samples per run; 0 means all.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument(
        "--save_plots",
        "--save-plots",
        dest="save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_detailed_results",
        "--save-detailed-results",
        dest="save_detailed_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-sample region metadata, skipped rows, and layer-level patching traces.",
    )
    args = parser.parse_args()
    args.region_mode = _normalize_region_mode(args.region_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_dir, filename="attn_region_ablation.log"))

    rows = read_jsonl(args.input)
    if args.max_samples > 0:
        rows = balanced_sample_by_rule(rows, max_samples=args.max_samples)
    if len(rows) == 0:
        raise ValueError("No rows available for attention region ablation")

    model = load_hooked_transformer(
        args.model_id,
        device=args.device,
        source=args.model_source,
        error_context="mechanistic patching modules",
    )
    true_id, false_id = resolve_true_false_token_ids(model)
    n_layers = model.cfg.n_layers
    layer_bands = _layer_band_assignment(n_layers)

    scores_acc = torch.zeros(n_layers, dtype=torch.float32)
    signed_scores_acc = torch.zeros(n_layers, dtype=torch.float32)
    dpd_acc = torch.zeros(n_layers, dtype=torch.float32)
    valid_rows = 0
    valid_rule_items: List[dict] = []
    sample_details: List[dict] = []
    skipped_rows: List[dict] = []
    region_selection_strategy_counts: Counter[str] = Counter()
    skipped_stage_counts: Counter[str] = Counter()

    layer_scores_path = args.output_dir / f"attn_{args.region_mode}_sample_layer_scores.csv"
    sample_details_path = args.output_dir / f"attn_{args.region_mode}_sample_details.jsonl"
    skipped_samples_path = args.output_dir / f"attn_{args.region_mode}_skipped_samples.jsonl"

    layer_scores_file = None
    layer_scores_writer = None
    if args.save_detailed_results:
        layer_scores_file = layer_scores_path.open("w", encoding="utf-8", newline="")
        layer_scores_writer = csv.DictWriter(
            layer_scores_file,
            fieldnames=[
                "sample_index",
                "row_id",
                "rule",
                "hop",
                "prompt_order",
                "label",
                "region_mode",
                "region_selection_strategy",
                "layer",
                "layer_band",
                "region_size",
                "model_input_seq_len",
                "base_margin",
                "base_selected_token_prob",
                "patched_margin",
                "dpd_shift",
                "dpd",
            ],
        )
        layer_scores_writer.writeheader()

    try:
        with torch.no_grad():
            progress = make_tqdm(rows, total=len(rows), desc=f"attn-{args.region_mode}", leave=True, disable=len(rows) <= 1)
            for idx, row in enumerate(progress, start=1):
                prompt = resolve_prompt(row, prompt_style=args.prompt_style, kind="clean")
                model_input_prompt = resolve_model_prompt(model, prompt, enable_thinking=False)
                tokens = to_tokens(model, model_input_prompt)
                str_tokens = model.to_str_tokens(tokens[0])
                prompt_order = str(row.get("prompt_order", "facts_first"))
                model_input_seq_len = int(tokens.shape[1])

                region_selection_strategy = "precise"
                region_selection_error = None
                try:
                    region_indices = _pick_region_indices_precise(
                        model=model,
                        raw_prompt=prompt,
                        model_input_prompt=model_input_prompt,
                        row=row,
                        prompt_style=args.prompt_style,
                        region_mode=args.region_mode,
                        model_input_seq_len=model_input_seq_len,
                    )
                except Exception as exc:
                    region_selection_strategy = "fallback_after_precise_error"
                    region_selection_error = f"{type(exc).__name__}: {exc}"
                    raw_tokens = to_tokens(model, prompt, prepend_bos=False)
                    raw_str_tokens = model.to_str_tokens(raw_tokens[0])
                    body_start = _prompt_body_start_token_idx(
                        model=model,
                        model_input_prompt=model_input_prompt,
                        raw_prompt=prompt,
                        tokens=tokens,
                    )
                    raw_region_indices = _pick_region_indices(
                        args.region_mode,
                        raw_str_tokens,
                        prompt_order=prompt_order,
                        hop=str(row.get("hop", "one_hop")),
                    )
                    region_indices = [
                        body_start + raw_idx
                        for raw_idx in raw_region_indices
                        if body_start + raw_idx < model_input_seq_len
                    ]
                    if args.region_mode == "query_region":
                        raw_prompt_end_idx = body_start + int(raw_tokens.shape[1])
                        region_indices.extend(range(raw_prompt_end_idx, model_input_seq_len))
                    region_indices = sorted(set(region_indices))

                if len(region_indices) == 0:
                    skipped_stage_counts["empty_region"] += 1
                    skipped_rows.append(
                        build_skipped_sample_record(
                            sample_index=idx,
                            row=row,
                            region_mode=args.region_mode,
                            stage="empty_region",
                            reason="Region index selection returned no tokens.",
                            region_selection_strategy=region_selection_strategy,
                            region_selection_error=region_selection_error,
                            prompt=prompt,
                            model_input_prompt=model_input_prompt,
                            model_input_seq_len=model_input_seq_len,
                            region_indices=region_indices,
                        )
                    )
                    continue

                valid_rows += 1
                valid_rule_items.append(dict(row))
                region_selection_strategy_counts[region_selection_strategy] += 1

                label = bool(row["label"])
                base_logits = model(tokens)
                base_margin = compute_margin_for_label(
                    base_logits,
                    label=label,
                    true_token_id=true_id,
                    false_token_id=false_id,
                )
                base_selected_token_prob = compute_selected_token_prob(
                    base_logits,
                    true_token_id=true_id,
                    false_token_id=false_id,
                )
                base_val = float(base_margin.item())
                base_selected_prob_val = float(base_selected_token_prob.item())
                region_size = len(region_indices)

                sample_layer_rows: List[Dict[str, object]] = []
                for layer in range(n_layers):
                    act_name = _attn_out_hook_name(layer)

                    def hook_fn(attn_out, hook, region_idxs=tuple(region_indices)):
                        del hook
                        mean_vec = attn_out.mean(dim=1, keepdim=True)
                        attn_out[:, region_idxs, :] = mean_vec
                        return attn_out

                    patched_logits = model.run_with_hooks(tokens, fwd_hooks=[(act_name, hook_fn)])
                    patched_margin = compute_margin_for_label(
                        patched_logits,
                        label=label,
                        true_token_id=true_id,
                        false_token_id=false_id,
                    )
                    patched_val = float(patched_margin.item())

                    dpd_shift = patched_val - base_val
                    signed_score = dpd_shift / max(base_selected_prob_val, args.eps)
                    score = abs(signed_score)
                    scores_acc[layer] += score
                    signed_scores_acc[layer] += signed_score
                    dpd_acc[layer] += dpd_shift

                    if layer_scores_writer is not None:
                        layer_row = {
                            "sample_index": idx,
                            "row_id": row.get("id"),
                            "rule": str(row.get("rule", "")),
                            "hop": str(row.get("hop", "")),
                            "prompt_order": prompt_order,
                            "label": label,
                            "region_mode": args.region_mode,
                            "region_selection_strategy": region_selection_strategy,
                            "layer": layer,
                            "layer_band": layer_bands.get(layer, "unknown"),
                            "region_size": region_size,
                            "model_input_seq_len": model_input_seq_len,
                            "base_margin": base_val,
                            "base_selected_token_prob": base_selected_prob_val,
                            "patched_margin": patched_val,
                            "dpd_shift": dpd_shift,
                            "dpd": dpd_shift,
                        }
                        layer_scores_writer.writerow(layer_row)
                        sample_layer_rows.append(layer_row)

                if args.save_detailed_results:
                    region_tokens = [str(str_tokens[i]) for i in region_indices if 0 <= i < len(str_tokens)]
                    sample_details.append(
                        build_sample_detail_record(
                            sample_index=idx,
                            row=row,
                            region_mode=args.region_mode,
                            region_selection_strategy=region_selection_strategy,
                            region_selection_error=region_selection_error,
                            prompt=prompt,
                            model_input_prompt=model_input_prompt,
                            model_input_seq_len=model_input_seq_len,
                            region_indices=region_indices,
                            region_tokens=region_tokens,
                            last_input_token=str(str_tokens[-1]) if str_tokens else "",
                            base_margin=base_val,
                            base_selected_token_prob=base_selected_prob_val,
                            sample_metrics=summarize_sample_layers(sample_layer_rows),
                        )
                    )

                if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(rows)):
                    progress.set_postfix(done=idx, valid=valid_rows)
                    log_event(logger, {"stage": "attn_region_progress", "region_mode": args.region_mode, "done": idx, "total": len(rows)})
    finally:
        if layer_scores_file is not None:
            layer_scores_file.close()

    if valid_rows == 0:
        raise ValueError(f"No valid rows with non-empty region indices for region_mode={args.region_mode}")

    scores = (scores_acc / valid_rows).tolist()
    signed_scores = (signed_scores_acc / valid_rows).tolist()
    dpds = (dpd_acc / valid_rows).tolist()
    band_metrics = _compute_band_metrics(scores=scores, signed_scores=signed_scores)

    csv_path = args.output_dir / f"attn_{args.region_mode}_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "layer_band",
                "dpd",
            ],
        )
        writer.writeheader()
        for layer in range(n_layers):
            writer.writerow(
                {
                    "layer": layer,
                    "layer_band": layer_bands.get(layer, "unknown"),
                    "dpd": dpds[layer],
                }
            )

    summary = {
        "model_id": args.model_id,
        "input": str(args.input),
        "hop": infer_hop_from_rows(rows),
        "prompt_style": args.prompt_style,
        "rows": len(rows),
        "sampled_rule_counts": count_by_field(rows, field="rule"),
        "valid_rows": int(valid_rows),
        "valid_rule_counts": count_by_field(valid_rule_items, field="rule"),
        "skipped_rows": int(len(skipped_rows)),
        "region_selection_strategy_counts": dict(sorted(region_selection_strategy_counts.items())),
        "skipped_stage_counts": dict(sorted(skipped_stage_counts.items())),
        "region_mode": args.region_mode,
        "hook_component": "attn_out",
        "ablation": "mean over token dimension within each sample",
        "n_layers": n_layers,
        "peak_layer": int(np.argmax(dpds)),
        "peak_score": float(np.max(dpds)),
        "mean_score": float(np.mean(dpds)),
        "peak_abs_score": float(np.max(np.abs(np.asarray(dpds, dtype=np.float64)))),
        "mean_abs_score": float(np.mean(np.abs(np.asarray(dpds, dtype=np.float64)))),
        "max_negative_score_layer": int(np.argmin(dpds)),
        "max_negative_score": float(np.min(dpds)),
        "peak_raw_score": float(np.max(np.abs(np.asarray(dpds, dtype=np.float64)))),
        "mean_raw_score": float(np.mean(np.abs(np.asarray(dpds, dtype=np.float64)))),
        "max_positive_signed_layer": int(np.argmax(dpds)),
        "max_positive_signed_score": float(np.max(dpds)),
        "max_negative_signed_layer": int(np.argmin(dpds)),
        "max_negative_signed_score": float(np.min(dpds)),
        "mean_signed_score": float(np.mean(dpds)),
        "max_positive_raw_signed_layer": int(np.argmax(dpds)),
        "max_positive_raw_signed_score": float(np.max(dpds)),
        "max_negative_raw_signed_layer": int(np.argmin(dpds)),
        "max_negative_raw_signed_score": float(np.min(dpds)),
        "mean_raw_signed_score": float(np.mean(dpds)),
        "score_definition": "patching_score = dPD = patched_margin - base_margin",
        "band_metrics_definition": "BMI/BCR/SBI are kept unchanged from the previous normalized-score pipeline.",
        "csv": str(csv_path),
        "band_metrics": band_metrics,
        "plot_png": str(args.output_dir / f"attn_{args.region_mode}_scores.png"),
        "plot_generated": bool(args.save_plots),
        "save_detailed_results": bool(args.save_detailed_results),
    }

    if args.save_detailed_results:
        write_jsonl(sample_details_path, sample_details)
        write_jsonl(skipped_samples_path, skipped_rows)
        summary["sample_details_jsonl"] = str(sample_details_path)
        summary["sample_layer_scores_csv"] = str(layer_scores_path)
        summary["skipped_samples_jsonl"] = str(skipped_samples_path)

    band_metrics_path = args.output_dir / f"attn_{args.region_mode}_band_metrics.json"
    band_metrics_payload = {
        "model_id": args.model_id,
        "input": str(args.input),
        "hop": infer_hop_from_rows(rows),
        "prompt_style": args.prompt_style,
        "region_mode": args.region_mode,
        "hook_component": "attn_out",
        "rows": len(rows),
        "sampled_rule_counts": count_by_field(rows, field="rule"),
        "valid_rows": int(valid_rows),
        "valid_rule_counts": count_by_field(valid_rule_items, field="rule"),
        "n_layers": n_layers,
        **band_metrics,
    }
    band_metrics_path.write_text(
        json.dumps(band_metrics_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["band_metrics_json"] = str(band_metrics_path)

    summary_path = args.output_dir / f"attn_{args.region_mode}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    plot_path = args.output_dir / f"attn_{args.region_mode}_scores.png"
    if args.save_plots:
        _plot_bar(
            output_png=plot_path,
            layers=list(range(n_layers)),
            scores=scores,
        )

    log_event(
        logger,
        {
            "output_dir": str(args.output_dir),
            "summary": str(summary_path),
            "band_metrics": str(band_metrics_path),
            "plot": str(plot_path),
            "plot_generated": bool(args.save_plots),
            "save_detailed_results": bool(args.save_detailed_results),
            "sample_details": str(sample_details_path) if args.save_detailed_results else None,
            "sample_layer_scores": str(layer_scores_path) if args.save_detailed_results else None,
            "skipped_samples": str(skipped_samples_path) if args.save_detailed_results else None,
        },
    )


if __name__ == "__main__":
    main()
