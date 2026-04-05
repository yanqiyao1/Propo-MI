"""Step 1 (fast) - impact screening + pattern classification.

Complexity reduction strategy:
1) Probe phase: evaluate all heads on a small probe set.
2) Refine phase: evaluate only candidate heads on larger impact set.
3) Pattern classification only on final top heads.
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import utils

from src.eval.io_utils import balanced_sample_by_rule, count_by_field
from src.model_loading import add_model_source_arg
from src.progress import log_event, make_tqdm, resolve_log_path, setup_file_logger

from .common import (
    ROLE_COLOR,
    ROLE_LABEL,
    ROLE_ORDER,
    SamplePair,
    apply_paper_style,
    bool_prediction_from_logits,
    compute_correct_incorrect_prob_diff,
    filter_rows,
    find_comma_positions,
    find_fact_positions,
    infer_prompt_order,
    load_hooked_transformer,
    query_is_pos,
    region_starts_from_commas,
    resolve_dual_correct_default,
    resolve_prompt,
    resolve_role_label,
    resolve_true_false_token_ids,
    safe_mean,
    serialize_head,
    tokenize_prompt_for_eval_alignment,
    tril_mean,
    write_csv,
    write_json,
)


LOGGER = logging.getLogger(__name__)


def _build_sample_pairs_and_means(
    *,
    model,
    rows: Sequence[Dict[str, object]],
    prompt_style: str,
    late_layers: Sequence[int],
    true_id: int,
    false_id: int,
    token_scope: str,
    strict_length_match: bool,
    progress_every: int,
) -> Tuple[List[SamplePair], Dict[int, torch.Tensor], int, int, int]:
    act_names = {utils.get_act_name("z", layer) for layer in late_layers}
    mean_acc: Dict[int, torch.Tensor] = {}
    mean_sample_count = 0
    mean_token_count = 0
    skipped_length = 0
    samples: List[SamplePair] = []

    with torch.no_grad():
        progress = make_tqdm(rows, total=len(rows), desc="step1-build-pairs", leave=False, disable=len(rows) <= 1)
        for idx, row in enumerate(progress, start=1):
            clean_prompt = resolve_prompt(row, prompt_style=prompt_style, kind="clean")
            corrupt_prompt = resolve_prompt(row, prompt_style=prompt_style, kind="corrupted")
            clean_tok = tokenize_prompt_for_eval_alignment(model, clean_prompt)
            corrupt_tok = tokenize_prompt_for_eval_alignment(model, corrupt_prompt)
            clean_tokens = clean_tok.tokens
            corrupt_tokens = corrupt_tok.tokens
            if strict_length_match and clean_tokens.shape[1] != corrupt_tokens.shape[1]:
                skipped_length += 1
                continue

            po_clean = infer_prompt_order(row, clean_prompt)
            po_corrupt = infer_prompt_order(row, corrupt_prompt)
            q_clean = clean_tok.body_start + query_is_pos(clean_tok.raw_str_tokens, po_clean)
            q_corrupt = corrupt_tok.body_start + query_is_pos(corrupt_tok.raw_str_tokens, po_corrupt)

            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: n in act_names)
            corrupt_logits = model(corrupt_tokens)

            lc = bool(row.get("label", False))
            lc_corrupt = bool(row.get("label_corrupted", lc))
            pred_clean, _, _ = bool_prediction_from_logits(clean_logits, true_id, false_id, pos=clean_tok.answer_pos)
            pred_corrupt, _, _ = bool_prediction_from_logits(corrupt_logits, true_id, false_id, pos=corrupt_tok.answer_pos)
            clean_dpd = float(compute_correct_incorrect_prob_diff(clean_logits, lc, true_id, false_id, pos=clean_tok.answer_pos).item())
            corrupt_dpd = float(compute_correct_incorrect_prob_diff(corrupt_logits, lc_corrupt, true_id, false_id, pos=corrupt_tok.answer_pos).item())
            sample = SamplePair(
                sample_id=str(row.get("id", "")),
                hop=str(row.get("hop", "")),
                rule=str(row.get("rule", "")),
                label_clean=lc,
                label_corrupt=lc_corrupt,
                clean_tokens=clean_tokens,
                corrupt_tokens=corrupt_tokens,
                query_pos_clean=q_clean,
                query_pos_corrupt=q_corrupt,
                clean_answer_pos=clean_tok.answer_pos,
                corrupt_answer_pos=corrupt_tok.answer_pos,
                clean_pred=bool(pred_clean.item()),
                corrupt_pred=bool(pred_corrupt.item()),
                clean_dpd=clean_dpd,
                corrupt_dpd=corrupt_dpd,
            )
            samples.append(sample)

            for layer in late_layers:
                layer_z = clean_cache[utils.get_act_name("z", layer)][0]
                if token_scope == "query_only":
                    z = layer_z[q_clean, :, :]
                    token_inc = 1
                else:
                    z = layer_z.sum(dim=0)
                    token_inc = int(layer_z.shape[0])
                if layer not in mean_acc:
                    mean_acc[layer] = torch.zeros_like(z, dtype=torch.float32)
                mean_acc[layer] += z.to(dtype=torch.float32)
            mean_sample_count += 1
            mean_token_count += token_inc

            if progress_every > 0 and (idx % progress_every == 0 or idx == len(rows)):
                progress.set_postfix(done=idx, kept=len(samples), skipped=skipped_length)
                log_event(LOGGER, {"stage": "step1_build_pairs_progress", "done": idx, "total": len(rows), "samples": len(samples), "skipped": skipped_length})

    if mean_sample_count <= 0:
        raise RuntimeError("No valid samples for mean estimation")
    denom = float(mean_sample_count) if token_scope == "query_only" else float(mean_token_count)
    mean_mats = {layer: (acc / denom).detach() for layer, acc in mean_acc.items()}
    return samples, mean_mats, mean_sample_count, mean_token_count, skipped_length


def _evaluate_necessity_for_heads(
    *,
    model,
    samples: Sequence[SamplePair],
    heads: Sequence[Tuple[int, int]],
    mean_mats: Mapping[int, torch.Tensor],
    true_id: int,
    false_id: int,
    token_scope: str,
) -> Dict[str, object]:
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for l, h in heads:
        by_layer[l].append(h)

    dpd_shifts: List[float] = []
    ladpd_shifts: List[float] = []
    patched_correct = 0

    with torch.no_grad():
        for sample in samples:
            hooks = []
            for layer, head_ids in by_layer.items():
                act_name = utils.get_act_name("z", layer)
                mean_layer = mean_mats[layer]

                def hook_fn(z, hook, q_pos=sample.query_pos_clean, head_idx=tuple(head_ids), mean_layer_=mean_layer):
                    del hook
                    mean_block = mean_layer_[list(head_idx), :].to(dtype=z.dtype, device=z.device)
                    if token_scope == "query_only":
                        z[:, q_pos, list(head_idx), :] = mean_block.unsqueeze(0)
                    else:
                        z[:, :, list(head_idx), :] = mean_block.unsqueeze(0).unsqueeze(0)
                    return z

                hooks.append((act_name, hook_fn))

            patched_logits = model.run_with_hooks(sample.clean_tokens, fwd_hooks=hooks)
            pred, _, _ = bool_prediction_from_logits(patched_logits, true_id, false_id, pos=sample.clean_answer_pos)
            patched_dpd = float(
                compute_correct_incorrect_prob_diff(patched_logits, sample.label_clean, true_id, false_id, pos=sample.clean_answer_pos).item()
            )
            raw_shift = patched_dpd - sample.clean_dpd

            dpd_shifts.append(raw_shift)
            ladpd_shifts.append(raw_shift)
            if bool(pred.item()) == sample.label_clean:
                patched_correct += 1

    n = len(samples)
    return {
        "n_samples": n,
        "accuracy": (patched_correct / n) if n else 0.0,
        "mean_dpd_shift": safe_mean(dpd_shifts),
        "mean_label_aligned_dpd_shift": safe_mean(ladpd_shifts),
        "dpd_shifts": dpd_shifts,
    }


def _evaluate_necessity_for_head_sets_batched(
    *,
    model,
    samples: Sequence[SamplePair],
    head_sets: Sequence[Sequence[Tuple[int, int]]],
    mean_mats: Mapping[int, torch.Tensor],
    true_id: int,
    false_id: int,
    token_scope: str,
    eval_batch_size: int,
    progress_every: int,
    progress_tag: str,
) -> List[Dict[str, object]]:
    """Evaluate multiple head-sets in batched conditions to reduce forward passes."""
    if not head_sets:
        return []
    cond_batch = max(1, int(eval_batch_size))

    prepared: List[Dict[int, Tuple[int, ...]]] = []
    for heads in head_sets:
        by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer, head in heads:
            by_layer[int(layer)].append(int(head))
        prepared.append({layer: tuple(sorted(set(head_ids))) for layer, head_ids in by_layer.items()})

    n_cond = len(prepared)
    dpd_shifts: List[List[float]] = [[] for _ in range(n_cond)]
    ladpd_shifts: List[List[float]] = [[] for _ in range(n_cond)]
    patched_correct = [0 for _ in range(n_cond)]

    with torch.no_grad():
        progress = make_tqdm(samples, total=len(samples), desc=progress_tag or "step1-necessity", leave=False, disable=len(samples) <= 1)
        for sample_idx, sample in enumerate(progress, start=1):
            for start in range(0, n_cond, cond_batch):
                end = min(start + cond_batch, n_cond)
                chunk = prepared[start:end]
                bsz = end - start
                toks = sample.clean_tokens.repeat(bsz, 1)

                layer_assign: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)
                for bi, per_layer in enumerate(chunk):
                    for layer, head_ids in per_layer.items():
                        if head_ids:
                            layer_assign[layer].append((bi, head_ids))

                hooks = []
                for layer, assignments in layer_assign.items():
                    act_name = utils.get_act_name("z", layer)
                    mean_layer = mean_mats[layer]

                    def hook_fn(
                        z,
                        hook,
                        q_pos=sample.query_pos_clean,
                        assignments_=tuple(assignments),
                        mean_layer_=mean_layer,
                        token_scope_=token_scope,
                    ):
                        del hook
                        for batch_idx, head_idx in assignments_:
                            idx = list(head_idx)
                            mean_block = mean_layer_[idx, :].to(dtype=z.dtype, device=z.device)
                            if token_scope_ == "query_only":
                                z[batch_idx, q_pos, idx, :] = mean_block
                            else:
                                z[batch_idx, :, idx, :] = mean_block.unsqueeze(0)
                        return z

                    hooks.append((act_name, hook_fn))

                patched_logits = model.run_with_hooks(toks, fwd_hooks=hooks)
                pred, _, _ = bool_prediction_from_logits(patched_logits, true_id, false_id, pos=sample.clean_answer_pos)
                pred_arr = pred.detach().to(dtype=torch.bool).cpu().numpy()
                dpd_arr = (
                    compute_correct_incorrect_prob_diff(patched_logits, sample.label_clean, true_id, false_id, pos=sample.clean_answer_pos)
                    .detach()
                    .cpu()
                    .numpy()
                )

                for bi in range(bsz):
                    ci = start + bi
                    raw_shift = float(dpd_arr[bi]) - sample.clean_dpd
                    dpd_shifts[ci].append(raw_shift)
                    ladpd_shifts[ci].append(raw_shift)
                    if bool(pred_arr[bi]) == sample.label_clean:
                        patched_correct[ci] += 1

            if progress_every > 0 and (sample_idx % progress_every == 0 or sample_idx == len(samples)):
                progress.set_postfix(sample=sample_idx, conditions=n_cond)
                log_event(LOGGER, {"stage": "step1_eval_progress", "tag": progress_tag, "sample": sample_idx, "total": len(samples), "conditions": n_cond, "batch": cond_batch})

    n = len(samples)
    out: List[Dict[str, object]] = []
    for ci in range(n_cond):
        out.append(
            {
                "n_samples": n,
                "accuracy": (patched_correct[ci] / n) if n else 0.0,
                "mean_dpd_shift": safe_mean(dpd_shifts[ci]),
                "mean_label_aligned_dpd_shift": safe_mean(ladpd_shifts[ci]),
                "dpd_shifts": dpd_shifts[ci],
            }
        )
    return out


def _zscore_matrix(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (x - mu) / sd


def _score_roles(metrics: Mapping[str, float]) -> Dict[str, float]:
    fact = metrics.get("fact_focus_ratio", 0.0)
    boundary = metrics.get("boundary_focus_ratio", 0.0)
    intra = metrics.get("intra_clause_ratio", 0.0)
    ent = metrics.get("attn_entropy", 0.0)
    top1 = metrics.get("top1_mass", 0.0)
    stab = metrics.get("stability_score", 0.0)
    return {
        "fact_retrieval": fact - 0.20 * intra - 0.15 * boundary + 0.20 * top1 - 0.10 * ent + 0.08 * stab,
        "splitting": boundary - 0.15 * intra - 0.10 * fact + 0.15 * top1 - 0.08 * ent + 0.10 * stab,
        "transmission": intra - 0.20 * boundary - 0.10 * fact + 0.18 * top1 - 0.08 * ent + 0.12 * stab,
    }


def _label_by_rules(metrics: Mapping[str, float], *, margin: float, min_score: float) -> Tuple[str, str, Dict[str, float]]:
    score_map = _score_roles(metrics)
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top_role, top_score = ranked[0]
    second_score = ranked[1][1]
    if top_score >= min_score and (top_score - second_score) >= margin:
        return top_role, "rule", score_map
    return top_role, "score_argmax_fallback", score_map


def _fact_focus(attn: torch.Tensor, query_pos: int, fact_pos: Sequence[int]) -> float:
    if not fact_pos or query_pos < 0 or query_pos >= attn.shape[0]:
        return 0.0
    row = attn[query_pos]
    valid = [p for p in fact_pos if 0 <= p < attn.shape[1]]
    return float((row[valid].mean() / (row.mean() + 1e-12)).item()) if valid else 0.0


def _boundary_focus(attn: torch.Tensor, commas: Sequence[int]) -> float:
    if not commas:
        return 0.0
    g = tril_mean(attn)
    n = attn.shape[0]
    vals = [float((attn[c:, c].mean() / (g + 1e-12)).item()) for c in commas if 0 <= c < n]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _intra_clause(attn: torch.Tensor, starts: Sequence[int], region_size: int = 3) -> float:
    if not starts:
        return 0.0
    n = attn.shape[0]
    g = tril_mean(attn)
    tri_i, tri_j = torch.tril_indices(row=region_size, col=region_size, device=attn.device)
    vals = []
    for rs in starts:
        if rs < 0 or rs + region_size > n:
            continue
        block = attn[rs : rs + region_size, rs : rs + region_size]
        vals.append(float((block[tri_i, tri_j].mean() / (g + 1e-12)).item()))
    return float(sum(vals) / len(vals)) if vals else 0.0


def _entropy_and_top1(attn: torch.Tensor, query_pos: int) -> Tuple[float, float]:
    if query_pos < 0 or query_pos >= attn.shape[0]:
        return 0.0, 0.0
    row = attn[query_pos]
    p = row / (row.sum() + 1e-12)
    return float((-(p * (p + 1e-12).log()).sum()).item()), float(p.max().item())


def _plot_layer_head_distribution(
    rows: List[Mapping[str, object]],
    output_path: Path,
    n_layers: int,
    n_heads: int,
    role_col: str = "role_label",
) -> None:
    pts_by_role: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"x": [], "y": [], "s": []})
    layer_role_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        layer = int(float(row.get("layer", 0)))
        head = int(float(row.get("head", 0)))
        role = resolve_role_label(row, role_col=role_col)
        if role is None:
            continue
        impact = abs(float(row.get("mean_abs_dpd_shift", 0.0)))
        pts_by_role[role]["x"].append(float(head))
        pts_by_role[role]["y"].append(float(layer))
        pts_by_role[role]["s"].append(35.0 + 300.0 * impact)
        layer_role_counts[layer][role] += 1

    apply_paper_style({"axes.grid": True, "grid.alpha": 0.20})
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(12.2, 6.1), gridspec_kw={"width_ratios": [1.45, 1.0]})

    gx, gy = np.meshgrid(np.arange(n_heads), np.arange(n_layers))
    ax_map.scatter(gx.ravel(), gy.ravel(), s=7, c="#DCDCDC", marker="s", alpha=0.35, linewidths=0)

    for role in ROLE_ORDER:
        xs, ys, ss = pts_by_role[role]["x"], pts_by_role[role]["y"], pts_by_role[role]["s"]
        if not xs:
            continue
        ax_map.scatter(
            xs,
            ys,
            s=ss,
            c=ROLE_COLOR[role],
            edgecolors="#1F1F1F",
            linewidths=0.45,
            alpha=0.92,
            marker="s",
            label=f"{ROLE_LABEL[role]} (n={len(xs)})",
            zorder=3,
        )

    ax_map.set_xlim(-0.6, n_heads - 0.4)
    ax_map.set_ylim(-0.6, n_layers - 0.4)
    ax_map.set_xlabel("Head Index")
    ax_map.set_ylabel("Layer")
    ax_map.set_xticks(np.arange(0, n_heads, max(1, n_heads // 8)))
    ax_map.set_yticks(np.arange(0, n_layers, max(1, n_layers // 10)))
    ax_map.legend(loc="upper left", frameon=True)

    layers_arr = np.arange(n_layers)
    bottom = np.zeros(n_layers, dtype=np.float64)
    for role in ROLE_ORDER:
        vals = np.asarray([layer_role_counts[int(l)].get(role, 0) for l in layers_arr], dtype=np.float64)
        if vals.sum() <= 0:
            continue
        ax_bar.barh(
            layers_arr,
            vals,
            left=bottom,
            color=ROLE_COLOR[role],
            edgecolor="#2F2F2F",
            linewidth=0.35,
            alpha=0.88,
            label=ROLE_LABEL[role],
        )
        bottom += vals
    ax_bar.set_ylim(-0.6, n_layers - 0.4)
    ax_bar.set_xlabel("# Heads in Layer")
    ax_bar.set_ylabel("Layer")
    ax_bar.set_yticks(np.arange(0, n_layers, max(1, n_layers // 10)))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320)
    plt.close(fig)


def _select_candidate_heads(
    probe_rows: Sequence[Dict[str, object]],
    *,
    top_m_per_layer: int,
    candidate_pool_size: int,
) -> List[Tuple[int, int]]:
    per_layer: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    global_sorted = sorted(
        ((float(r.get("selection_score", 0.0)), int(r["layer"]), int(r["head"])) for r in probe_rows),
        key=lambda x: x[0],
        reverse=True,
    )
    for row in probe_rows:
        per_layer[int(row["layer"])].append((float(row.get("selection_score", 0.0)), int(row["head"])))

    keep: set[Tuple[int, int]] = set()
    for layer, vals in per_layer.items():
        vals.sort(key=lambda x: x[0], reverse=True)
        for _, hd in vals[:top_m_per_layer]:
            keep.add((layer, hd))

    for _, layer, head in global_sorted[:candidate_pool_size]:
        keep.add((layer, head))

    return sorted(keep)


def _selection_score_from_shift(mean_label_aligned_dpd_shift: float) -> float:
    # Step1 now prioritizes heads whose ablation makes PD drop, i.e. patched PD
    # is lower than clean PD. Since dPD = patched - clean, more negative dPD is
    # more important, so we rank by its negation.
    return -float(mean_label_aligned_dpd_shift)


def run_step1(
    model_id: str,
    output_dir: Path,
    input_path: str = "",
    hop: str = "one_hop",
    prompt_order: str = "facts_first",
    prompt_style: str = "symbolic",
    impact_samples: int = 500,
    probe_samples: int = 64,
    classify_samples: int = 500,
    top_n: int = 64,
    top_m_per_layer: int = 4,
    candidate_pool_mult: int = 4,
    quantile_keep: float = 0.6,
    late_layer_frac: float = 0.0,
    token_scope: str = "all_tokens",
    score_mode: str = "zscore",
    eval_batch_size: int = 16,
    seed: int = 42,
    device: str = "cuda",
    model_source: str = "modelscope",
    progress_every: int = 10,
    save_plots: bool = True,
) -> Dict[str, object]:
    logger = setup_file_logger(__name__, resolve_log_path(output_dir=output_dir, filename="step1_discover_fast.log"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = resolve_dual_correct_default(model_id, input_path)
    rows = filter_rows(
        input_path=dataset,
        hop=hop,
        prompt_order=prompt_order,
        max_samples=0,
        require_dual_correct=True,
        prompt_style=prompt_style,
    )
    if not rows:
        raise ValueError("No rows after filtering")

    model = load_hooked_transformer(
        model_id,
        device=device,
        source=model_source,
        error_context="heads_analysis",
    )
    true_id, false_id = resolve_true_false_token_ids(model)
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    late_start = max(0, min(n_layers - 1, int(round(late_layer_frac * n_layers))))
    late_layers = list(range(late_start, n_layers))

    impact_rows_data = balanced_sample_by_rule(rows, max_samples=impact_samples) if impact_samples > 0 else list(rows)
    probe_rows_data = balanced_sample_by_rule(impact_rows_data, max_samples=probe_samples) if probe_samples > 0 else list(impact_rows_data)

    # Phase A0: build pairs/means for probe
    log_event(logger, {"stage": "step1_probe_start", "late_layers": len(late_layers), "n_heads": n_heads, "probe_samples": len(probe_rows_data)})
    probe_samples_data, probe_means, _, _, _ = _build_sample_pairs_and_means(
        model=model,
        rows=probe_rows_data,
        prompt_style=prompt_style,
        late_layers=late_layers,
        true_id=true_id,
        false_id=false_id,
        token_scope=token_scope,
        strict_length_match=True,
        progress_every=progress_every,
    )
    if not probe_samples_data:
        raise RuntimeError("No valid probe samples")

    probe_base_acc = sum(int(s.clean_pred == s.label_clean) for s in probe_samples_data) / len(probe_samples_data)
    probe_heads = [(layer, head) for layer in late_layers for head in range(n_heads)]
    probe_evals = _evaluate_necessity_for_head_sets_batched(
        model=model,
        samples=probe_samples_data,
        head_sets=[[(layer, head)] for layer, head in probe_heads],
        mean_mats=probe_means,
        true_id=true_id,
        false_id=false_id,
        token_scope=token_scope,
        eval_batch_size=eval_batch_size,
        progress_every=progress_every,
        progress_tag="probe-impact",
    )
    probe_impact_rows: List[Dict[str, object]] = []
    for (layer, head), ev in zip(probe_heads, probe_evals):
        shifts = ev["dpd_shifts"]
        abs_mean = float(np.mean(np.abs(np.asarray(shifts, dtype=np.float64)))) if shifts else 0.0
        probe_impact_rows.append(
            {
                "layer": layer,
                "head": head,
                "head_id": serialize_head(layer, head),
                "n_samples": ev["n_samples"],
                "mean_abs_dpd_shift": abs_mean,
                "mean_dpd_shift": ev["mean_dpd_shift"],
                "mean_label_aligned_dpd_shift": ev["mean_label_aligned_dpd_shift"],
                "selection_score": _selection_score_from_shift(ev["mean_label_aligned_dpd_shift"]),
                "accuracy": ev["accuracy"],
                "accuracy_drop": probe_base_acc - float(ev["accuracy"]),
            }
        )

    candidate_pool_size = max(top_n, int(top_n * max(1, candidate_pool_mult)))
    candidates = _select_candidate_heads(
        probe_impact_rows,
        top_m_per_layer=max(1, top_m_per_layer),
        candidate_pool_size=candidate_pool_size,
    )
    log_event(logger, {"stage": "step1_probe_done", "candidate_heads": len(candidates)})

    # Phase A1: refine on larger set but only candidate heads
    refine_samples_data, refine_means, _, _, _ = _build_sample_pairs_and_means(
        model=model,
        rows=impact_rows_data,
        prompt_style=prompt_style,
        late_layers=late_layers,
        true_id=true_id,
        false_id=false_id,
        token_scope=token_scope,
        strict_length_match=True,
        progress_every=progress_every,
    )
    if not refine_samples_data:
        raise RuntimeError("No valid refine samples")
    base_clean_acc = sum(int(s.clean_pred == s.label_clean) for s in refine_samples_data) / len(refine_samples_data)

    refine_evals = _evaluate_necessity_for_head_sets_batched(
        model=model,
        samples=refine_samples_data,
        head_sets=[[(layer, head)] for layer, head in candidates],
        mean_mats=refine_means,
        true_id=true_id,
        false_id=false_id,
        token_scope=token_scope,
        eval_batch_size=eval_batch_size,
        progress_every=progress_every,
        progress_tag="refine-impact",
    )
    refined_impact: List[Dict[str, object]] = []
    for (layer, head), ev in zip(candidates, refine_evals):
        shifts = ev["dpd_shifts"]
        abs_mean = float(np.mean(np.abs(np.asarray(shifts, dtype=np.float64)))) if shifts else 0.0
        refined_impact.append(
            {
                "layer": layer,
                "head": head,
                "head_id": serialize_head(layer, head),
                "n_samples": ev["n_samples"],
                "mean_abs_dpd_shift": abs_mean,
                "mean_dpd_shift": ev["mean_dpd_shift"],
                "mean_label_aligned_dpd_shift": ev["mean_label_aligned_dpd_shift"],
                "selection_score": _selection_score_from_shift(ev["mean_label_aligned_dpd_shift"]),
                "accuracy": ev["accuracy"],
                "accuracy_drop": base_clean_acc - float(ev["accuracy"]),
            }
        )

    # Step1 is a necessity screen on clean prompts. We therefore keep heads whose
    # ablation drives PD downward, i.e. negative dPD = patched_PD - clean_PD.
    refined_impact.sort(key=lambda x: float(x.get("selection_score", 0.0)), reverse=True)
    decreasing_rows = [r for r in refined_impact if float(r.get("selection_score", 0.0)) > 0.0]
    selection_pool = decreasing_rows if decreasing_rows else list(refined_impact)
    score_vals = np.asarray([float(r.get("selection_score", 0.0)) for r in selection_pool], dtype=np.float64)
    thresh = float(np.quantile(score_vals, 1.0 - quantile_keep)) if score_vals.size > 0 else 0.0
    filtered = [r for r in selection_pool if float(r.get("selection_score", 0.0)) >= thresh]

    ranked_pool = filtered if filtered else selection_pool
    ranked_pool.sort(
        key=lambda x: (
            float(x.get("selection_score", 0.0)),
            float(x.get("accuracy_drop", 0.0)),
            float(x.get("mean_abs_dpd_shift", 0.0)),
        ),
        reverse=True,
    )
    top_heads = ranked_pool[:top_n]
    for rank, row in enumerate(top_heads, 1):
        row["rank"] = rank

    impact_dir = output_dir / "impact"
    impact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(impact_dir / "impact_probe_all_heads.csv", probe_impact_rows)
    write_csv(impact_dir / "impact_refined_candidates.csv", refined_impact)
    write_csv(impact_dir / "impact_top_heads.csv", top_heads)
    write_json(
        impact_dir / "impact_top_heads.json",
        {
            "model_id": model_id,
            "hop": hop,
            "token_scope": token_scope,
            "probe_samples": len(probe_samples_data),
            "impact_samples": len(refine_samples_data),
            "candidate_count": len(candidates),
            "quantile_threshold": thresh,
            "selection_metric": "selection_score = -(mean_label_aligned_dpd_shift), where dPD = patched_PD - clean_PD",
            "top_heads": top_heads,
        },
    )

    # Phase B: pattern classification on top heads
    log_event(logger, {"stage": "step1_classify_start", "top_heads": len(top_heads)})
    classify_rows = balanced_sample_by_rule(rows, max_samples=classify_samples) if classify_samples > 0 else list(rows)

    heads_by_layer: Dict[int, List[int]] = defaultdict(list)
    impact_by_head: Dict[Tuple[int, int], Dict[str, object]] = {}
    for item in top_heads:
        l, h = int(item["layer"]), int(item["head"])
        heads_by_layer[l].append(h)
        impact_by_head[(l, h)] = dict(item)

    pattern_names = {utils.get_act_name("pattern", l) for l in heads_by_layer}
    metric_acc: Dict[Tuple[int, int], Dict[str, List[float]]] = {
        k: {
            "fact_focus_ratio": [],
            "boundary_focus_ratio": [],
            "intra_clause_ratio": [],
            "attn_entropy": [],
            "top1_mass": [],
        }
        for k in impact_by_head
    }

    with torch.no_grad():
        classify_progress = make_tqdm(classify_rows, total=len(classify_rows), desc="step1-classify", leave=False, disable=len(classify_rows) <= 1)
        for i, row in enumerate(classify_progress, 1):
            clean_prompt = resolve_prompt(row, prompt_style=prompt_style, kind="clean")
            tok_meta = tokenize_prompt_for_eval_alignment(model, clean_prompt)
            str_toks = tok_meta.raw_str_tokens
            po = infer_prompt_order(row, clean_prompt)
            q_pos = tok_meta.body_start + query_is_pos(str_toks, po)
            fact_pos = [tok_meta.body_start + pos for pos in find_fact_positions(str_toks)]
            comma_pos = [tok_meta.body_start + pos for pos in find_comma_positions(str_toks)]
            reg_starts = [tok_meta.body_start + pos for pos in region_starts_from_commas(str_toks)]

            _, cache = model.run_with_cache(tok_meta.tokens, names_filter=lambda n: n in pattern_names)
            for layer, hds in heads_by_layer.items():
                patt = cache[utils.get_act_name("pattern", layer)][0]
                for hd in hds:
                    attn = patt[hd]
                    rec = metric_acc[(layer, hd)]
                    rec["fact_focus_ratio"].append(_fact_focus(attn, q_pos, fact_pos))
                    rec["boundary_focus_ratio"].append(_boundary_focus(attn, comma_pos))
                    rec["intra_clause_ratio"].append(_intra_clause(attn, reg_starts))
                    ent, t1 = _entropy_and_top1(attn, q_pos)
                    rec["attn_entropy"].append(ent)
                    rec["top1_mass"].append(t1)

            if progress_every > 0 and (i % progress_every == 0 or i == len(classify_rows)):
                classify_progress.set_postfix(done=i, total=len(classify_rows))
                log_event(logger, {"stage": "step1_classify_progress", "done": i, "total": len(classify_rows)})

    classify_out: List[Dict[str, object]] = []
    feat_rows: List[List[float]] = []

    for (layer, head), metrics in metric_acc.items():
        mm = {k: safe_mean(v) for k, v in metrics.items()}
        n = len(metrics["fact_focus_ratio"])
        h = n // 2
        stability = 0.0
        if h > 0 and n - h > 0:
            parts = []
            for k in ["fact_focus_ratio", "boundary_focus_ratio", "intra_clause_ratio", "top1_mass"]:
                a, b = safe_mean(metrics[k][:h]), safe_mean(metrics[k][h:])
                parts.append(1.0 - abs(a - b) / (abs(a) + abs(b) + 1e-9))
            stability = safe_mean(parts)

        imp = impact_by_head[(layer, head)]
        row_out = {
            "rank": imp.get("rank", 0),
            "layer": layer,
            "head": head,
            "head_id": serialize_head(layer, head),
            "mean_abs_dpd_shift": imp.get("mean_abs_dpd_shift", 0.0),
            "mean_dpd_shift": imp.get("mean_dpd_shift", 0.0),
            "mean_label_aligned_dpd_shift": imp.get("mean_label_aligned_dpd_shift", 0.0),
            "selection_score": imp.get("selection_score", 0.0),
            "accuracy_drop": imp.get("accuracy_drop", 0.0),
            **mm,
            "stability_score": stability,
        }
        classify_out.append(row_out)
        feat_rows.append(
            [mm["fact_focus_ratio"], mm["boundary_focus_ratio"], mm["intra_clause_ratio"], mm["attn_entropy"], mm["top1_mass"], stability]
        )

    paired_rows = list(zip(classify_out, feat_rows))
    paired_rows.sort(key=lambda item: int(item[0].get("rank", 0)))
    classify_out = [row for row, _ in paired_rows]
    feat_np = np.asarray([feat for _, feat in paired_rows], dtype=np.float64)
    feat_z = _zscore_matrix(feat_np) if feat_np.size > 0 else np.zeros((0, 6))

    metric_keys = ["fact_focus_ratio", "boundary_focus_ratio", "intra_clause_ratio", "attn_entropy", "top1_mass", "stability_score"]
    score_mat = feat_z if score_mode == "zscore" and feat_np.size > 0 else feat_np
    rule_margin = 0.08 if score_mode == "zscore" else 0.06
    min_rule_score = 0.15 if score_mode == "zscore" else 1.02

    for idx, row in enumerate(classify_out):
        metrics_score = {k: float(score_mat[idx, j]) for j, k in enumerate(metric_keys)}
        role_label, role_source, role_scores = _label_by_rules(metrics_score, margin=rule_margin, min_score=min_rule_score)
        row["role_label"] = role_label
        row["role_source"] = role_source
        row["score_fact"] = role_scores["fact_retrieval"]
        row["score_split"] = role_scores["splitting"]
        row["score_trans"] = role_scores["transmission"]

    classify_dir = output_dir / "classify"
    classify_dir.mkdir(parents=True, exist_ok=True)
    write_csv(classify_dir / "top_heads_pattern_labels.csv", classify_out)
    write_json(
        classify_dir / "top_heads_pattern_labels.json",
        {
            "model_id": model_id,
            "hop": hop,
            "n_heads": len(classify_out),
            "role_counts": {r: sum(1 for x in classify_out if x.get("role_label") == r) for r in ROLE_ORDER},
            "top_heads": classify_out,
        },
    )
    plot_path = classify_dir / "layer_head_role_distribution.png"
    if save_plots:
        _plot_layer_head_distribution(classify_out, plot_path, n_layers=n_layers, n_heads=n_heads)

    summary = {
        "model_id": model_id,
        "hop": hop,
        "token_scope": token_scope,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "late_layer_frac": float(late_layer_frac),
        "late_layer_start": int(late_start),
        "late_layers_count": int(len(late_layers)),
        "eval_batch_size": int(max(1, eval_batch_size)),
        "filtered_rows": len(rows),
        "filtered_rule_counts": count_by_field(rows, field="rule"),
        "probe_samples": len(probe_samples_data),
        "probe_rule_counts": count_by_field(probe_samples_data, field="rule"),
        "impact_samples": len(refine_samples_data),
        "impact_rule_counts": count_by_field(refine_samples_data, field="rule"),
        "classify_rows": len(classify_rows),
        "classify_rule_counts": count_by_field(classify_rows, field="rule"),
        "candidate_count": len(candidates),
        "top_n": len(top_heads),
        "selection_metric": "selection_score = -(mean_label_aligned_dpd_shift), larger means stronger PD decrease after ablation",
        "classify_csv": str(classify_dir / "top_heads_pattern_labels.csv"),
        "classify_json": str(classify_dir / "top_heads_pattern_labels.json"),
        "plot_png": str(plot_path),
        "plot_generated": bool(save_plots),
    }
    write_json(output_dir / "step1_summary.json", summary)
    log_event(logger, {"stage": "step1_fast_done", **summary, "output_dir": str(output_dir)})

    return {
        "model_id": model_id,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step1-fast: low-complexity impact+classification")
    parser.add_argument("--model_id", required=True)
    add_model_source_arg(parser)
    parser.add_argument("--input", default="")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hop", default="one_hop")
    parser.add_argument("--prompt_order", default="facts_first")
    parser.add_argument("--prompt_style", default="symbolic")
    parser.add_argument("--impact_samples", type=int, default=500, help="Balanced per-rule budget for impact/refine stage.")
    parser.add_argument("--probe_samples", type=int, default=64, help="Balanced per-rule budget for probe stage.")
    parser.add_argument("--classify_samples", type=int, default=500, help="Balanced per-rule budget for pattern classification.")
    parser.add_argument("--top_n", type=int, default=64)
    parser.add_argument("--top_m_per_layer", type=int, default=4)
    parser.add_argument("--candidate_pool_mult", type=int, default=4)
    parser.add_argument("--quantile_keep", type=float, default=0.6)
    parser.add_argument("--late_layer_frac", type=float, default=0.0)
    parser.add_argument("--token_scope", default="all_tokens", choices=["query_only", "all_tokens"])
    parser.add_argument("--score_mode", default="zscore")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument(
        "--save_plots",
        "--save-plots",
        dest="save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    run_step1(
        model_id=args.model_id,
        output_dir=args.output_dir,
        input_path=args.input,
        hop=args.hop,
        prompt_order=args.prompt_order,
        prompt_style=args.prompt_style,
        impact_samples=args.impact_samples,
        probe_samples=args.probe_samples,
        classify_samples=args.classify_samples,
        top_n=args.top_n,
        top_m_per_layer=args.top_m_per_layer,
        candidate_pool_mult=args.candidate_pool_mult,
        quantile_keep=args.quantile_keep,
        late_layer_frac=args.late_layer_frac,
        token_scope=args.token_scope,
        score_mode=args.score_mode,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        device=args.device,
        model_source=args.model_source,
        progress_every=args.progress_every,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
