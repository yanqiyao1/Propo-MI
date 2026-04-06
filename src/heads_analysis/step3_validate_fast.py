"""Step 3 - clean-prompt validation with shared PD/accuracy outputs."""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import utils

from src.eval.io_utils import count_by_field
from src.model_loading import add_model_source_arg
from src.progress import log_event, make_tqdm, resolve_log_path, setup_file_logger
from src.plot_style import stylize_axis

from .common import (
    ROLE_COLOR,
    ROLE_LABEL,
    ROLE_ORDER,
    apply_paper_style,
    bool_prediction_from_logits,
    compute_correct_incorrect_prob_diff,
    filter_rows,
    infer_prompt_order,
    load_hooked_transformer,
    query_is_pos,
    resolve_dual_correct_default,
    resolve_prompt,
    resolve_true_false_token_ids,
    safe_mean,
    safe_sem,
    serialize_heads,
    stable_int_from_text,
    tokenize_prompt_for_eval_alignment,
    write_csv,
    write_json,
)


LOGGER = logging.getLogger(__name__)

RATIO_EPS = 1e-6


@dataclass(frozen=True)
class ValidationSample:
    row_key: str
    sample_id: str
    hop: str
    rule: str
    label: bool
    clean_tokens: torch.Tensor
    query_pos: int
    answer_pos: int
    clean_pred: bool
    clean_dpd: float


@dataclass(frozen=True)
class HeadRecord:
    order: int
    role: str
    layer: int
    head: int


@dataclass(frozen=True)
class InterventionSpec:
    strategy_id: str
    strategy_label: str
    strategy_type: str
    roles: Tuple[str, ...]
    k: int
    trial: int
    heads: Tuple[Tuple[int, int], ...]


STRATEGY_ORDER = [
    "single_fact_retrieval",
    "single_splitting",
    "single_transmission",
    "pair_fact_retrieval__splitting",
    "pair_fact_retrieval__transmission",
    "pair_splitting__transmission",
    "mixed_all",
    "random_outside_selected",
]

STRATEGY_LABEL = {
    "single_fact_retrieval": ROLE_LABEL["fact_retrieval"],
    "single_splitting": ROLE_LABEL["splitting"],
    "single_transmission": ROLE_LABEL["transmission"],
    "pair_fact_retrieval__splitting": f'{ROLE_LABEL["fact_retrieval"]} + {ROLE_LABEL["splitting"]}',
    "pair_fact_retrieval__transmission": f'{ROLE_LABEL["fact_retrieval"]} + {ROLE_LABEL["transmission"]}',
    "pair_splitting__transmission": f'{ROLE_LABEL["splitting"]} + {ROLE_LABEL["transmission"]}',
    "mixed_all": "Mixed-All",
    "random_outside_selected": "Random-Outside",
}

STRATEGY_TYPE = {
    "single_fact_retrieval": "single",
    "single_splitting": "single",
    "single_transmission": "single",
    "pair_fact_retrieval__splitting": "pair",
    "pair_fact_retrieval__transmission": "pair",
    "pair_splitting__transmission": "pair",
    "mixed_all": "mixed",
    "random_outside_selected": "random",
}

STRATEGY_ROLES = {
    "single_fact_retrieval": ("fact_retrieval",),
    "single_splitting": ("splitting",),
    "single_transmission": ("transmission",),
    "pair_fact_retrieval__splitting": ("fact_retrieval", "splitting"),
    "pair_fact_retrieval__transmission": ("fact_retrieval", "transmission"),
    "pair_splitting__transmission": ("splitting", "transmission"),
    "mixed_all": tuple(ROLE_ORDER),
    "random_outside_selected": tuple(),
}

STRATEGY_COLOR = {
    "single_fact_retrieval": ROLE_COLOR["fact_retrieval"],
    "single_splitting": ROLE_COLOR["splitting"],
    "single_transmission": ROLE_COLOR["transmission"],
    "pair_fact_retrieval__splitting": "#7A4C99",
    "pair_fact_retrieval__transmission": "#B8860B",
    "pair_splitting__transmission": "#3B8EA5",
    "mixed_all": "#2F2F2F",
    "random_outside_selected": "#9A9A9A",
}

STRATEGY_MARKER = {
    "single_fact_retrieval": "o",
    "single_splitting": "s",
    "single_transmission": "^",
    "pair_fact_retrieval__splitting": "D",
    "pair_fact_retrieval__transmission": "P",
    "pair_splitting__transmission": "X",
    "mixed_all": "*",
    "random_outside_selected": "v",
}


def _row_key(row: Mapping[str, object]) -> str:
    row_id = str(row.get("id", "")).strip()
    if row_id:
        return row_id
    return json.dumps(dict(row), ensure_ascii=True, sort_keys=True, default=str)


def _select_balanced_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    max_samples: int,
    rule_key: str = "rule",
) -> List[Dict[str, object]]:
    rows_list = [dict(row) for row in rows]
    if max_samples <= 0 or len(rows_list) <= max_samples:
        return rows_list

    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows_list):
        grouped_indices[str(row.get(rule_key, ""))].append(idx)

    ordered_rules = sorted(grouped_indices.keys(), key=lambda name: (len(grouped_indices[name]), name))
    cursor = {name: 0 for name in ordered_rules}
    active_rules = [name for name in ordered_rules if grouped_indices[name]]
    selected_indices: List[int] = []

    while len(selected_indices) < max_samples and active_rules:
        next_active: List[str] = []
        for rule_name in active_rules:
            offset = cursor[rule_name]
            if offset < len(grouped_indices[rule_name]) and len(selected_indices) < max_samples:
                selected_indices.append(grouped_indices[rule_name][offset])
                cursor[rule_name] = offset + 1
            if cursor[rule_name] < len(grouped_indices[rule_name]):
                next_active.append(rule_name)
            if len(selected_indices) >= max_samples:
                break
        active_rules = next_active

    return [rows_list[idx] for idx in selected_indices]


def _merge_selected_rows(*row_groups: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    seen: set[str] = set()
    for group in row_groups:
        for row in group:
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(row))
    return merged


def _resolve_selected_samples(
    rows: Sequence[Mapping[str, object]],
    sample_by_key: Mapping[str, ValidationSample],
) -> List[ValidationSample]:
    resolved: List[ValidationSample] = []
    missing: List[str] = []
    for row in rows:
        key = _row_key(row)
        sample = sample_by_key.get(key)
        if sample is None:
            missing.append(key)
            continue
        resolved.append(sample)
    if missing:
        raise RuntimeError(f"Missing built validation samples for {len(missing)} selected rows.")
    return resolved


def _build_validation_samples_and_layer_means(
    *,
    model,
    rows: Sequence[Dict[str, object]],
    prompt_style: str,
    late_layers: Sequence[int],
    true_id: int,
    false_id: int,
    token_scope: str,
    progress_every: int,
) -> Tuple[List[ValidationSample], Dict[int, torch.Tensor]]:
    act_names = {utils.get_act_name("z", layer) for layer in late_layers}
    layer_sum: Dict[int, torch.Tensor] = {}
    layer_count: Dict[int, int] = defaultdict(int)
    samples: List[ValidationSample] = []

    with torch.no_grad():
        progress = make_tqdm(rows, total=len(rows), desc="step3-build-clean", leave=False, disable=len(rows) <= 1)
        for idx, row in enumerate(progress, start=1):
            clean_prompt = resolve_prompt(row, prompt_style=prompt_style, kind="clean")
            tok_meta = tokenize_prompt_for_eval_alignment(model, clean_prompt)
            clean_tokens = tok_meta.tokens
            prompt_order = infer_prompt_order(row, clean_prompt)
            query_pos = tok_meta.body_start + query_is_pos(tok_meta.raw_str_tokens, prompt_order)

            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: n in act_names)
            label = bool(row.get("label", False))
            pred, _, _ = bool_prediction_from_logits(clean_logits, true_id, false_id, pos=tok_meta.answer_pos)
            clean_dpd = float(compute_correct_incorrect_prob_diff(clean_logits, label, true_id, false_id, pos=tok_meta.answer_pos).item())
            samples.append(
                ValidationSample(
                    row_key=_row_key(row),
                    sample_id=str(row.get("id", "")),
                    hop=str(row.get("hop", "")),
                    rule=str(row.get("rule", "")),
                    label=label,
                    clean_tokens=clean_tokens,
                    query_pos=query_pos,
                    answer_pos=tok_meta.answer_pos,
                    clean_pred=bool(pred.item()),
                    clean_dpd=clean_dpd,
                )
            )

            for layer in late_layers:
                z = clean_cache[utils.get_act_name("z", layer)][0]
                if token_scope == "query_only":
                    layer_value = z[query_pos, :, :]
                    count_inc = 1
                else:
                    layer_value = z.sum(dim=0)
                    count_inc = int(z.shape[0])
                if layer not in layer_sum:
                    layer_sum[layer] = torch.zeros_like(layer_value, dtype=torch.float32)
                layer_sum[layer] += layer_value.to(dtype=torch.float32)
                layer_count[layer] += count_inc

            if progress_every > 0 and (idx % progress_every == 0 or idx == len(rows)):
                progress.set_postfix(done=idx, kept=len(samples))
                log_event(
                    LOGGER,
                    {
                        "stage": "step3_build_progress",
                        "done": idx,
                        "total": len(rows),
                        "samples": len(samples),
                    },
                )

    if not samples:
        raise RuntimeError("No validation samples were built.")
    layer_means = {
        layer: (layer_sum[layer] / float(max(1, layer_count[layer]))).detach()
        for layer in late_layers
        if layer in layer_sum
    }
    return samples, layer_means


def _group_records_by_layer(records: Sequence[HeadRecord]) -> Dict[int, List[HeadRecord]]:
    grouped: Dict[int, List[HeadRecord]] = defaultdict(list)
    for rec in records:
        grouped[int(rec.layer)].append(rec)
    for layer in grouped:
        grouped[layer].sort(key=lambda item: item.order)
    return dict(grouped)


def _next_group_order(grouped: Mapping[int, Sequence[HeadRecord]]) -> int:
    orders = [items[0].order for items in grouped.values() if items]
    return min(orders) if orders else 10**12


def _select_balanced_from_groups(group_pools: Mapping[str, Sequence[HeadRecord]], target_k: int) -> List[HeadRecord]:
    grouped = {
        group: _group_records_by_layer(records)
        for group, records in group_pools.items()
        if records
    }
    if target_k <= 0 or not grouped:
        return []

    selected: List[HeadRecord] = []
    group_counts: Dict[str, int] = defaultdict(int)
    layer_counts: Dict[int, int] = defaultdict(int)

    while len(selected) < target_k:
        active_groups = [group for group, by_layer in grouped.items() if any(by_layer.values())]
        if not active_groups:
            break
        active_groups.sort(
            key=lambda group: (
                group_counts[group],
                _next_group_order(grouped[group]),
                group,
            )
        )
        chosen_group = active_groups[0]
        by_layer = grouped[chosen_group]
        active_layers = [layer for layer, items in by_layer.items() if items]
        if not active_layers:
            break
        active_layers.sort(
            key=lambda layer: (
                layer_counts[layer],
                by_layer[layer][0].order,
                layer,
            )
        )
        chosen_layer = active_layers[0]
        rec = by_layer[chosen_layer].pop(0)
        selected.append(rec)
        group_counts[chosen_group] += 1
        layer_counts[chosen_layer] += 1
    return selected


def _sample_balanced_random_heads(
    pool_by_layer: Mapping[int, Sequence[int]],
    *,
    target_k: int,
    seed: int,
) -> Tuple[Tuple[int, int], ...]:
    rng = random.Random(seed)
    available: Dict[int, List[int]] = {int(layer): list(heads) for layer, heads in pool_by_layer.items() if heads}
    for heads in available.values():
        rng.shuffle(heads)

    chosen: List[Tuple[int, int]] = []
    layer_counts: Dict[int, int] = defaultdict(int)
    while len(chosen) < target_k:
        candidate_layers = [layer for layer, heads in available.items() if heads]
        if not candidate_layers:
            break
        min_count = min(layer_counts[layer] for layer in candidate_layers)
        balanced_layers = [layer for layer in candidate_layers if layer_counts[layer] == min_count]
        chosen_layer = rng.choice(sorted(balanced_layers))
        head_idx = rng.randrange(len(available[chosen_layer]))
        head = available[chosen_layer].pop(head_idx)
        chosen.append((chosen_layer, int(head)))
        layer_counts[chosen_layer] += 1
    return tuple(sorted(chosen))


def _load_classified_heads(
    classify_json: Path,
    *,
    late_layers: set[int],
) -> Tuple[Dict[str, List[HeadRecord]], List[Tuple[int, int]]]:
    payload = json.loads(classify_json.read_text(encoding="utf-8"))
    top_heads = payload.get("top_heads", [])
    role_heads: Dict[str, List[HeadRecord]] = defaultdict(list)
    selected_heads: List[Tuple[int, int]] = []
    seen_selected: set[Tuple[int, int]] = set()
    for order, item in enumerate(top_heads):
        layer = int(item["layer"])
        head = int(item["head"])
        if layer not in late_layers:
            continue
        role = str(item.get("role_label", "")).strip()
        if role not in ROLE_ORDER:
            continue
        rec = HeadRecord(order=order, role=role, layer=layer, head=head)
        role_heads[role].append(rec)
        key = (layer, head)
        if key not in seen_selected:
            seen_selected.add(key)
            selected_heads.append(key)
    return dict(role_heads), selected_heads


def _build_intervention_specs(
    *,
    role_heads: Mapping[str, Sequence[HeadRecord]],
    outside_selected_pool_by_layer: Mapping[int, Sequence[int]],
    k_values: Sequence[int],
    random_trials: int,
    seed: int,
) -> Tuple[List[InterventionSpec], Dict[str, int]]:
    specs: List[InterventionSpec] = []
    random_trials_used: Dict[str, int] = {}

    deterministic_strategy_ids = [
        "single_fact_retrieval",
        "single_splitting",
        "single_transmission",
        "pair_fact_retrieval__splitting",
        "pair_fact_retrieval__transmission",
        "pair_splitting__transmission",
        "mixed_all",
    ]

    for strategy_id in deterministic_strategy_ids:
        roles = STRATEGY_ROLES[strategy_id]
        group_pools = {role: list(role_heads.get(role, [])) for role in roles}
        if any(len(group_pools[role]) == 0 for role in roles):
            continue
        total_available = sum(len(group_pools[role]) for role in roles)
        if total_available <= 0:
            continue
        for k in k_values:
            if total_available < k:
                continue
            chosen = _select_balanced_from_groups(group_pools, k)
            if len(chosen) != k:
                continue
            specs.append(
                InterventionSpec(
                    strategy_id=strategy_id,
                    strategy_label=STRATEGY_LABEL[strategy_id],
                    strategy_type=STRATEGY_TYPE[strategy_id],
                    roles=roles,
                    k=k,
                    trial=0,
                    heads=tuple((rec.layer, rec.head) for rec in chosen),
                )
            )

    for k in k_values:
        used = 0
        seen: set[Tuple[Tuple[int, int], ...]] = set()
        for trial in range(max(0, random_trials)):
            sampled = _sample_balanced_random_heads(
                outside_selected_pool_by_layer,
                target_k=k,
                seed=seed + 1009 * (trial + 1) + 131 * k + stable_int_from_text("random_outside_selected"),
            )
            if len(sampled) != k or sampled in seen:
                continue
            seen.add(sampled)
            specs.append(
                InterventionSpec(
                    strategy_id="random_outside_selected",
                    strategy_label=STRATEGY_LABEL["random_outside_selected"],
                    strategy_type=STRATEGY_TYPE["random_outside_selected"],
                    roles=tuple(),
                    k=k,
                    trial=used,
                    heads=sampled,
                )
            )
            used += 1
        random_trials_used[f"random_outside_selected::k{k}"] = used
    return specs, random_trials_used


def _prepare_head_sets(head_sets: Sequence[Sequence[Tuple[int, int]]]) -> List[Dict[int, Tuple[int, ...]]]:
    prepared: List[Dict[int, Tuple[int, ...]]] = []
    for heads in head_sets:
        by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer, head in heads:
            by_layer[int(layer)].append(int(head))
        prepared.append({layer: tuple(sorted(set(hids))) for layer, hids in by_layer.items()})
    return prepared


def _eval_clean_interventions_many(
    *,
    model,
    samples: Sequence[ValidationSample],
    specs: Sequence[InterventionSpec],
    layer_means: Mapping[int, torch.Tensor],
    true_id: int,
    false_id: int,
    token_scope: str,
    eval_batch_size: int,
    progress_every: int,
    progress_tag: str,
) -> List[Dict[str, object]]:
    if not specs:
        return []

    prepared = _prepare_head_sets([spec.heads for spec in specs])
    cond_batch = max(1, int(eval_batch_size))
    n_cond = len(prepared)

    intervened_dpd_values: List[List[float]] = [[] for _ in range(n_cond)]
    dpd_shift_values: List[List[float]] = [[] for _ in range(n_cond)]
    abs_ratio_values: List[List[float]] = [[] for _ in range(n_cond)]
    signed_ratio_values: List[List[float]] = [[] for _ in range(n_cond)]
    correct_counts = [0 for _ in range(n_cond)]

    base_accuracy = sum(int(sample.clean_pred == sample.label) for sample in samples) / len(samples)
    base_mean_dpd = safe_mean([sample.clean_dpd for sample in samples])

    with torch.no_grad():
        progress = make_tqdm(samples, total=len(samples), desc=progress_tag or "step3-eval", leave=False, disable=len(samples) <= 1)
        for sample_idx, sample in enumerate(progress, start=1):
            for start in range(0, n_cond, cond_batch):
                end = min(start + cond_batch, n_cond)
                chunk = prepared[start:end]
                batch_size = end - start
                toks = sample.clean_tokens.repeat(batch_size, 1)

                layer_assignments: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)
                for batch_idx, by_layer in enumerate(chunk):
                    for layer, head_ids in by_layer.items():
                        if head_ids:
                            layer_assignments[layer].append((batch_idx, head_ids))

                hooks = []
                for layer, assignments in layer_assignments.items():
                    act = utils.get_act_name("z", layer)
                    layer_mean = layer_means[layer]

                    def hfn(
                        z,
                        hook,
                        assignments_=tuple(assignments),
                        layer_mean_=layer_mean,
                        query_pos_=sample.query_pos,
                        token_scope_=token_scope,
                    ):
                        del hook
                        replacement = layer_mean_.to(dtype=z.dtype, device=z.device)
                        for batch_idx, head_ids in assignments_:
                            idx = list(head_ids)
                            replacement_heads = replacement[idx, :]
                            if token_scope_ == "query_only":
                                if query_pos_ < z.shape[1]:
                                    z[batch_idx, query_pos_, idx, :] = replacement_heads
                            else:
                                z[batch_idx, :, idx, :] = replacement_heads.unsqueeze(0).expand(z.shape[1], len(idx), -1)
                        return z

                    hooks.append((act, hfn))

                logits = model.run_with_hooks(toks, fwd_hooks=hooks)
                pred, _, _ = bool_prediction_from_logits(logits, true_id, false_id, pos=sample.answer_pos)
                pred_arr = pred.detach().to(dtype=torch.bool).cpu().numpy()
                dpd_arr = (
                    compute_correct_incorrect_prob_diff(logits, sample.label, true_id, false_id, pos=sample.answer_pos)
                    .detach()
                    .cpu()
                    .numpy()
                )

                for batch_idx in range(batch_size):
                    cond_idx = start + batch_idx
                    intervened_dpd = float(dpd_arr[batch_idx])
                    dpd_shift = intervened_dpd - sample.clean_dpd
                    intervened_dpd_values[cond_idx].append(intervened_dpd)
                    dpd_shift_values[cond_idx].append(dpd_shift)
                    if abs(sample.clean_dpd) > RATIO_EPS:
                        abs_ratio_values[cond_idx].append(abs(dpd_shift) / abs(sample.clean_dpd))
                        signed_ratio_values[cond_idx].append(dpd_shift / sample.clean_dpd)
                    if bool(pred_arr[batch_idx]) == sample.label:
                        correct_counts[cond_idx] += 1

            if progress_every > 0 and (sample_idx % progress_every == 0 or sample_idx == len(samples)):
                progress.set_postfix(sample=sample_idx)
                log_event(
                    LOGGER,
                    {
                        "stage": "step3_eval_progress",
                        "tag": progress_tag,
                        "sample": sample_idx,
                        "total": len(samples),
                    },
                )

    results: List[Dict[str, object]] = []
    for cond_idx, spec in enumerate(specs):
        results.append(
            {
                "strategy_id": spec.strategy_id,
                "strategy_label": spec.strategy_label,
                "strategy_type": spec.strategy_type,
                "roles": ",".join(spec.roles),
                "k": int(spec.k),
                "trial": int(spec.trial),
                "n_heads": int(len(spec.heads)),
                "heads": serialize_heads(spec.heads),
                "base_accuracy": float(base_accuracy),
                "patched_accuracy": float(correct_counts[cond_idx] / len(samples)) if samples else 0.0,
                "accuracy_drop": float(base_accuracy - (correct_counts[cond_idx] / len(samples))) if samples else 0.0,
                "mean_original_dpd": float(base_mean_dpd),
                "mean_intervened_dpd": float(safe_mean(intervened_dpd_values[cond_idx])),
                "mean_dpd_shift": float(safe_mean(dpd_shift_values[cond_idx])),
                "mean_abs_relative_dpd": float(safe_mean(abs_ratio_values[cond_idx])),
                "mean_signed_relative_dpd": float(safe_mean(signed_ratio_values[cond_idx])),
                "n_ratio_valid": int(len(abs_ratio_values[cond_idx])),
                "n": int(len(samples)),
            }
        )
    return results


def _aggregate_metric_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["strategy_id"]), int(row["k"]))].append(row)

    aggregated: List[Dict[str, object]] = []
    order_index = {strategy_id: idx for idx, strategy_id in enumerate(STRATEGY_ORDER)}
    for (strategy_id, k), group_rows in sorted(grouped.items(), key=lambda item: (order_index.get(item[0][0], 999), item[0][1])):
        first = group_rows[0]
        aggregated.append(
            {
                "strategy_id": strategy_id,
                "strategy_label": str(first["strategy_label"]),
                "strategy_type": str(first["strategy_type"]),
                "roles": str(first["roles"]),
                "k": int(k),
                "n_trials": int(len(group_rows)),
                "base_accuracy": float(first["base_accuracy"]),
                "patched_accuracy_mean": float(safe_mean([float(row["patched_accuracy"]) for row in group_rows])),
                "patched_accuracy_sem": float(safe_sem([float(row["patched_accuracy"]) for row in group_rows])),
                "accuracy_drop_mean": float(safe_mean([float(row["accuracy_drop"]) for row in group_rows])),
                "accuracy_drop_sem": float(safe_sem([float(row["accuracy_drop"]) for row in group_rows])),
                "mean_original_dpd": float(first["mean_original_dpd"]),
                "mean_intervened_dpd_mean": float(safe_mean([float(row["mean_intervened_dpd"]) for row in group_rows])),
                "mean_intervened_dpd_sem": float(safe_sem([float(row["mean_intervened_dpd"]) for row in group_rows])),
                "mean_dpd_shift_mean": float(safe_mean([float(row["mean_dpd_shift"]) for row in group_rows])),
                "mean_dpd_shift_sem": float(safe_sem([float(row["mean_dpd_shift"]) for row in group_rows])),
                "mean_abs_relative_dpd_mean": float(safe_mean([float(row["mean_abs_relative_dpd"]) for row in group_rows])),
                "mean_abs_relative_dpd_sem": float(safe_sem([float(row["mean_abs_relative_dpd"]) for row in group_rows])),
                "mean_signed_relative_dpd_mean": float(safe_mean([float(row["mean_signed_relative_dpd"]) for row in group_rows])),
                "mean_signed_relative_dpd_sem": float(safe_sem([float(row["mean_signed_relative_dpd"]) for row in group_rows])),
                "n_ratio_valid_mean": float(safe_mean([float(row["n_ratio_valid"]) for row in group_rows])),
                "n_samples": int(first["n"]),
            }
        )
    return aggregated


def _plot_pd_curves(
    aggregated_rows: Sequence[Mapping[str, object]],
    output: Path,
    *,
    y_mean_key: str,
    y_sem_key: str,
    ylabel: str,
    k_values: Sequence[int],
) -> None:
    if not aggregated_rows:
        return
    apply_paper_style(
        {
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 14.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 14.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.0,
        }
    )
    fig, ax = plt.subplots(figsize=(11.6, 6.8))

    any_data = False
    for strategy_id in STRATEGY_ORDER:
        strat_rows = [row for row in aggregated_rows if str(row["strategy_id"]) == strategy_id]
        if not strat_rows:
            continue
        strat_rows = sorted(strat_rows, key=lambda row: int(row["k"]))
        xs = [int(row["k"]) for row in strat_rows]
        ys = [float(row[y_mean_key]) for row in strat_rows]
        sems = [float(row[y_sem_key]) for row in strat_rows]
        color = STRATEGY_COLOR.get(strategy_id, "#4D4D4D")
        marker = STRATEGY_MARKER.get(strategy_id, "o")
        linestyle = "--" if strategy_id == "random_outside_selected" else "-"
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.35,
            label=STRATEGY_LABEL.get(strategy_id, strategy_id),
        )
        if any(sems):
            xs_arr = np.asarray(xs, dtype=np.float64)
            ys_arr = np.asarray(ys, dtype=np.float64)
            sem_arr = np.asarray(sems, dtype=np.float64)
            ax.fill_between(xs_arr, ys_arr - sem_arr, ys_arr + sem_arr, color=color, alpha=0.12)
        any_data = True

    if not any_data:
        plt.close(fig)
        return

    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("k (number of heads)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(k_values))
    ax.legend(loc="best", ncol=2, frameon=True)
    stylize_axis(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320)
    plt.close(fig)


def _write_accuracy_markdown(path: Path, rows: Sequence[Mapping[str, object]], *, base_accuracy: float) -> None:
    lines = [
        f"Base accuracy (no intervention): {base_accuracy:.4f}",
        "",
        "| Strategy | k | Accuracy | Accuracy Drop | Trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        acc_mean = float(row["patched_accuracy_mean"])
        acc_sem = float(row["patched_accuracy_sem"])
        drop_mean = float(row["accuracy_drop_mean"])
        drop_sem = float(row["accuracy_drop_sem"])
        trials = int(row["n_trials"])
        if trials > 1:
            acc_text = f"{acc_mean:.4f} +/- {acc_sem:.4f}"
            drop_text = f"{drop_mean:.4f} +/- {drop_sem:.4f}"
        else:
            acc_text = f"{acc_mean:.4f}"
            drop_text = f"{drop_mean:.4f}"
        lines.append(
            f'| {row["strategy_label"]} | {int(row["k"])} | {acc_text} | {drop_text} | {trials} |'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step3(
    model_id: str,
    classify_json: Path,
    output_dir: Path,
    input_path: str = "",
    hop: str = "one_hop",
    prompt_order: str = "facts_first",
    prompt_style: str = "symbolic",
    max_samples: int = 400,
    accuracy_samples: int = 500,
    k_values: str = "1,2,4,8,16,32,64",
    late_layer_frac: float = 0.0,
    token_scope: str = "all_tokens",
    random_trials_min: int = 6,
    random_trials_max: int = 20,
    random_sem_target: float = 0.01,
    eval_batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
    model_source: str = "modelscope",
    progress_every: int = 10,
    save_plots: bool = True,
    include_signed_ratio_plot: bool = False,
) -> Dict[str, object]:
    del random_trials_min
    del random_sem_target

    logger = setup_file_logger(__name__, resolve_log_path(output_dir=output_dir, filename="step3_validate_fast.log"))
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = resolve_dual_correct_default(model_id, input_path)
    filtered_rows = filter_rows(
        input_path=dataset,
        hop=hop,
        prompt_order=prompt_order,
        max_samples=0,
        require_dual_correct=True,
        prompt_style=prompt_style,
    )
    if not filtered_rows:
        raise ValueError("No rows after filtering.")
    pd_rows = _select_balanced_rows(filtered_rows, max_samples=int(max_samples))
    acc_rows = _select_balanced_rows(filtered_rows, max_samples=int(accuracy_samples))
    selected_rows = _merge_selected_rows(pd_rows, acc_rows)
    if not selected_rows:
        raise ValueError("No selected validation rows after balanced sampling.")

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
    late_layer_set = set(late_layers)

    role_heads, selected_heads = _load_classified_heads(classify_json, late_layers=late_layer_set)
    if not role_heads:
        raise ValueError("No classified heads found in JSON for the selected late-layer range.")

    selected_head_set = set(selected_heads)
    outside_selected_pool_by_layer = {
        layer: [head for head in range(n_heads) if (layer, head) not in selected_head_set]
        for layer in late_layers
    }

    log_event(
        logger,
        {
            "stage": "step3_build_start",
            "filtered_rows": len(filtered_rows),
            "selected_rows": len(selected_rows),
            "pd_rows": len(pd_rows),
            "accuracy_rows": len(acc_rows),
            "late_layers": len(late_layers),
            "token_scope": token_scope,
        },
    )
    samples, layer_means = _build_validation_samples_and_layer_means(
        model=model,
        rows=selected_rows,
        prompt_style=prompt_style,
        late_layers=late_layers,
        true_id=true_id,
        false_id=false_id,
        token_scope=token_scope,
        progress_every=progress_every,
    )

    if len({sample.row_key for sample in samples}) != len(samples):
        raise RuntimeError("Duplicate validation row keys encountered while building samples.")
    sample_by_key = {sample.row_key: sample for sample in samples}
    pd_samples = _resolve_selected_samples(pd_rows, sample_by_key)
    acc_samples = _resolve_selected_samples(acc_rows, sample_by_key)
    if not pd_samples:
        raise RuntimeError("No PD validation samples available.")
    if not acc_samples:
        raise RuntimeError("No accuracy validation samples available.")

    ks = sorted({int(x.strip()) for x in k_values.split(",") if x.strip() and int(x.strip()) > 0})
    intervention_specs, random_trials_used = _build_intervention_specs(
        role_heads=role_heads,
        outside_selected_pool_by_layer=outside_selected_pool_by_layer,
        k_values=ks,
        random_trials=max(1, int(random_trials_max)),
        seed=seed,
    )
    if not intervention_specs:
        raise RuntimeError("No valid intervention conditions were constructed.")

    log_event(logger, {"stage": "step3_eval_pd_start", "conditions": len(intervention_specs), "samples": len(pd_samples)})
    pd_condition_rows = _eval_clean_interventions_many(
        model=model,
        samples=pd_samples,
        specs=intervention_specs,
        layer_means=layer_means,
        true_id=true_id,
        false_id=false_id,
        token_scope=token_scope,
        eval_batch_size=eval_batch_size,
        progress_every=progress_every,
        progress_tag="step3-pd",
    )

    log_event(logger, {"stage": "step3_eval_acc_start", "conditions": len(intervention_specs), "samples": len(acc_samples)})
    accuracy_condition_rows = _eval_clean_interventions_many(
        model=model,
        samples=acc_samples,
        specs=intervention_specs,
        layer_means=layer_means,
        true_id=true_id,
        false_id=false_id,
        token_scope=token_scope,
        eval_batch_size=eval_batch_size,
        progress_every=progress_every,
        progress_tag="step3-accuracy",
    )

    pd_curve_rows = _aggregate_metric_rows(pd_condition_rows)
    accuracy_table_rows = _aggregate_metric_rows(accuracy_condition_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "condition_level.csv", pd_condition_rows)
    write_csv(output_dir / "pd_condition_level.csv", pd_condition_rows)
    write_csv(output_dir / "accuracy_condition_level.csv", accuracy_condition_rows)
    write_csv(output_dir / "pd_curve_metrics.csv", pd_curve_rows)
    write_csv(output_dir / "accuracy_table.csv", accuracy_table_rows)
    _write_accuracy_markdown(
        output_dir / "accuracy_table.md",
        accuracy_table_rows,
        base_accuracy=float(accuracy_table_rows[0]["base_accuracy"]) if accuracy_table_rows else 0.0,
    )

    plots_dir = output_dir / "plots"
    abs_ratio_plot = plots_dir / "pd_abs_ratio_curve.png"
    signed_ratio_plot = plots_dir / "pd_signed_ratio_curve.png"
    dpd_shift_plot = plots_dir / "dpd_shift_curve.png"

    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_pd_curves(
            pd_curve_rows,
            abs_ratio_plot,
            y_mean_key="mean_abs_relative_dpd_mean",
            y_sem_key="mean_abs_relative_dpd_sem",
            ylabel="|dPD| / |PD_original|",
            k_values=ks,
        )
        if include_signed_ratio_plot:
            _plot_pd_curves(
                pd_curve_rows,
                signed_ratio_plot,
                y_mean_key="mean_signed_relative_dpd_mean",
                y_sem_key="mean_signed_relative_dpd_sem",
                ylabel="dPD / PD_original",
                k_values=ks,
            )
        _plot_pd_curves(
            pd_curve_rows,
            dpd_shift_plot,
            y_mean_key="mean_dpd_shift_mean",
            y_sem_key="mean_dpd_shift_sem",
            ylabel="dPD",
            k_values=ks,
        )

    base_pd_accuracy = sum(int(sample.clean_pred == sample.label) for sample in pd_samples) / len(pd_samples)
    base_acc_accuracy = sum(int(sample.clean_pred == sample.label) for sample in acc_samples) / len(acc_samples)
    summary = {
        "model_id": model_id,
        "hop": hop,
        "prompt_order": prompt_order,
        "prompt_style": prompt_style,
        "token_scope": token_scope,
        "dataset": dataset,
        "filtered_rows": len(filtered_rows),
        "filtered_rule_counts": count_by_field(filtered_rows, field="rule"),
        "selected_rows": len(selected_rows),
        "selected_rule_counts": count_by_field(selected_rows, field="rule"),
        "late_layer_frac": float(late_layer_frac),
        "late_layer_start": int(late_start),
        "late_layers_count": int(len(late_layers)),
        "k_values": ks,
        "pd_samples": len(pd_samples),
        "pd_sample_rule_counts": count_by_field(pd_samples, field="rule"),
        "pd_base_accuracy": float(base_pd_accuracy),
        "accuracy_samples": len(acc_samples),
        "accuracy_sample_rule_counts": count_by_field(acc_samples, field="rule"),
        "accuracy_base_accuracy": float(base_acc_accuracy),
        "n_conditions": len(intervention_specs),
        "n_strategies": len({spec.strategy_id for spec in intervention_specs}),
        "random_trials": int(max(1, random_trials_max)),
        "random_trials_used": random_trials_used,
        "ratio_eps": RATIO_EPS,
        "mean_ablation": {
            "token_scope": token_scope,
            "replacement": "per-layer, per-head mean over clean activations across the selected token positions",
        },
        "artifacts": {
            "pd_condition_level_csv": str(output_dir / "pd_condition_level.csv"),
            "pd_curve_metrics_csv": str(output_dir / "pd_curve_metrics.csv"),
            "accuracy_condition_level_csv": str(output_dir / "accuracy_condition_level.csv"),
            "accuracy_table_csv": str(output_dir / "accuracy_table.csv"),
            "accuracy_table_md": str(output_dir / "accuracy_table.md"),
        },
        "plots": {
            "pd_abs_ratio": str(abs_ratio_plot),
            "dpd_shift": str(dpd_shift_plot),
        },
        "plot_generated": bool(save_plots),
        "signed_ratio_plot_generated": bool(save_plots and include_signed_ratio_plot),
    }
    if include_signed_ratio_plot:
        summary["plots"]["pd_signed_ratio"] = str(signed_ratio_plot)
    write_json(output_dir / "summary.json", summary)
    log_event(logger, {"stage": "step3_done", "output_dir": str(output_dir), "summary": str(output_dir / "summary.json")})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3: clean-prompt PD/accuracy validation")
    parser.add_argument("--model_id", required=True)
    add_model_source_arg(parser)
    parser.add_argument("--classify_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--hop", default="one_hop")
    parser.add_argument("--prompt_order", default="facts_first")
    parser.add_argument("--prompt_style", default="symbolic")
    parser.add_argument("--max_samples", type=int, default=400, help="Balanced per-rule budget for PD curves.")
    parser.add_argument("--accuracy_samples", type=int, default=500, help="Balanced per-rule budget for the accuracy table.")
    parser.add_argument("--k_values", default="1,2,4,8,16,32,64")
    parser.add_argument("--late_layer_frac", type=float, default=0.0)
    parser.add_argument("--token_scope", default="all_tokens", choices=["query_only", "all_tokens"])
    parser.add_argument("--random_trials_min", type=int, default=6, help="Accepted for compatibility; not used in the refactored step3.")
    parser.add_argument("--random_trials_max", type=int, default=20, help="Number of random outside-selected trials per k.")
    parser.add_argument("--random_sem_target", type=float, default=0.01, help="Accepted for compatibility; not used in the refactored step3.")
    parser.add_argument("--eval_batch_size", type=int, default=8)
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
    parser.add_argument(
        "--include_signed_ratio_plot",
        "--include-signed-ratio-plot",
        dest="include_signed_ratio_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate pd_signed_ratio_curve.png. Disabled by default.",
    )
    args = parser.parse_args()

    run_step3(
        model_id=args.model_id,
        classify_json=args.classify_json,
        output_dir=args.output_dir,
        input_path=args.input,
        hop=args.hop,
        prompt_order=args.prompt_order,
        prompt_style=args.prompt_style,
        max_samples=args.max_samples,
        accuracy_samples=args.accuracy_samples,
        k_values=args.k_values,
        late_layer_frac=args.late_layer_frac,
        token_scope=args.token_scope,
        random_trials_min=args.random_trials_min,
        random_trials_max=args.random_trials_max,
        random_sem_target=args.random_sem_target,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        device=args.device,
        model_source=args.model_source,
        progress_every=args.progress_every,
        save_plots=args.save_plots,
        include_signed_ratio_plot=args.include_signed_ratio_plot,
    )


if __name__ == "__main__":
    main()
