from __future__ import annotations

from typing import Dict, Mapping, Sequence


def _argmax(values: Sequence[float]) -> int | None:
    if not values:
        return None
    return int(max(range(len(values)), key=values.__getitem__))


def _argmin(values: Sequence[float]) -> int | None:
    if not values:
        return None
    return int(min(range(len(values)), key=values.__getitem__))


def summarize_sample_layers(layer_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not layer_rows:
        return {
            "sample_peak_positive_dpd_layer": None,
            "sample_peak_positive_dpd": 0.0,
            "sample_peak_negative_dpd_layer": None,
            "sample_peak_negative_dpd": 0.0,
            "sample_mean_dpd": 0.0,
            "sample_mean_abs_dpd": 0.0,
        }

    dpds = [float(row["dpd"]) for row in layer_rows]

    peak_positive_dpd_layer = _argmax(dpds)
    peak_negative_dpd_layer = _argmin(dpds)

    return {
        "sample_peak_positive_dpd_layer": peak_positive_dpd_layer,
        "sample_peak_positive_dpd": float(dpds[peak_positive_dpd_layer]) if peak_positive_dpd_layer is not None else 0.0,
        "sample_peak_negative_dpd_layer": peak_negative_dpd_layer,
        "sample_peak_negative_dpd": float(dpds[peak_negative_dpd_layer]) if peak_negative_dpd_layer is not None else 0.0,
        "sample_mean_dpd": float(sum(dpds) / len(dpds)),
        "sample_mean_abs_dpd": float(sum(abs(value) for value in dpds) / len(dpds)),
    }


def build_sample_detail_record(
    *,
    sample_index: int,
    row: Mapping[str, object],
    region_mode: str,
    region_selection_strategy: str,
    region_selection_error: str | None,
    prompt: str,
    model_input_prompt: str,
    model_input_seq_len: int,
    region_indices: Sequence[int],
    region_tokens: Sequence[object],
    last_input_token: object,
    base_margin: float,
    base_selected_token_prob: float,
    sample_metrics: Mapping[str, object],
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "sample_index": int(sample_index),
        "row_id": row.get("id"),
        "rule": str(row.get("rule", "")),
        "hop": str(row.get("hop", "")),
        "prompt_order": str(row.get("prompt_order", "facts_first")),
        "label": row.get("label"),
        "region_mode": region_mode,
        "region_selection_strategy": region_selection_strategy,
        "region_selection_error": region_selection_error,
        "raw_prompt_char_len": len(prompt),
        "model_input_prompt_char_len": len(model_input_prompt),
        "extra_model_input_char_len": max(0, len(model_input_prompt) - len(prompt)),
        "model_input_seq_len": int(model_input_seq_len),
        "last_input_token_index": int(model_input_seq_len - 1) if model_input_seq_len > 0 else None,
        "last_input_token": str(last_input_token),
        "region_indices": [int(idx) for idx in region_indices],
        "region_size": int(len(region_indices)),
        "region_tokens": [str(tok) for tok in region_tokens],
        "base_margin": float(base_margin),
        "base_selected_token_prob": float(base_selected_token_prob),
    }
    record.update(sample_metrics)
    return record


def build_skipped_sample_record(
    *,
    sample_index: int,
    row: Mapping[str, object],
    region_mode: str,
    stage: str,
    reason: str,
    region_selection_strategy: str,
    region_selection_error: str | None,
    prompt: str,
    model_input_prompt: str,
    model_input_seq_len: int,
    region_indices: Sequence[int],
) -> Dict[str, object]:
    return {
        "sample_index": int(sample_index),
        "row_id": row.get("id"),
        "rule": str(row.get("rule", "")),
        "hop": str(row.get("hop", "")),
        "prompt_order": str(row.get("prompt_order", "facts_first")),
        "label": row.get("label"),
        "region_mode": region_mode,
        "stage": stage,
        "reason": reason,
        "region_selection_strategy": region_selection_strategy,
        "region_selection_error": region_selection_error,
        "raw_prompt_char_len": len(prompt),
        "model_input_prompt_char_len": len(model_input_prompt),
        "extra_model_input_char_len": max(0, len(model_input_prompt) - len(prompt)),
        "model_input_seq_len": int(model_input_seq_len),
        "region_indices": [int(idx) for idx in region_indices],
        "region_size": int(len(region_indices)),
    }
