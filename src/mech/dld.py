from __future__ import annotations

from typing import Tuple

import torch


def _pair_probs_from_logits(
    logits: torch.Tensor,
    token_a_id: int,
    token_b_id: int,
    pos: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute binary probabilities over two target tokens only:
    p(a) = exp(logit_a) / (exp(logit_a) + exp(logit_b))
    p(b) = exp(logit_b) / (exp(logit_a) + exp(logit_b))
    """
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


def compute_logit_diff(logits: torch.Tensor, true_token_id: int, false_token_id: int, pos: int = -1) -> torch.Tensor:
    """
    Backward-compatible name.
    NOTE: this now returns probability difference (p_true - p_false) computed
    with a 2-way softmax over the True/False logits.
    """
    return compute_prob_diff(logits, true_token_id, false_token_id, pos=pos)


def compute_dpd_shift(
    base_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    true_token_id: int,
    false_token_id: int,
    label: bool | torch.Tensor | None = None,
    pos: int = -1,
) -> torch.Tensor:
    if label is None:
        base = compute_prob_diff(base_logits, true_token_id, false_token_id, pos=pos)
        patched = compute_prob_diff(patched_logits, true_token_id, false_token_id, pos=pos)
    else:
        base = compute_correct_incorrect_prob_diff(base_logits, label, true_token_id, false_token_id, pos=pos)
        patched = compute_correct_incorrect_prob_diff(patched_logits, label, true_token_id, false_token_id, pos=pos)
    return patched - base


def compute_dld_shift(
    base_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    true_token_id: int,
    false_token_id: int,
    label: bool | torch.Tensor | None = None,
    pos: int = -1,
) -> torch.Tensor:
    """
    Backward-compatible alias for compute_dpd_shift.
    """
    return compute_dpd_shift(
        base_logits=base_logits,
        patched_logits=patched_logits,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        label=label,
        pos=pos,
    )


def bool_prediction_from_logits(
    logits: torch.Tensor,
    true_token_id: int,
    false_token_id: int,
    pos: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_prob, false_prob = _pair_probs_from_logits(logits, true_token_id, false_token_id, pos=pos)
    pred = true_prob > false_prob
    return pred, true_prob, false_prob
