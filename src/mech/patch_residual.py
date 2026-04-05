from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
from transformer_lens import utils

from src.eval.io_utils import balanced_sample_by_rule, read_jsonl
from src.model_loading import add_model_source_arg
from src.progress import log_event, make_tqdm, resolve_log_path, setup_file_logger
from .dld import compute_correct_incorrect_prob_diff
from .tl_utils import load_hooked_transformer, resolve_true_false_token_ids, to_tokens


def _ensure_same_length(clean_tokens: torch.Tensor, corrupt_tokens: torch.Tensor) -> int:
    return min(clean_tokens.shape[1], corrupt_tokens.shape[1]) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual stream activation patching scan")
    parser.add_argument("--model_id", required=True)
    add_model_source_arg(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt_style", choices=["symbolic", "semi_natural"], default="symbolic")
    parser.add_argument("--max_samples", type=int, default=64, help="Balanced per-rule sample budget.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch_position", choices=["query"], default="query")
    args = parser.parse_args()
    logger = setup_file_logger(__name__, resolve_log_path(output_path=args.output))

    model = load_hooked_transformer(
        args.model_id,
        device=args.device,
        source=args.model_source,
        error_context="mechanistic patching modules",
    )
    true_id, false_id = resolve_true_false_token_ids(model)
    rows = balanced_sample_by_rule(read_jsonl(args.input), max_samples=args.max_samples)

    output_rows: List[Dict[str, object]] = []
    progress = make_tqdm(total=len(rows) * int(model.cfg.n_layers), desc="residual-patching", leave=True, disable=len(rows) == 0)

    for row in rows:
        clean_prompt = str(row[f"clean_prompt_{args.prompt_style}"])
        corrupt_prompt = str(row[f"corrupted_prompt_{args.prompt_style}"])

        clean_tokens = to_tokens(model, clean_prompt)
        corrupt_tokens = to_tokens(model, corrupt_prompt)

        patch_pos = _ensure_same_length(clean_tokens, corrupt_tokens)
        corrupt_label = bool(row.get("label_corrupted", row.get("label", False)))

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            corrupt_logits, _ = model.run_with_cache(corrupt_tokens)

            corrupt_base = compute_correct_incorrect_prob_diff(
                corrupt_logits, corrupt_label, true_id, false_id, pos=-1
            ).item()

            for layer in range(model.cfg.n_layers):
                act_name = utils.get_act_name("resid_pre", layer)

                def hook_fn(resid, hook, layer_name=act_name):
                    del hook
                    clean_resid = clean_cache[layer_name]
                    resid[:, patch_pos, :] = clean_resid[:, patch_pos, :]
                    return resid

                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(act_name, hook_fn)])
                patched_val = compute_correct_incorrect_prob_diff(
                    patched_logits, corrupt_label, true_id, false_id, pos=-1
                ).item()

                output_rows.append(
                    {
                        "id": row["id"],
                        "rule": row["rule"],
                        "hop": row["hop"],
                        "layer": layer,
                        "patch_pos": patch_pos,
                        "corrupt_prob_diff": corrupt_base,
                        "patched_prob_diff": patched_val,
                        "dpd_shift": patched_val - corrupt_base,
                    }
                )
                progress.update(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()) if output_rows else [])
        if output_rows:
            writer.writeheader()
            writer.writerows(output_rows)

    progress.close()
    log_event(logger, {"output": str(args.output), "rows": len(output_rows)})


if __name__ == "__main__":
    main()
