from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from src.data.formatters import build_depth_prompt, resolve_prompt_ending, resolve_query_expr_text
from src.model_loading import add_model_source_arg, load_causal_lm
from src.progress import log_event, make_tqdm, resolve_log_path, setup_file_logger
from .io_utils import balanced_sample_by_rule, read_jsonl, write_jsonl


_BOOL_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def _parse_bool(text: str) -> Optional[bool]:
    m = _BOOL_RE.search(text)
    if not m:
        return None
    return m.group(1).lower() == "true"


def _has_chat_template(tokenizer) -> bool:
    return bool(
        getattr(tokenizer, "chat_template", None)
        or getattr(tokenizer, "default_chat_template", None)
    )


def _chat_encode(tokenizer, prompt: str, enable_thinking: Optional[bool] = None) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template") and _has_chat_template(tokenizer):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            # Qwen3 tokenizer supports this flag. For tokenizers that do not,
            # we gracefully fall back in the TypeError branch below.
            kwargs["enable_thinking"] = enable_thinking
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **kwargs,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            text = prompt
    else:
        text = prompt
    return tokenizer([text], return_tensors="pt")


def _resolve_pad_token_id(tokenizer) -> Optional[int]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, (list, tuple)):
        return int(eos_token_id[0]) if eos_token_id else None
    return int(eos_token_id)


def _predict_bool(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: Optional[bool] = None,
) -> Tuple[Optional[bool], str]:
    model_inputs = _chat_encode(tokenizer, prompt, enable_thinking=enable_thinking).to(model.device)
    do_sample = temperature > 0
    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    pad_token_id = _resolve_pad_token_id(tokenizer)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    generated = model.generate(**generate_kwargs)
    new_tokens = generated[0][len(model_inputs.input_ids[0]) :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_bool(text), text


def _to_cot_if_needed(prompt: str, mode: str) -> str:
    if mode != "cot":
        return prompt
    return prompt.replace(
        "Answer with one word only: True or False.",
        "Reason step by step, then end with one final word: True or False.",
    )


def _rebuild_prompt_from_row(row: Dict[str, object], prompt_style: str, kind: str, mode: str) -> str:
    if kind not in {"clean", "corrupted"}:
        raise ValueError(f"Unknown prompt kind {kind!r}")

    hop = str(row["hop"])

    if kind == "clean":
        facts = row.get("facts")
    else:
        facts = row.get("corrupted_facts")

    if facts is None:
        raise KeyError(
            f"Cannot rebuild {kind} prompt for sample id={row.get('id')} because "
            f"{'facts' if kind == 'clean' else 'corrupted_facts'} is missing."
        )

    derived_steps = None
    if hop == "two_hop":
        iv = str(row["intermediate_var"])
        ie_field = "intermediate_expr_symbolic" if prompt_style == "symbolic" else "intermediate_expr_semi_natural"
        derived_steps = [(iv, str(row[ie_field]))]
    prompt_order = str(row.get("prompt_order", "facts_first"))

    prompt_ending = resolve_prompt_ending(row)
    if mode == "cot":
        prompt_ending = "answer_suffix"

    return build_depth_prompt(
        hop=hop,
        facts=dict(facts),
        query_expr_text=resolve_query_expr_text(row, prompt_style=prompt_style, kind=kind),
        mode=mode,
        derived_steps=derived_steps,
        prompt_order=prompt_order,
        prompt_ending=prompt_ending,
    )


def _resolve_prompt(row: Dict[str, object], prompt_style: str, kind: str, mode: str) -> str:
    field = f"{kind}_prompt_{prompt_style}"
    prompt_ending = resolve_prompt_ending(row)
    if mode == "cot" and prompt_ending != "answer_suffix":
        return _rebuild_prompt_from_row(row, prompt_style=prompt_style, kind=kind, mode=mode)
    if field in row and row[field] is not None:
        return _to_cot_if_needed(str(row[field]), mode)
    return _rebuild_prompt_from_row(row, prompt_style=prompt_style, kind=kind, mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model inference on clean/corrupted prompts")
    parser.add_argument("--model_id", type=str, required=True)
    add_model_source_arg(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt_style", choices=["symbolic", "semi_natural"], default="symbolic")
    parser.add_argument("--mode", choices=["nocot", "cot"], default="nocot")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument(
        "--device_map",
        default="auto",
        help="HF device_map setting (auto/balanced/sequential). Use 'none' to disable sharding.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Explicit device to move model to when device_map is disabled (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 means evaluate all selected rows; if >0, use balanced per-rule sampling.",
    )
    parser.add_argument(
        "--enable_thinking",
        choices=["auto", "true", "false"],
        default="auto",
        help="Control tokenizer thinking mode if supported by model. "
        "auto => nocot uses false, cot uses true.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Print progress every N rows (0 to disable).",
    )
    args = parser.parse_args()
    logger = setup_file_logger(__name__, resolve_log_path(output_path=args.output))

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {args.dtype!r}. Use one of: {sorted(dtype_map)}")

    if args.enable_thinking == "auto":
        enable_thinking = (args.mode == "cot")
    else:
        enable_thinking = (args.enable_thinking == "true")

    device_map = args.device_map
    if args.device != "auto":
        device_map = "none"
    if device_map == "none":
        device_map = None

    try:
        resolved_model_id, tokenizer, model = load_causal_lm(
            args.model_id,
            source=args.model_source,
            torch_dtype=dtype_map[args.dtype],
            device_map=device_map,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model {args.model_id!r} from {args.model_source}. "
            f"You passed --model_id {args.model_id!r}. "
            "Please verify the repo name on the selected backend or pass a local model path."
        ) from exc

    if args.device != "auto":
        model = model.to(args.device)

    rows = read_jsonl(args.input)
    if args.max_samples > 0:
        rows = balanced_sample_by_rule(rows, max_samples=args.max_samples)

    total = len(rows)
    log_event(
        logger,
        {
            "stage": "inference_start",
            "input": str(args.input),
            "count": total,
            "prompt_style": args.prompt_style,
            "mode": args.mode,
            "model_id": args.model_id,
            "model_source": args.model_source,
            "resolved_model_id": resolved_model_id,
            "device_map": args.device_map,
            "device": args.device,
            "enable_thinking": enable_thinking,
        },
    )
    if total == 0:
        write_jsonl(args.output, [])
        log_event(logger, {"output": str(args.output), "count": 0, "note": "No rows selected after max_samples filter"})
        return

    out_rows = []

    start_ts = time.time()
    progress = make_tqdm(rows, total=total, desc="inference", leave=True, disable=total <= 1)
    for idx, row in enumerate(progress, start=1):
        clean_prompt = _resolve_prompt(row, prompt_style=args.prompt_style, kind="clean", mode=args.mode)
        corrupt_prompt = _resolve_prompt(row, prompt_style=args.prompt_style, kind="corrupted", mode=args.mode)

        pred_clean, raw_clean = _predict_bool(
            model,
            tokenizer,
            clean_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=enable_thinking,
        )
        pred_corrupt, raw_corrupt = _predict_bool(
            model,
            tokenizer,
            corrupt_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=enable_thinking,
        )

        label_clean = bool(row["label"])
        label_corrupt = bool(row["label_corrupted"])

        item = dict(row)
        item.update(
            {
                "pred_clean": pred_clean,
                "pred_corrupted": pred_corrupt,
                "raw_clean": raw_clean,
                "raw_corrupted": raw_corrupt,
                "correct_clean": pred_clean is not None and pred_clean == label_clean,
                "correct_corrupted": pred_corrupt is not None and pred_corrupt == label_corrupt,
                "eval_mode": args.mode,
                "prompt_style": args.prompt_style,
                "model_id": args.model_id,
                "model_source": args.model_source,
                "resolved_model_id": resolved_model_id,
            }
        )
        out_rows.append(item)

        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == total):
            elapsed = max(time.time() - start_ts, 1e-6)
            speed = idx / elapsed
            eta = (total - idx) / speed if speed > 0 else 0.0
            progress.set_postfix(done=idx, eta_sec=round(eta, 1))
            log_event(
                logger,
                {
                    "stage": "inference_progress",
                    "done": idx,
                    "total": total,
                    "pct": round(100.0 * idx / total, 2),
                    "elapsed_sec": round(elapsed, 1),
                    "speed_rows_per_sec": round(speed, 3),
                    "eta_sec": round(eta, 1),
                },
            )

    write_jsonl(args.output, out_rows)
    total_elapsed = time.time() - start_ts
    log_event(
        logger,
        {
            "output": str(args.output),
            "count": len(out_rows),
            "elapsed_sec": round(total_elapsed, 1),
            "avg_rows_per_sec": round(len(out_rows) / max(total_elapsed, 1e-6), 3),
        },
    )


if __name__ == "__main__":
    main()
