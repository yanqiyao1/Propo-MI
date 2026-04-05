#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:0}"

python -m src.eval.inference \
  --model_id Qwen/Qwen3-8B \
  --model_source "${MODEL_SOURCE}" \
  --input dataset/proplogic_mi.jsonl \
  --output artifacts/preds_Qwen3-8b.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device "${INFERENCE_DEVICE}"

python -m src.eval.inference \
  --model_id Qwen/Qwen3-14B \
  --model_source "${MODEL_SOURCE}" \
  --input dataset/proplogic_mi.jsonl \
  --output artifacts/preds_Qwen3-14b.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device "${INFERENCE_DEVICE}"

python -m src.eval.inference \
  --model_id Qwen/Qwen3-8B \
  --model_source "${MODEL_SOURCE}" \
  --input dataset/proplogic_mi_expr_first.jsonl \
  --output artifacts/preds_Qwen3-8b_expr_first.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device "${INFERENCE_DEVICE}"

python -m src.eval.inference \
  --model_id Qwen/Qwen3-14B \
  --model_source "${MODEL_SOURCE}" \
  --input dataset/proplogic_mi_expr_first.jsonl \
  --output artifacts/preds_Qwen3-14b_expr_first.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device "${INFERENCE_DEVICE}"

python -m src.eval.filtering \
  --input artifacts/preds_Qwen3-8b.jsonl \
  --output artifacts/filtered_dual_correct_Qwen3-8b.jsonl

python -m src.eval.filtering \
  --input artifacts/preds_Qwen3-14b.jsonl \
  --output artifacts/filtered_dual_correct_Qwen3-14b.jsonl

python -m src.eval.filtering \
  --input artifacts/preds_Qwen3-8b_expr_first.jsonl \
  --output artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl

python -m src.eval.filtering \
  --input artifacts/preds_Qwen3-14b_expr_first.jsonl \
  --output artifacts/filtered_dual_correct_Qwen3-14b_expr_first.jsonl

python -m src.eval.metrics \
  --input artifacts/preds_Qwen3-8b.jsonl \
  --output reports/metrics_Qwen3-8b.json

python -m src.eval.metrics \
  --input artifacts/preds_Qwen3-14b.jsonl \
  --output reports/metrics_Qwen3-14b.json

python -m src.eval.metrics \
  --input artifacts/preds_Qwen3-8b_expr_first.jsonl \
  --output reports/metrics_Qwen3-8b_expr_first.json

python -m src.eval.metrics \
  --input artifacts/preds_Qwen3-14b_expr_first.jsonl \
  --output reports/metrics_Qwen3-14b_expr_first.json
