#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/MLP_region_patching.sh \
    --model_id "Qwen/Qwen3-14B" \
    --model_source "huggingface" \
    --prompt_style "symbolic" \
    --device "cuda:1" \
    --max_samples 3000 \
    --progress_every 100 \
    --split "all" \
    --source_input "artifacts/filtered_dual_correct_Qwen3-14b_expr_first.jsonl" \
    --one_hop_input "dataset/mlp_14b_one_hop_expr_first.jsonl" \
    --two_hop_input "dataset/mlp_14b_two_hop_expr_first.jsonl" \
    --split_summary "reports/mlp_analysis/14b_dual_correct_split_expr_first.json" \
    --output_root "reports/mlp_analysis_expr_first" \
    --auto_split 1 \
    --make_plots 1 \
    --plot_output_dir "reports/mlp_analysis_expr_first/comparison_expr_first_14B"

bash scripts/MLP_region_patching.sh \
    --model_id "Qwen/Qwen3-8B" \
    --model_source "huggingface" \
    --prompt_style "symbolic" \
    --device "cuda:1" \
    --max_samples 3000 \
    --progress_every 100 \
    --split "all" \
    --source_input "artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl" \
    --one_hop_input "dataset/mlp_8b_one_hop_expr_first.jsonl" \
    --two_hop_input "dataset/mlp_8b_two_hop_expr_first.jsonl" \
    --split_summary "reports/mlp_analysis/8b_dual_correct_split_expr_first.json" \
    --output_root "reports/mlp_analysis_expr_first" \
    --auto_split 1 \
    --make_plots 1 \
    --plot_output_dir "reports/mlp_analysis_expr_first/comparison_expr_first"
