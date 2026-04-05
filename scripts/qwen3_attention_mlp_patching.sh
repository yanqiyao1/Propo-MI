#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
REGION_DEVICE="${REGION_GPU:-cuda:1}"
REGION_MAX_SAMPLES="${REGION_MAX_SAMPLES:-1000}"

plot_region_scores_for_hop_dir() {
  local hop_dir="$1"
  for region in facts_region expression_region query_region; do
    python -m src.attn_mlp_analysis.plot_region_scores \
      --scores_csv "${hop_dir}/${region}/attn_mlp_${region}_scores.csv" \
      --output_png "${hop_dir}/${region}/attn_mlp_${region}_scores.png"
  done
}

plot_experiment() {
  local one_hop_dir="$1"
  local two_hop_dir="$2"
  local comparison_dir="$3"

  plot_region_scores_for_hop_dir "${one_hop_dir}"
  plot_region_scores_for_hop_dir "${two_hop_dir}"
  python -m src.attn_mlp_analysis.plot_band_metrics \
    --one_hop_dir "${one_hop_dir}" \
    --two_hop_dir "${two_hop_dir}" \
    --output_dir "${comparison_dir}"
}

bash scripts/ATTN_MLP_region_patching.sh \
  --model_id "Qwen/Qwen3-14B" \
  --model_source "${MODEL_SOURCE}" \
  --prompt_style "symbolic" \
  --device "${REGION_DEVICE}" \
  --max_samples "${REGION_MAX_SAMPLES}" \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts/filtered_dual_correct_Qwen3-14b.jsonl" \
  --one_hop_input "dataset/attn_mlp_14b_one_hop.jsonl" \
  --two_hop_input "dataset/attn_mlp_14b_two_hop.jsonl" \
  --split_summary "reports/attn_mlp_analysis/14b_dual_correct_split.json" \
  --output_root "reports/attn_mlp_analysis" \
  --auto_split 1 \
  --make_plots 0 \
  --plot_output_dir "reports/attn_mlp_analysis/comparison_14B"

plot_experiment \
  "reports/attn_mlp_analysis/attn_mlp_14b_one_hop" \
  "reports/attn_mlp_analysis/attn_mlp_14b_two_hop" \
  "reports/attn_mlp_analysis/comparison_14B"

bash scripts/ATTN_MLP_region_patching.sh \
  --model_id "Qwen/Qwen3-14B" \
  --model_source "${MODEL_SOURCE}" \
  --prompt_style "symbolic" \
  --device "${REGION_DEVICE}" \
  --max_samples "${REGION_MAX_SAMPLES}" \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts/filtered_dual_correct_Qwen3-14b_expr_first.jsonl" \
  --one_hop_input "dataset/attn_mlp_14b_one_hop_expr_first.jsonl" \
  --two_hop_input "dataset/attn_mlp_14b_two_hop_expr_first.jsonl" \
  --split_summary "reports/attn_mlp_analysis/14b_dual_correct_split_expr_first.json" \
  --output_root "reports/attn_mlp_analysis_expr_first" \
  --auto_split 1 \
  --make_plots 0 \
  --plot_output_dir "reports/attn_mlp_analysis_expr_first/comparison_expr_first_14B"

plot_experiment \
  "reports/attn_mlp_analysis_expr_first/attn_mlp_14b_one_hop_expr_first" \
  "reports/attn_mlp_analysis_expr_first/attn_mlp_14b_two_hop_expr_first" \
  "reports/attn_mlp_analysis_expr_first/comparison_expr_first_14B"

bash scripts/ATTN_MLP_region_patching.sh \
  --model_id "Qwen/Qwen3-8B" \
  --model_source "${MODEL_SOURCE}" \
  --prompt_style "symbolic" \
  --device "${REGION_DEVICE}" \
  --max_samples "${REGION_MAX_SAMPLES}" \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts/filtered_dual_correct_Qwen3-8b.jsonl" \
  --one_hop_input "dataset/attn_mlp_8b_one_hop.jsonl" \
  --two_hop_input "dataset/attn_mlp_8b_two_hop.jsonl" \
  --split_summary "reports/attn_mlp_analysis/8b_dual_correct_split.json" \
  --output_root "reports/attn_mlp_analysis" \
  --auto_split 1 \
  --make_plots 0 \
  --plot_output_dir "reports/attn_mlp_analysis/comparison"

plot_experiment \
  "reports/attn_mlp_analysis/attn_mlp_8b_one_hop" \
  "reports/attn_mlp_analysis/attn_mlp_8b_two_hop" \
  "reports/attn_mlp_analysis/comparison"

bash scripts/ATTN_MLP_region_patching.sh \
  --model_id "Qwen/Qwen3-8B" \
  --model_source "${MODEL_SOURCE}" \
  --prompt_style "symbolic" \
  --device "${REGION_DEVICE}" \
  --max_samples "${REGION_MAX_SAMPLES}" \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl" \
  --one_hop_input "dataset/attn_mlp_8b_one_hop_expr_first.jsonl" \
  --two_hop_input "dataset/attn_mlp_8b_two_hop_expr_first.jsonl" \
  --split_summary "reports/attn_mlp_analysis/8b_dual_correct_split_expr_first.json" \
  --output_root "reports/attn_mlp_analysis_expr_first" \
  --auto_split 1 \
  --make_plots 0 \
  --plot_output_dir "reports/attn_mlp_analysis_expr_first/comparison_expr_first"

plot_experiment \
  "reports/attn_mlp_analysis_expr_first/attn_mlp_8b_one_hop_expr_first" \
  "reports/attn_mlp_analysis_expr_first/attn_mlp_8b_two_hop_expr_first" \
  "reports/attn_mlp_analysis_expr_first/comparison_expr_first"
