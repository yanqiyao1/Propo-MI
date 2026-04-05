#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_ID="${MODEL_ID:?MODEL_ID must be set}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
PROMPT_STYLE="${PROMPT_STYLE:-symbolic}"
ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
REPORT_DIR="${REPORT_DIR:?REPORT_DIR must be set}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR must be set}"
ATTN_MLP_DEVICE="${ATTN_MLP_DEVICE:-cuda:1}"
ATTN_MLP_MAX_SAMPLES="${ATTN_MLP_MAX_SAMPLES:-1000}"
ATTN_MLP_PROGRESS_EVERY="${ATTN_MLP_PROGRESS_EVERY:-100}"

mkdir -p "${ARTIFACT_DIR}" "${REPORT_DIR}" "${DATASET_DIR}"

run_attn_mlp() {
  local prompt_tag="$1"
  local input_path="$2"
  local one_hop_input="$3"
  local two_hop_input="$4"
  local split_summary="$5"
  local output_root="$6"
  local comparison_dir="$7"

  echo "[attn-mlp] prompt_tag=${prompt_tag} output_root=${output_root}"
  bash scripts/ATTN_MLP_region_patching.sh \
    --model_id "${MODEL_ID}" \
    --model_source "${MODEL_SOURCE}" \
    --prompt_style "${PROMPT_STYLE}" \
    --device "${ATTN_MLP_DEVICE}" \
    --max_samples "${ATTN_MLP_MAX_SAMPLES}" \
    --progress_every "${ATTN_MLP_PROGRESS_EVERY}" \
    --split all \
    --source_input "${input_path}" \
    --one_hop_input "${one_hop_input}" \
    --two_hop_input "${two_hop_input}" \
    --split_summary "${split_summary}" \
    --output_root "${output_root}" \
    --auto_split 1 \
    --make_plots 1 \
    --plot_output_dir "${comparison_dir}"
}

run_attn_mlp \
  "facts_first" \
  "${ARTIFACT_DIR}/filtered_dual_correct_facts_first.jsonl" \
  "${DATASET_DIR}/attn_mlp_one_hop_facts_first.jsonl" \
  "${DATASET_DIR}/attn_mlp_two_hop_facts_first.jsonl" \
  "${REPORT_DIR}/attn_mlp_analysis_facts_first/dual_correct_split.json" \
  "${REPORT_DIR}/attn_mlp_analysis_facts_first" \
  "${REPORT_DIR}/attn_mlp_analysis_facts_first/comparison"

run_attn_mlp \
  "expr_first" \
  "${ARTIFACT_DIR}/filtered_dual_correct_expr_first.jsonl" \
  "${DATASET_DIR}/attn_mlp_one_hop_expr_first.jsonl" \
  "${DATASET_DIR}/attn_mlp_two_hop_expr_first.jsonl" \
  "${REPORT_DIR}/attn_mlp_analysis_expr_first/dual_correct_split.json" \
  "${REPORT_DIR}/attn_mlp_analysis_expr_first" \
  "${REPORT_DIR}/attn_mlp_analysis_expr_first/comparison"

echo "[done] attention+mlp patching completed for ${MODEL_ID}"
