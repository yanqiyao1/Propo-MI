#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
TOKEN_DEVICE="${TOKEN_GPU:-cuda:0}"
TOKEN_MAX_SAMPLES="${TOKEN_MAX_SAMPLES:-1000}"
TOKEN_PROGRESS_EVERY="${TOKEN_PROGRESS_EVERY:-10}"
SAVE_RAW_PLOTS="${SAVE_RAW_PLOTS:-0}"
SAVE_REFINED_PLOTS="${SAVE_REFINED_PLOTS:-0}"

raw_summary_is_current() {
  local summary_json="$1"
  local results_pkl="$2"

  if [[ ! -f "${summary_json}" || ! -f "${results_pkl}" ]]; then
    return 1
  fi

  if [[ "${TOKEN_MAX_SAMPLES}" == "0" ]]; then
    return 0
  fi

  "${PYTHON_BIN}" - "${summary_json}" "${TOKEN_MAX_SAMPLES}" <<'PY'
import json
import sys

summary_path = sys.argv[1]
max_samples = int(sys.argv[2])

try:
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    selected_rows = int(payload.get("selected_rows", -1))
except Exception:
    raise SystemExit(1)

raise SystemExit(0 if 0 < selected_rows <= max_samples else 1)
PY
}

refined_outputs_exist() {
  local refined_dir="$1"
  [[ -f "${refined_dir}/refined_stats_sum.json" && -f "${refined_dir}/refined_stats_mean.json" ]]
}

run_raw() {
  local model_id="$1"
  local input_path="$2"
  local output_dir="$3"
  local early_end="$4"
  local middle_end="$5"
  local plot_flag="--no-save_plots"
  local summary_json="${output_dir}/summary.json"
  local results_pkl="${output_dir}/patching_results.pkl"

  if [[ "${SAVE_RAW_PLOTS}" == "1" ]]; then
    plot_flag="--save_plots"
  fi

  mkdir -p "${output_dir}"
  if raw_summary_is_current "${summary_json}" "${results_pkl}"; then
    echo "[skip token-raw] output=${output_dir}"
    return
  fi

  echo "[token-raw] input=${input_path} output=${output_dir}"
  "${PYTHON_BIN}" -m src.token_analysis.activation_patching_dataset \
    --model_id "${model_id}" \
    --model_source "${MODEL_SOURCE}" \
    --input "${input_path}" \
    --output_dir "${output_dir}" \
    --prompt_style symbolic \
    --hop all \
    --max_samples "${TOKEN_MAX_SAMPLES}" \
    --require_dual_correct \
    --strict_length_match \
    --early_end "${early_end}" \
    --middle_end "${middle_end}" \
    --device "${TOKEN_DEVICE}" \
    --progress_every "${TOKEN_PROGRESS_EVERY}" \
    "${plot_flag}"
}

run_simple_plot() {
  local raw_dir="$1"
  local pkl_path="${raw_dir}/patching_results.pkl"
  local summary_json="${raw_dir}/summary.json"
  local comparison_png="${raw_dir}/category_comparison_simple.png"
  local layer_stage_png="${raw_dir}/layer_stage_simple.png"
  local heatmap_png="${raw_dir}/heatmap_simple.png"

  if [[ -f "${comparison_png}" && -f "${layer_stage_png}" && -f "${heatmap_png}" && ! "${pkl_path}" -nt "${comparison_png}" && ! "${summary_json}" -nt "${comparison_png}" ]]; then
    echo "[skip token-plot] output=${raw_dir}"
    return
  fi

  "${PYTHON_BIN}" -m src.token_analysis.plot_simple_analysis \
    --input_pkl "${raw_dir}/patching_results.pkl" \
    --summary_json "${raw_dir}/summary.json" \
    --output_dir "${raw_dir}"
}

run_refined() {
  local raw_dir="$1"
  local refined_dir="$2"
  local title="$3"
  local early_end="$4"
  local middle_end="$5"
  local n_layers="$6"
  local plot_flag="--no-save-plots"
  local raw_pkl="${raw_dir}/patching_results.pkl"
  local sum_json="${refined_dir}/refined_stats_sum.json"
  local mean_json="${refined_dir}/refined_stats_mean.json"

  if [[ "${SAVE_REFINED_PLOTS}" == "1" ]]; then
    plot_flag="--save-plots"
  fi

  mkdir -p "${refined_dir}"
  if refined_outputs_exist "${refined_dir}" && [[ ! "${raw_pkl}" -nt "${sum_json}" && ! "${raw_pkl}" -nt "${mean_json}" ]]; then
    echo "[skip token-refined] output=${refined_dir}"
    return
  fi

  echo "[token-refined] output=${refined_dir}"
  "${PYTHON_BIN}" -m src.token_analysis.refined_token_analysis \
    --input_pkl "${raw_dir}/patching_results.pkl" \
    --output_dir "${refined_dir}" \
    --title "${title}" \
    --early_end "${early_end}" \
    --middle_end "${middle_end}" \
    --n_layers "${n_layers}" \
    --include-derived-assignment \
    "${plot_flag}" \
    --save-csv
}

run_refined_plot() {
  local refined_dir="$1"
  local sum_json="${refined_dir}/refined_stats_sum.json"
  local mean_json="${refined_dir}/refined_stats_mean.json"
  local sum_png="${refined_dir}/refined_by_stage_sum.png"
  local mean_png="${refined_dir}/refined_by_stage_mean.png"

  if [[ -f "${sum_png}" && -f "${mean_png}" && ! "${sum_json}" -nt "${sum_png}" && ! "${mean_json}" -nt "${mean_png}" ]]; then
    echo "[skip token-refined-plot] output=${refined_dir}"
    return
  fi

  "${PYTHON_BIN}" -m src.token_analysis.plot_refined_analysis \
    --sum_stats_json "${refined_dir}/refined_stats_sum.json" \
    --mean_stats_json "${refined_dir}/refined_stats_mean.json" \
    --output_dir "${refined_dir}"
}

run_suite() {
  local model_id="$1"
  local facts_input="$2"
  local facts_raw_dir="$3"
  local expr_input="$4"
  local expr_raw_dir="$5"
  local facts_refined_dir="$6"
  local expr_refined_dir="$7"
  local title_prefix="$8"
  local early_end="$9"
  local middle_end="${10}"
  local n_layers="${11}"

  run_raw "${model_id}" "${facts_input}" "${facts_raw_dir}" "${early_end}" "${middle_end}"
  run_raw "${model_id}" "${expr_input}" "${expr_raw_dir}" "${early_end}" "${middle_end}"

  run_simple_plot "${facts_raw_dir}"
  run_simple_plot "${expr_raw_dir}"

  run_refined "${facts_raw_dir}" "${facts_refined_dir}" "${title_prefix} Facts-first All-hop Refined Token Analysis" "${early_end}" "${middle_end}" "${n_layers}"
  run_refined "${expr_raw_dir}" "${expr_refined_dir}" "${title_prefix} Expr-first All-hop Refined Token Analysis" "${early_end}" "${middle_end}" "${n_layers}"

  run_refined_plot "${facts_refined_dir}"
  run_refined_plot "${expr_refined_dir}"
}

run_suite \
  "Qwen/Qwen3-14B" \
  "artifacts/filtered_dual_correct_Qwen3-14b.jsonl" \
  "reports/token_analysis/14b_facts_first_all_hop_raw" \
  "artifacts/filtered_dual_correct_Qwen3-14b_expr_first.jsonl" \
  "reports/token_analysis/14b_expr_first_all_hop_raw" \
  "reports/token_analysis/14b_facts_first_all_hop_refined" \
  "reports/token_analysis/14b_expr_first_all_hop_refined" \
  "Qwen3-14B" \
  14 \
  24 \
  40

run_suite \
  "Qwen/Qwen3-8B" \
  "artifacts/filtered_dual_correct_Qwen3-8b.jsonl" \
  "reports/token_analysis/8b_facts_first_all_hop_raw" \
  "artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl" \
  "reports/token_analysis/8b_expr_first_all_hop_raw" \
  "reports/token_analysis/8b_facts_first_all_hop_refined" \
  "reports/token_analysis/8b_expr_first_all_hop_refined" \
  "Qwen3-8B" \
  14 \
  24 \
  36
