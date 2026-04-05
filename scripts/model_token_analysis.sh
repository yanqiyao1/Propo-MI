#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:?MODEL_ID must be set}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
REPORT_DIR="${REPORT_DIR:?REPORT_DIR must be set}"
PROMPT_STYLE="${PROMPT_STYLE:-symbolic}"
TOKEN_DEVICE="${TOKEN_DEVICE:-cuda:1}"
TOKEN_MAX_SAMPLES="${TOKEN_MAX_SAMPLES:-1000}"
TOKEN_PROGRESS_EVERY="${TOKEN_PROGRESS_EVERY:-10}"
EARLY_END="${EARLY_END:-12}"
MIDDLE_END="${MIDDLE_END:-22}"
N_LAYERS="${N_LAYERS:?N_LAYERS must be set}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL must be set}"
SAVE_RAW_PLOTS="${SAVE_RAW_PLOTS:-0}"
SAVE_REFINED_PLOTS="${SAVE_REFINED_PLOTS:-0}"

mkdir -p "${ARTIFACT_DIR}" "${REPORT_DIR}"

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
  local input_path="$1"
  local output_dir="$2"
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
  "${PYTHON_BIN}" -m src.token_analysis.activation_patching_dataset     --model_id "${MODEL_ID}"     --model_source "${MODEL_SOURCE}"     --input "${input_path}"     --output_dir "${output_dir}"     --prompt_style "${PROMPT_STYLE}"     --hop all     --max_samples "${TOKEN_MAX_SAMPLES}"     --require_dual_correct     --strict_length_match     --early_end "${EARLY_END}"     --middle_end "${MIDDLE_END}"     --device "${TOKEN_DEVICE}"     --progress_every "${TOKEN_PROGRESS_EVERY}"     "${plot_flag}"
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

  "${PYTHON_BIN}" -m src.token_analysis.plot_simple_analysis     --input_pkl "${raw_dir}/patching_results.pkl"     --summary_json "${raw_dir}/summary.json"     --output_dir "${raw_dir}"
}

run_refined() {
  local raw_dir="$1"
  local refined_dir="$2"
  local title="$3"
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
  "${PYTHON_BIN}" -m src.token_analysis.refined_token_analysis     --input_pkl "${raw_dir}/patching_results.pkl"     --output_dir "${refined_dir}"     --title "${title}"     --early_end "${EARLY_END}"     --middle_end "${MIDDLE_END}"     --n_layers "${N_LAYERS}"     --include-derived-assignment     "${plot_flag}"     --save-csv
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

  "${PYTHON_BIN}" -m src.token_analysis.plot_refined_analysis     --sum_stats_json "${refined_dir}/refined_stats_sum.json"     --mean_stats_json "${refined_dir}/refined_stats_mean.json"     --output_dir "${refined_dir}"
}

FACTS_RAW_DIR="${REPORT_DIR}/token_analysis_facts_first_raw"
EXPR_RAW_DIR="${REPORT_DIR}/token_analysis_expr_first_raw"
FACTS_REFINED_DIR="${REPORT_DIR}/token_analysis_facts_first_refined"
EXPR_REFINED_DIR="${REPORT_DIR}/token_analysis_expr_first_refined"

run_raw "${ARTIFACT_DIR}/filtered_dual_correct_facts_first.jsonl" "${FACTS_RAW_DIR}"
run_raw "${ARTIFACT_DIR}/filtered_dual_correct_expr_first.jsonl" "${EXPR_RAW_DIR}"

run_simple_plot "${FACTS_RAW_DIR}"
run_simple_plot "${EXPR_RAW_DIR}"

run_refined "${FACTS_RAW_DIR}" "${FACTS_REFINED_DIR}" "${MODEL_LABEL} Facts-first All-hop Refined Token Analysis"
run_refined "${EXPR_RAW_DIR}" "${EXPR_REFINED_DIR}" "${MODEL_LABEL} Expr-first All-hop Refined Token Analysis"

run_refined_plot "${FACTS_REFINED_DIR}"
run_refined_plot "${EXPR_REFINED_DIR}"

echo "[done] token analysis completed for ${MODEL_ID}"
