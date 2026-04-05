#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:?MODEL_ID must be set}"
MODEL_SOURCE="${MODEL_SOURCE:-huggingface}"
ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
REPORT_DIR="${REPORT_DIR:?REPORT_DIR must be set}"
PROMPT_STYLE="${PROMPT_STYLE:-symbolic}"
HEADS_DEVICE="${HEADS_DEVICE:-cuda:1}"
IMPACT_SAMPLES="${IMPACT_SAMPLES:-2000}"
PROBE_SAMPLES="${PROBE_SAMPLES:-256}"
CLASSIFY_SAMPLES="${CLASSIFY_SAMPLES:-2000}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-1000}"
VALIDATION_ACCURACY_SAMPLES="${VALIDATION_ACCURACY_SAMPLES:-500}"
TOP_N="${TOP_N:-512}"
TOP_M_PER_LAYER="${TOP_M_PER_LAYER:-4}"
CANDIDATE_POOL_MULT="${CANDIDATE_POOL_MULT:-4}"
QUANTILE_KEEP="${QUANTILE_KEEP:-0.6}"
LATE_LAYER_FRAC="${LATE_LAYER_FRAC:-0.0}"
SCORE_MODE="${SCORE_MODE:-zscore}"
K_VALUES="${K_VALUES:-1,2,4,8,16,32,64}"
RANDOM_TRIALS_MIN="${RANDOM_TRIALS_MIN:-6}"
RANDOM_TRIALS_MAX="${RANDOM_TRIALS_MAX:-20}"
RANDOM_SEM_TARGET="${RANDOM_SEM_TARGET:-0.01}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SEED="${SEED:-42}"
STEPS="${STEPS:-1,2,3}"
NO_SAVE_PLOTS="${NO_SAVE_PLOTS:-1}"

mkdir -p "${REPORT_DIR}"

FACTS_INPUT="${ARTIFACT_DIR}/filtered_dual_correct_facts_first.jsonl"
EXPR_INPUT="${ARTIFACT_DIR}/filtered_dual_correct_expr_first.jsonl"
FACTS_OUTPUT_DIR="${REPORT_DIR}/heads_analysis_facts_first"
EXPR_OUTPUT_DIR="${REPORT_DIR}/heads_analysis_expr_first"

plot_heads_experiment() {
  local output_dir="$1"

  "${PYTHON_BIN}" -m src.heads_analysis.plot_step1_heads     --classify_csv "${output_dir}/classify/top_heads_pattern_labels.csv"     --summary_json "${output_dir}/step1_summary.json"     --output_png "${output_dir}/classify/layer_head_role_distribution.png"

  "${PYTHON_BIN}" -m src.heads_analysis.plot_step2_taxonomy     --counts_csv "${output_dir}/taxonomy/head_taxonomy_counts.csv"     --output_png "${output_dir}/taxonomy/head_taxonomy_line_chart.png"

  "${PYTHON_BIN}" -m src.heads_analysis.plot_step3_curves     --pd_curve_csv "${output_dir}/validation/pd_curve_metrics.csv"     --output_dir "${output_dir}/validation/plots"
}

run_heads() {
  local input_path="$1"
  local output_dir="$2"
  local prompt_order="$3"
  local extra_args=()

  if [[ "${NO_SAVE_PLOTS}" == "1" ]]; then
    extra_args+=(--no-save-plots)
  fi

  mkdir -p "${output_dir}"

  echo "[heads] prompt_order=${prompt_order} output=${output_dir}"
  "${PYTHON_BIN}" -m src.heads_analysis.run_all     --model_id "${MODEL_ID}"     --model_source "${MODEL_SOURCE}"     --output_dir "${output_dir}"     --input "${input_path}"     --hop all     --prompt_order "${prompt_order}"     --prompt_style "${PROMPT_STYLE}"     --token_scope all_tokens     --impact_samples "${IMPACT_SAMPLES}"     --probe_samples "${PROBE_SAMPLES}"     --classify_samples "${CLASSIFY_SAMPLES}"     --validation_samples "${VALIDATION_SAMPLES}"     --validation_accuracy_samples "${VALIDATION_ACCURACY_SAMPLES}"     --top_n "${TOP_N}"     --top_m_per_layer "${TOP_M_PER_LAYER}"     --candidate_pool_mult "${CANDIDATE_POOL_MULT}"     --quantile_keep "${QUANTILE_KEEP}"     --late_layer_frac "${LATE_LAYER_FRAC}"     --score_mode "${SCORE_MODE}"     --k_values "${K_VALUES}"     --random_trials_min "${RANDOM_TRIALS_MIN}"     --random_trials_max "${RANDOM_TRIALS_MAX}"     --random_sem_target "${RANDOM_SEM_TARGET}"     --eval_batch_size "${EVAL_BATCH_SIZE}"     --seed "${SEED}"     --device "${HEADS_DEVICE}"     --steps "${STEPS}"     "${extra_args[@]}"

  plot_heads_experiment "${output_dir}"
}

run_heads "${FACTS_INPUT}" "${FACTS_OUTPUT_DIR}" "facts_first"
run_heads "${EXPR_INPUT}" "${EXPR_OUTPUT_DIR}" "expr_first"

echo "[done] heads analysis completed for ${MODEL_ID}"
