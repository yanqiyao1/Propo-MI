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
FACTS_INPUT="${FACTS_INPUT:-dataset/proplogic_mi.jsonl}"
EXPR_INPUT="${EXPR_INPUT:-dataset/proplogic_mi_expr_first.jsonl}"
PROMPT_STYLE="${PROMPT_STYLE:-symbolic}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
MODE="${MODE:-nocot}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:0}"
ENABLE_THINKING="${ENABLE_THINKING:-auto}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_FILTERING="${RUN_FILTERING:-1}"
RUN_METRICS="${RUN_METRICS:-1}"

mkdir -p "${ARTIFACT_DIR}" "${REPORT_DIR}"

FACTS_PREDS="${ARTIFACT_DIR}/preds_facts_first.jsonl"
EXPR_PREDS="${ARTIFACT_DIR}/preds_expr_first.jsonl"
FACTS_FILTERED="${ARTIFACT_DIR}/filtered_dual_correct_facts_first.jsonl"
EXPR_FILTERED="${ARTIFACT_DIR}/filtered_dual_correct_expr_first.jsonl"
FACTS_METRICS="${REPORT_DIR}/metrics_facts_first.json"
EXPR_METRICS="${REPORT_DIR}/metrics_expr_first.json"

run_inference() {
  local input_path="$1"
  local output_path="$2"

  echo "[inference] input=${input_path} output=${output_path}"
  "${PYTHON_BIN}" -m src.eval.inference     --model_id "${MODEL_ID}"     --model_source "${MODEL_SOURCE}"     --input "${input_path}"     --output "${output_path}"     --prompt_style "${PROMPT_STYLE}"     --max_new_tokens "${MAX_NEW_TOKENS}"     --mode "${MODE}"     --temperature "${TEMPERATURE}"     --top_p "${TOP_P}"     --max_samples "${MAX_SAMPLES}"     --dtype "${DTYPE}"     --device_map "${DEVICE_MAP}"     --device "${INFERENCE_DEVICE}"     --enable_thinking "${ENABLE_THINKING}"     --progress_every "${PROGRESS_EVERY}"
}

run_filtering() {
  local input_path="$1"
  local output_path="$2"

  echo "[filter] input=${input_path} output=${output_path}"
  "${PYTHON_BIN}" -m src.eval.filtering     --input "${input_path}"     --output "${output_path}"
}

run_metrics() {
  local input_path="$1"
  local output_path="$2"

  echo "[metrics] input=${input_path} output=${output_path}"
  "${PYTHON_BIN}" -m src.eval.metrics     --input "${input_path}"     --output "${output_path}"
}

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  run_inference "${FACTS_INPUT}" "${FACTS_PREDS}"
  run_inference "${EXPR_INPUT}" "${EXPR_PREDS}"
fi

if [[ "${RUN_FILTERING}" == "1" ]]; then
  run_filtering "${FACTS_PREDS}" "${FACTS_FILTERED}"
  run_filtering "${EXPR_PREDS}" "${EXPR_FILTERED}"
fi

if [[ "${RUN_METRICS}" == "1" ]]; then
  run_metrics "${FACTS_PREDS}" "${FACTS_METRICS}"
  run_metrics "${EXPR_PREDS}" "${EXPR_METRICS}"
fi

echo "[done] inference suite completed for ${MODEL_ID}"
