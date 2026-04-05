#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CONDA_ENV_PREFIX="/ssd/data/qiyaoyan/conda/envs/minimind"
CONDA_SH="${CONDA_SH:-/home/mingly/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-minimind}"
if [[ -d "${DEFAULT_CONDA_ENV_PREFIX}" ]]; then
  CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${DEFAULT_CONDA_ENV_PREFIX}}"
else
  CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
fi

TOKEN_GPU="${TOKEN_GPU:-cuda:0}"
REGION_GPU="${REGION_GPU:-cuda:1}"

TOKEN_MAX_SAMPLES="${TOKEN_MAX_SAMPLES:-1000}"
REGION_MAX_SAMPLES="${REGION_MAX_SAMPLES:-1000}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"

RUN_INFERENCE="${RUN_INFERENCE:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "${LOG_DIR}"

if [[ "${MODEL_SOURCE}" != "huggingface" && "${MODEL_SOURCE}" != "modelscope" ]]; then
  echo "MODEL_SOURCE must be huggingface or modelscope, got: ${MODEL_SOURCE}" >&2
  exit 1
fi

setup_env() {
  if [[ -n "${CONDA_ENV_PREFIX}" && -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
    case ":${PATH}:" in
      *":${CONDA_ENV_PREFIX}/bin:"*) ;;
      *) export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}" ;;
    esac
    export CONDA_PREFIX="${CONDA_ENV_PREFIX}"
    export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
    hash -r
    cd "${REPO_ROOT}"
    return
  fi

  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    if [[ -n "${CONDA_ENV_PREFIX}" && -d "${CONDA_ENV_PREFIX}" ]]; then
      conda activate "${CONDA_ENV_PREFIX}"
    else
      conda activate "${CONDA_ENV_NAME}"
    fi
    cd "${REPO_ROOT}"
    return
  fi

  echo "Unable to activate environment. Tried CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-<unset>} and CONDA_SH=${CONDA_SH}." >&2
  exit 1
}

resolve_llama_model_id() {
  local local_model="/media/snail-ssd/models/Llama-3.1-8B-Instruct"
  if [[ -d "${local_model}" ]]; then
    printf '%s\n' "${local_model}"
  else
    printf '%s\n' "meta-llama/Llama-3.1-8B-Instruct"
  fi
}

resolve_mistral_model_id() {
  local local_model="/media/snail-ssd/models/Mistral-7B-Instruct-v0.1"
  if [[ -d "${local_model}" ]]; then
    printf '%s\n' "${local_model}"
  else
    printf '%s\n' "mistralai/Mistral-7B-Instruct-v0.1"
  fi
}

run_optional_inference() {
  if [[ "${RUN_INFERENCE}" != "1" ]]; then
    return
  fi

  local llama_model_id
  local mistral_model_id
  llama_model_id="$(resolve_llama_model_id)"
  mistral_model_id="$(resolve_mistral_model_id)"

  MODEL_SOURCE="${MODEL_SOURCE}" \
  bash scripts/qwen3_inference.sh

  MODEL_ID="${llama_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  bash scripts/llama3_inference.sh

  MODEL_ID="${mistral_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  bash scripts/mistral_inference.sh
}

run_token_queue() {
  setup_env

  local python_bin
  python_bin="$(command -v python)"
  local llama_model_id
  local mistral_model_id
  llama_model_id="$(resolve_llama_model_id)"
  mistral_model_id="$(resolve_mistral_model_id)"

  # Qwen uses model-specific artifact names under artifacts/.
  # Use the dedicated script so we don't reference nonexistent facts_first filenames.
  PYTHON_BIN="${python_bin}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  TOKEN_GPU="${TOKEN_GPU}" \
  TOKEN_MAX_SAMPLES="${TOKEN_MAX_SAMPLES}" \
  bash scripts/qwen3_token_analysis.sh

  MODEL_ID="${llama_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  ARTIFACT_DIR="artifacts_llama3" \
  REPORT_DIR="reports_llama3" \
  MODEL_LABEL="Llama-3.1-8B-Instruct" \
  N_LAYERS="32" \
  TOKEN_DEVICE="${TOKEN_GPU}" \
  TOKEN_MAX_SAMPLES="${TOKEN_MAX_SAMPLES}" \
  PYTHON_BIN="${python_bin}" \
  bash scripts/model_token_analysis.sh

  MODEL_ID="${mistral_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  ARTIFACT_DIR="artifacts_mistral" \
  REPORT_DIR="reports_mistral" \
  MODEL_LABEL="Mistral-7B-Instruct-v0.1" \
  N_LAYERS="32" \
  TOKEN_DEVICE="${TOKEN_GPU}" \
  TOKEN_MAX_SAMPLES="${TOKEN_MAX_SAMPLES}" \
  PYTHON_BIN="${python_bin}" \
  bash scripts/model_token_analysis.sh
}

run_region_queue() {
  setup_env

  local llama_model_id
  local mistral_model_id
  llama_model_id="$(resolve_llama_model_id)"
  mistral_model_id="$(resolve_mistral_model_id)"

  # Qwen uses model-specific artifact names under artifacts/.
  # Use the dedicated scripts so we don't reference nonexistent facts_first filenames.
  MODEL_SOURCE="${MODEL_SOURCE}" \
  REGION_GPU="${REGION_GPU}" \
  REGION_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/qwen3_mlp_patching.sh
  MODEL_SOURCE="${MODEL_SOURCE}" \
  REGION_GPU="${REGION_GPU}" \
  REGION_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/qwen3_attention_patching.sh
  MODEL_SOURCE="${MODEL_SOURCE}" \
  REGION_GPU="${REGION_GPU}" \
  REGION_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/qwen3_attention_mlp_patching.sh

  MODEL_ID="${llama_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  ARTIFACT_DIR="artifacts_llama3" \
  REPORT_DIR="reports_llama3" \
  DATASET_DIR="dataset_llama3" \
  MLP_DEVICE="${REGION_GPU}" \
  MLP_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/model_mlp_patching.sh

  MODEL_ID="${llama_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  ARTIFACT_DIR="artifacts_llama3" \
  REPORT_DIR="reports_llama3" \
  DATASET_DIR="dataset_llama3" \
  ATTN_DEVICE="${REGION_GPU}" \
  ATTN_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/model_attention_patching.sh

  MODEL_ID="${llama_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  ARTIFACT_DIR="artifacts_llama3" \
  REPORT_DIR="reports_llama3" \
  DATASET_DIR="dataset_llama3" \
  ATTN_MLP_DEVICE="${REGION_GPU}" \
  ATTN_MLP_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/model_attention_mlp_patching.sh

  MODEL_ID="${mistral_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  ARTIFACT_DIR="artifacts_mistral" \
  REPORT_DIR="reports_mistral" \
  DATASET_DIR="dataset_mistral" \
  MLP_DEVICE="${REGION_GPU}" \
  MLP_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/model_mlp_patching.sh

  MODEL_ID="${mistral_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  ARTIFACT_DIR="artifacts_mistral" \
  REPORT_DIR="reports_mistral" \
  DATASET_DIR="dataset_mistral" \
  ATTN_DEVICE="${REGION_GPU}" \
  ATTN_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/model_attention_patching.sh

  MODEL_ID="${mistral_model_id}" \
  MODEL_SOURCE="${MODEL_SOURCE}" \
  ARTIFACT_DIR="artifacts_mistral" \
  REPORT_DIR="reports_mistral" \
  DATASET_DIR="dataset_mistral" \
  ATTN_MLP_DEVICE="${REGION_GPU}" \
  ATTN_MLP_MAX_SAMPLES="${REGION_MAX_SAMPLES}" \
  bash scripts/model_attention_mlp_patching.sh
}

main() {
  run_optional_inference

  local token_log="${LOG_DIR}/logs_token_${TOKEN_GPU//:/_}.out"
  local region_log="${LOG_DIR}/logs_region_${REGION_GPU//:/_}.out"

  run_token_queue >"${token_log}" 2>&1 &
  local token_pid=$!

  run_region_queue >"${region_log}" 2>&1 &
  local region_pid=$!

  echo "token queue pid: ${token_pid}"
  echo "region queue pid: ${region_pid}"
  echo "token log: ${token_log}"
  echo "region log: ${region_log}"

  wait "${token_pid}"
  wait "${region_pid}"

  echo "[done] all token_analysis and region patching queues completed"
}

run_token_analysis_main() {
  run_optional_inference

  local token_log="${LOG_DIR}/logs_token_${TOKEN_GPU//:/_}.out"

  run_token_queue >"${token_log}" 2>&1 &
  local token_pid=$!

  echo "token queue pid: ${token_pid}"
  echo "token log: ${token_log}"

  wait "${token_pid}"

  echo "[done] token_analysis queue completed"
}

run_region_patching_main() {
  run_optional_inference

  local region_log="${LOG_DIR}/logs_region_${REGION_GPU//:/_}.out"

  run_region_queue >"${region_log}" 2>&1 &
  local region_pid=$!

  echo "region queue pid: ${region_pid}"
  echo "region log: ${region_log}"

  wait "${region_pid}"

  echo "[done] region patching queue completed"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
