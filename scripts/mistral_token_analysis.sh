#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_LOCAL_MODEL="/media/snail-ssd/models/Mistral-7B-Instruct-v0.1"
if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
  MODEL_ID="${MODEL_ID:-${DEFAULT_LOCAL_MODEL}}"
else
  MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.1}"
fi

export MODEL_ID
export MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts_mistral}"
export REPORT_DIR="${REPORT_DIR:-reports_mistral}"
export TOKEN_DEVICE="${TOKEN_DEVICE:-cuda:1}"
export TOKEN_MAX_SAMPLES="${TOKEN_MAX_SAMPLES:-1000}"
export EARLY_END="${EARLY_END:-12}"
export MIDDLE_END="${MIDDLE_END:-22}"
export N_LAYERS="${N_LAYERS:-32}"
export MODEL_LABEL="${MODEL_LABEL:-Mistral-7B-Instruct-v0.1}"

bash scripts/model_token_analysis.sh
