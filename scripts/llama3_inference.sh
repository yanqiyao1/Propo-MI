#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_LOCAL_MODEL="/media/snail-ssd/models/Llama-3.1-8B-Instruct"
if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
  MODEL_ID="${MODEL_ID:-${DEFAULT_LOCAL_MODEL}}"
else
  MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
fi

export MODEL_ID
export MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts_llama3}"
export REPORT_DIR="${REPORT_DIR:-reports_llama3}"
export INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:0}"

bash scripts/model_inference_suite.sh
