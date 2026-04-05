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
export MODEL_SOURCE="${MODEL_SOURCE:-huggingface}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts_mistral}"
export REPORT_DIR="${REPORT_DIR:-reports_mistral}"
export DATASET_DIR="${DATASET_DIR:-dataset_mistral}"
export MLP_DEVICE="${MLP_DEVICE:-cuda:1}"
export MLP_MAX_SAMPLES="${MLP_MAX_SAMPLES:-1000}"

bash scripts/model_mlp_patching.sh
