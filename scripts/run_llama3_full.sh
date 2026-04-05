#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_FILTERING="${RUN_FILTERING:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
RUN_HEADS="${RUN_HEADS:-1}"
RUN_MLP="${RUN_MLP:-1}"
RUN_TOKEN="${RUN_TOKEN:-1}"

if [[ "${RUN_INFERENCE}" == "1" || "${RUN_FILTERING}" == "1" || "${RUN_METRICS}" == "1" ]]; then
  export RUN_INFERENCE RUN_FILTERING RUN_METRICS
  bash scripts/llama3_inference.sh
fi

if [[ "${RUN_HEADS}" == "1" ]]; then
  bash scripts/llama3_head_analysis.sh
fi

if [[ "${RUN_MLP}" == "1" ]]; then
  bash scripts/llama3_mlp_patching.sh
fi

if [[ "${RUN_TOKEN}" == "1" ]]; then
  bash scripts/llama3_token_analysis.sh
fi

echo "[done] llama3 full pipeline completed"
