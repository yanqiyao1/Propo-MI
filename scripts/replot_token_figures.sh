#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORT_ROOTS=(reports reports_llama3 reports_mistral)

log() {
  printf '[replot-token] %s\n' "$*"
}

log "Redrawing simple token-analysis plots"
while IFS= read -r -d '' pkl_path; do
  output_dir="$(dirname "$pkl_path")"
  summary_json="$output_dir/summary.json"
  cmd=("$PYTHON_BIN" -m src.token_analysis.plot_simple_analysis --input_pkl "$pkl_path" --output_dir "$output_dir")
  if [[ -f "$summary_json" ]]; then
    cmd+=(--summary_json "$summary_json")
  fi
  log "$output_dir"
  "${cmd[@]}"
done < <(find "${REPORT_ROOTS[@]}" -type f -name 'patching_results.pkl' -print0 | sort -z)

log "Redrawing refined token-analysis plots"
while IFS= read -r -d '' sum_json; do
  output_dir="$(dirname "$sum_json")"
  mean_json="$output_dir/refined_stats_mean.json"
  cmd=("$PYTHON_BIN" -m src.token_analysis.plot_refined_analysis --sum_stats_json "$sum_json" --output_dir "$output_dir")
  if [[ -f "$mean_json" ]]; then
    cmd+=(--mean_stats_json "$mean_json")
  fi
  log "$output_dir"
  "${cmd[@]}"
done < <(find "${REPORT_ROOTS[@]}" -type f -name 'refined_stats_sum.json' -print0 | sort -z)

log "Done"
