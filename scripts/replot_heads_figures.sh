#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORT_ROOTS=(reports reports_llama3 reports_mistral)

log() {
  printf '[replot-heads] %s\n' "$*"
}

log "Redrawing Step 1 head distribution plots"
while IFS= read -r -d '' classify_csv; do
  classify_dir="$(dirname "$classify_csv")"
  analysis_dir="$(dirname "$classify_dir")"
  output_png="$classify_dir/layer_head_role_distribution.png"
  summary_json="$analysis_dir/step1_summary.json"
  cmd=("$PYTHON_BIN" -m src.heads_analysis.plot_step1_heads --classify_csv "$classify_csv" --output_png "$output_png")
  if [[ -f "$summary_json" ]]; then
    cmd+=(--summary_json "$summary_json")
  fi
  log "$output_png"
  "${cmd[@]}"
done < <(find "${REPORT_ROOTS[@]}" -type f -path '*/classify/top_heads_pattern_labels.csv' -print0 | sort -z)

log "Redrawing Step 2 taxonomy plots"
while IFS= read -r -d '' counts_csv; do
  taxonomy_dir="$(dirname "$counts_csv")"
  output_png="$taxonomy_dir/head_taxonomy_line_chart.png"
  log "$output_png"
  "$PYTHON_BIN" -m src.heads_analysis.plot_step2_taxonomy --counts_csv "$counts_csv" --output_png "$output_png"
done < <(find "${REPORT_ROOTS[@]}" -type f -path '*/taxonomy/head_taxonomy_counts.csv' -print0 | sort -z)

log "Redrawing Step 3 validation curves"
while IFS= read -r -d '' pd_curve_csv; do
  validation_dir="$(dirname "$pd_curve_csv")"
  output_dir="$validation_dir/plots"
  log "$output_dir"
  "$PYTHON_BIN" -m src.heads_analysis.plot_step3_curves --pd_curve_csv "$pd_curve_csv" --output_dir "$output_dir" --include_signed_ratio_plot
done < <(find "${REPORT_ROOTS[@]}" -type f -path '*/validation/pd_curve_metrics.csv' -print0 | sort -z)

log "Done"
