#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORT_ROOTS=(reports reports_llama3 reports_mistral)

log() {
  printf '[replot-region] %s\n' "$*"
}

plot_module_for_csv() {
  local name="$1"
  case "$name" in
    mlp_*) echo "src.mlp_analysis.plot_region_scores" ;;
    attn_mlp_*) echo "src.attn_mlp_analysis.plot_region_scores" ;;
    attn_*) echo "src.attn_analysis.plot_region_scores" ;;
    *) return 1 ;;
  esac
}

band_module_for_analysis_dir() {
  local analysis_dir="$1"
  case "$analysis_dir" in
    mlp_analysis|mlp_analysis_expr_first|mlp_analysis_facts_first) echo "src.mlp_analysis.plot_band_metrics" ;;
    attn_analysis|attn_analysis_expr_first|attn_analysis_facts_first) echo "src.attn_analysis.plot_band_metrics" ;;
    attn_mlp_analysis|attn_mlp_analysis_expr_first|attn_mlp_analysis_facts_first) echo "src.attn_mlp_analysis.plot_band_metrics" ;;
    *) return 1 ;;
  esac
}

band_output_dir() {
  local parent_dir="$1"
  local one_hop_name="$2"

  if [[ "$parent_dir" == reports/* ]]; then
    if [[ "$one_hop_name" == *_expr_first ]]; then
      if [[ "$one_hop_name" == *14b_* ]]; then
        printf '%s/comparison_expr_first_14B\n' "$parent_dir"
      else
        printf '%s/comparison_expr_first\n' "$parent_dir"
      fi
      return 0
    fi

    if [[ "$one_hop_name" == *14b_* ]]; then
      printf '%s/comparison_14B\n' "$parent_dir"
    else
      printf '%s/comparison\n' "$parent_dir"
    fi
    return 0
  fi

  printf '%s/comparison\n' "$parent_dir"
}

log "Redrawing region score plots from saved csv files"
while IFS= read -r -d '' csv_path; do
  file_name="$(basename "$csv_path")"
  if [[ "$file_name" == *_sample_layer_scores.csv ]]; then
    continue
  fi

  module="$(plot_module_for_csv "$file_name")" || continue
  output_png="${csv_path%.csv}.png"
  log "$module -> $output_png"
  "$PYTHON_BIN" -m "$module" --scores_csv "$csv_path" --output_png "$output_png"

  if [[ "$module" == "src.mlp_analysis.plot_region_scores" ]]; then
    abs_png="${output_png%_scores.png}_abs_dpd_scores.png"
    log "$module (abs_dpd) -> $abs_png"
    "$PYTHON_BIN" -m "$module" --scores_csv "$csv_path" --score_column abs_dpd --output_png "$abs_png"
  fi
done < <(find "${REPORT_ROOTS[@]}" -type f -name '*_scores.csv' -print0 | sort -z)

log "Redrawing band comparison panels"
while IFS= read -r -d '' matrix_csv; do
  one_hop_dir="$(dirname "$matrix_csv")"
  one_hop_name="$(basename "$one_hop_dir")"
  if [[ "$one_hop_name" != *one_hop* ]]; then
    continue
  fi

  two_hop_dir="${one_hop_dir/one_hop/two_hop}"
  if [[ ! -f "$two_hop_dir/BMI_matrix.csv" ]]; then
    log "Skipping $one_hop_dir because matching two-hop directory is missing"
    continue
  fi

  parent_dir="$(dirname "$one_hop_dir")"
  analysis_dir="$(basename "$parent_dir")"
  module="$(band_module_for_analysis_dir "$analysis_dir")" || continue
  output_dir="$(band_output_dir "$parent_dir" "$one_hop_name")"

  log "$module -> $output_dir"
  "$PYTHON_BIN" -m "$module" --one_hop_dir "$one_hop_dir" --two_hop_dir "$two_hop_dir" --output_dir "$output_dir"
done < <(find "${REPORT_ROOTS[@]}" -type f -name 'BMI_matrix.csv' -print0 | sort -z)

log "Done"
