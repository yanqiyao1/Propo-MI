#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/scripts/replot_region_figures.sh"
"$ROOT_DIR/scripts/replot_token_figures.sh"
"$ROOT_DIR/scripts/replot_heads_figures.sh"
"$ROOT_DIR/scripts/sync_report_figures_to_figures_experiments.sh"
