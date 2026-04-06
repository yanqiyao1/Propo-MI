#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p figures_experiments

printf '[sync-figures] copying png/pdf files into figures_experiments\n'
while IFS= read -r -d '' figure_path; do
  cp --parents "$figure_path" figures_experiments/
done < <(find reports reports_llama3 reports_mistral -type f \( -name '*.png' -o -name '*.pdf' \) -print0 | sort -z)

printf '[sync-figures] done\n'
