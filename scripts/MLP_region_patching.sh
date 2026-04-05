#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
PROMPT_STYLE="${PROMPT_STYLE:-symbolic}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
SPLIT="${SPLIT:-all}"
SOURCE_INPUT="${SOURCE_INPUT:-artifacts/filtered_dual_correct_Qwen3-8b.jsonl}"
ONE_HOP_INPUT="${ONE_HOP_INPUT:-dataset/mlp_8b_one_hop.jsonl}"
TWO_HOP_INPUT="${TWO_HOP_INPUT:-dataset/mlp_8b_two_hop.jsonl}"
SPLIT_SUMMARY="${SPLIT_SUMMARY:-reports/mlp_analysis/8b_dual_correct_split.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reports/mlp_analysis}"
AUTO_SPLIT="${AUTO_SPLIT:-1}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-${OUTPUT_ROOT}/comparison}"
REGIONS=(facts_region expression_region query_region)
HOPS=(one_hop two_hop)

usage() {
  cat <<USAGE
Usage: bash scripts/MLP_region_patching.sh [options]

Options:
  --model_id ID            Model id or local path. Default: ${MODEL_ID}
  --model_source SOURCE    huggingface | modelscope. Default: ${MODEL_SOURCE}
  --prompt_style STYLE     symbolic | semi_natural. Default: ${PROMPT_STYLE}
  --device DEVICE          Torch device. Default: ${DEVICE}
  --max_samples N          Balanced per-rule max samples per run; 0 means all. Default: ${MAX_SAMPLES}
  --progress_every N       Progress interval. Default: ${PROGRESS_EVERY}
  --split SPLIT            Deprecated; ignored by current split_by_hop. Default: ${SPLIT}
  --source_input PATH      Source jsonl used for auto split. Default: ${SOURCE_INPUT}
  --one_hop_input PATH     One-hop jsonl path. Default: ${ONE_HOP_INPUT}
  --two_hop_input PATH     Two-hop jsonl path. Default: ${TWO_HOP_INPUT}
  --split_summary PATH     Split summary json path. Default: ${SPLIT_SUMMARY}
  --output_root PATH       Root output dir. Default: ${OUTPUT_ROOT}
  --auto_split 0|1         Auto-generate one/two-hop files if missing. Default: ${AUTO_SPLIT}
  --make_plots 0|1         Generate panel plots after aggregation. Default: ${MAKE_PLOTS}
  --plot_output_dir PATH   Plot output dir. Default: ${PLOT_OUTPUT_DIR}
  -h, --help               Show this help message.

Environment variables with the same names are also supported.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id)
      MODEL_ID="$2"
      shift 2
      ;;
    --model_source)
      MODEL_SOURCE="$2"
      shift 2
      ;;
    --prompt_style)
      PROMPT_STYLE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --progress_every)
      PROGRESS_EVERY="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --source_input)
      SOURCE_INPUT="$2"
      shift 2
      ;;
    --one_hop_input)
      ONE_HOP_INPUT="$2"
      shift 2
      ;;
    --two_hop_input)
      TWO_HOP_INPUT="$2"
      shift 2
      ;;
    --split_summary)
      SPLIT_SUMMARY="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      PLOT_OUTPUT_DIR="${OUTPUT_ROOT}/comparison"
      ;;
    --auto_split)
      AUTO_SPLIT="$2"
      shift 2
      ;;
    --make_plots)
      MAKE_PLOTS="$2"
      shift 2
      ;;
    --plot_output_dir)
      PLOT_OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${PROMPT_STYLE}" != "symbolic" && "${PROMPT_STYLE}" != "semi_natural" ]]; then
  echo "PROMPT_STYLE must be symbolic or semi_natural, got: ${PROMPT_STYLE}" >&2
  exit 1
fi

if [[ "${MODEL_SOURCE}" != "huggingface" && "${MODEL_SOURCE}" != "modelscope" ]]; then
  echo "MODEL_SOURCE must be huggingface or modelscope, got: ${MODEL_SOURCE}" >&2
  exit 1
fi

if [[ "${SPLIT}" != "all" ]]; then
  echo "[warn] --split ${SPLIT} is ignored because src.data.split_by_hop no longer supports split filtering" >&2
fi

if [[ ! -f "${ONE_HOP_INPUT}" || ! -f "${TWO_HOP_INPUT}" ]]; then
  if [[ "${AUTO_SPLIT}" != "1" ]]; then
    echo "Missing split dataset files and AUTO_SPLIT=0:" >&2
    echo "  ${ONE_HOP_INPUT}" >&2
    echo "  ${TWO_HOP_INPUT}" >&2
    exit 1
  fi
  if [[ ! -f "${SOURCE_INPUT}" ]]; then
    echo "Source input not found for auto split: ${SOURCE_INPUT}" >&2
    exit 1
  fi

  echo "[split] Generating one-hop/two-hop files from ${SOURCE_INPUT}"
  python -m src.data.split_by_hop \
    --input "${SOURCE_INPUT}" \
    --out_one "${ONE_HOP_INPUT}" \
    --out_two "${TWO_HOP_INPUT}" \
    --summary "${SPLIT_SUMMARY}"
fi

region_output_is_current() {
  local summary_path="$1"
  local scores_csv="$2"

  [[ -f "${summary_path}" && -f "${scores_csv}" ]]
}

run_one() {
  local hop="$1"
  local region="$2"
  local input_path
  local output_dir
  local summary_path
  local scores_csv

  if [[ "${hop}" == "one_hop" ]]; then
    input_path="${ONE_HOP_INPUT}"
  else
    input_path="${TWO_HOP_INPUT}"
  fi

  output_dir="${OUTPUT_ROOT}/$(basename "${input_path}" .jsonl)/${region}"
  summary_path="${output_dir}/mlp_${region}_summary.json"
  scores_csv="${output_dir}/mlp_${region}_scores.csv"

  if region_output_is_current "${summary_path}" "${scores_csv}"; then
    echo "[skip] hop=${hop} region=${region} output=${output_dir}"
    return
  fi

  echo "[run] hop=${hop} region=${region} input=${input_path}"
  python -m src.mlp_analysis.mlp_region_ablation \
    --model_id "${MODEL_ID}" \
    --model_source "${MODEL_SOURCE}" \
    --input "${input_path}" \
    --output_dir "${output_dir}" \
    --prompt_style "${PROMPT_STYLE}" \
    --region_mode "${region}" \
    --max_samples "${MAX_SAMPLES}" \
    --device "${DEVICE}" \
    --progress_every "${PROGRESS_EVERY}" \
    --no-save-plots
}

plot_one() {
  local hop="$1"
  local region="$2"
  local input_path
  local output_dir

  if [[ "${hop}" == "one_hop" ]]; then
    input_path="${ONE_HOP_INPUT}"
  else
    input_path="${TWO_HOP_INPUT}"
  fi

  output_dir="${OUTPUT_ROOT}/$(basename "${input_path}" .jsonl)/${region}"
  echo "[plot] hop=${hop} region=${region} output=${output_dir}"
  python -m src.mlp_analysis.plot_region_scores \
    --scores_csv "${output_dir}/mlp_${region}_scores.csv" \
    --output_png "${output_dir}/mlp_${region}_scores.png"
}

aggregate_hop() {
  local hop="$1"
  local input_path
  local hop_dir

  if [[ "${hop}" == "one_hop" ]]; then
    input_path="${ONE_HOP_INPUT}"
  else
    input_path="${TWO_HOP_INPUT}"
  fi
  hop_dir="${OUTPUT_ROOT}/$(basename "${input_path}" .jsonl)"

  python - "${hop_dir}" <<'PY'
import csv
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
regions = ["facts_region", "expression_region", "query_region"]
bands = ["early", "middle", "late"]
metrics = ["BMI", "BCR", "SBI"]

payload = {
    "base_dir": str(base_dir),
    "regions": regions,
    "bands": bands,
    "BMI": {},
    "BCR": {},
    "SBI": {},
}

for region in regions:
    summary_path = base_dir / region / f"mlp_{region}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary for region {region}: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    band_metrics = summary["band_metrics"]
    for metric in metrics:
        payload[metric][region] = {band: float(band_metrics[metric][band]) for band in bands}

json_path = base_dir / "band_metrics_matrix.json"
json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

for metric in metrics:
    csv_path = base_dir / f"{metric}_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["region", *bands])
        writer.writeheader()
        for region in regions:
            row = {"region": region}
            row.update(payload[metric][region])
            writer.writerow(row)

print({"stage": "band_metrics_matrix_done", "base_dir": str(base_dir), "json": str(json_path)})
PY
}

for hop in "${HOPS[@]}"; do
  for region in "${REGIONS[@]}"; do
    run_one "${hop}" "${region}"
    if [[ "${MAKE_PLOTS}" == "1" ]]; then
      plot_one "${hop}" "${region}"
    fi
  done
  aggregate_hop "${hop}"
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  ONE_HOP_DIR="${OUTPUT_ROOT}/$(basename "${ONE_HOP_INPUT}" .jsonl)"
  TWO_HOP_DIR="${OUTPUT_ROOT}/$(basename "${TWO_HOP_INPUT}" .jsonl)"
  echo "[plot] Generating comparison plots in ${PLOT_OUTPUT_DIR}"
  python -m src.mlp_analysis.plot_band_metrics \
    --one_hop_dir "${ONE_HOP_DIR}" \
    --two_hop_dir "${TWO_HOP_DIR}" \
    --output_dir "${PLOT_OUTPUT_DIR}"
fi

echo "[done] Outputs saved under ${OUTPUT_ROOT}"
