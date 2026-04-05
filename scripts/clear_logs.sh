#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/clear_logs.sh [--dry-run]

Delete all .log files under the repository root, excluding .git.

Options:
  --dry-run   Show which files would be deleted without removing them.
  -h, --help  Show this help message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
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

mapfile -d '' LOG_FILES < <(find . -path './.git' -prune -o -type f -name '*.log' -print0 | sort -z)
LOG_COUNT="${#LOG_FILES[@]}"

if [[ "${LOG_COUNT}" -eq 0 ]]; then
  echo "No .log files found under ${REPO_ROOT}"
  exit 0
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Would delete ${LOG_COUNT} .log file(s):"
  for log_file in "${LOG_FILES[@]}"; do
    echo "  ${log_file#./}"
  done
  exit 0
fi

echo "Deleting ${LOG_COUNT} .log file(s):"
for log_file in "${LOG_FILES[@]}"; do
  rm -f -- "${log_file}"
  echo "  removed ${log_file#./}"
done

echo "Done."
