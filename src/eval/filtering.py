from __future__ import annotations

import argparse
from pathlib import Path

from src.progress import log_event, resolve_log_path, setup_file_logger

from .io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter dual-correct samples for causal patching")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require_label_change", action="store_true")
    args = parser.parse_args()
    logger = setup_file_logger(__name__, resolve_log_path(output_path=args.output))

    rows = read_jsonl(args.input)
    kept = []
    for row in rows:
        if not row.get("correct_clean", False):
            continue
        if not row.get("correct_corrupted", False):
            continue
        if args.require_label_change and bool(row["label"]) == bool(row["label_corrupted"]):
            continue
        kept.append(row)

    write_jsonl(args.output, kept)
    log_event(logger, {"input": len(rows), "kept": len(kept), "output": str(args.output)})


if __name__ == "__main__":
    main()
