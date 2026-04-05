from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from src.progress import log_event, setup_file_logger

from .io_utils import read_jsonl


def _mean(items: Iterable[float]) -> float:
    vals = list(items)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def summarize_predictions(rows: List[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    summary["count"] = len(rows)
    summary["accuracy_clean"] = _mean(float(bool(r.get("correct_clean", False))) for r in rows)
    summary["accuracy_corrupted"] = _mean(float(bool(r.get("correct_corrupted", False))) for r in rows)
    summary["dual_correct_rate"] = _mean(
        float(bool(r.get("correct_clean", False) and r.get("correct_corrupted", False))) for r in rows
    )

    by_hop: Dict[str, List[float]] = defaultdict(list)
    by_rule: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_hop[str(row["hop"])].append(float(bool(row.get("correct_clean", False))))
        by_rule[str(row["rule"])].append(float(bool(row.get("correct_clean", False))))

    summary["accuracy_by_hop"] = {k: _mean(v) for k, v in sorted(by_hop.items())}
    summary["accuracy_by_rule"] = {k: _mean(v) for k, v in sorted(by_rule.items())}
    return summary


def compare_before_after(
    before_rows: List[Dict[str, object]],
    after_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    before_map = {str(r["id"]): r for r in before_rows}
    after_map = {str(r["id"]): r for r in after_rows}
    shared_ids = sorted(set(before_map) & set(after_map))

    error_total = 0
    flipped = 0
    regressed = 0

    for sid in shared_ids:
        b = bool(before_map[sid].get("correct_clean", False))
        a = bool(after_map[sid].get("correct_clean", False))
        if not b:
            error_total += 1
            if a:
                flipped += 1
        if b and not a:
            regressed += 1

    return {
        "shared_count": len(shared_ids),
        "before_error_count": error_total,
        "error_flip_rate": (flipped / error_total) if error_total else 0.0,
        "regression_rate_over_shared": (regressed / len(shared_ids)) if shared_ids else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--after", type=Path, default=None)
    args = parser.parse_args()
    log_path = args.output.with_suffix(".log") if args.output else args.input.with_suffix(".metrics.log")
    logger = setup_file_logger(__name__, log_path)

    rows = read_jsonl(args.input)
    report = {"summary": summarize_predictions(rows)}

    if args.after is not None:
        after_rows = read_jsonl(args.after)
        report["before_after"] = compare_before_after(rows, after_rows)

    text = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    log_event(logger, text)


if __name__ == "__main__":
    main()
