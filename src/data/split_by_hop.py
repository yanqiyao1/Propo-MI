from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.progress import log_event, resolve_log_path, setup_file_logger

from src.eval.io_utils import read_jsonl, write_jsonl


def split_rows_by_hop(rows: Iterable[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    one_hop: List[Dict[str, object]] = []
    two_hop: List[Dict[str, object]] = []
    unknown: List[str] = []

    for row in rows:
        hop = str(row.get("hop", ""))
        if hop == "one_hop":
            one_hop.append(row)
        elif hop == "two_hop":
            two_hop.append(row)
        else:
            unknown.append(str(row.get("id", "<missing-id>")))

    if unknown:
        preview = ", ".join(unknown[:10])
        raise ValueError(f"Found rows with unknown hop values (showing first 10 ids): {preview}")

    return one_hop, two_hop


def _rule_counts(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(r.get("rule", "")) for r in rows).items()))


def build_summary(
    input_path: Path,
    selected_rows: List[Dict[str, object]],
    one_hop_rows: List[Dict[str, object]],
    two_hop_rows: List[Dict[str, object]],
    out_one: Path,
    out_two: Path,
) -> Dict[str, object]:
    return {
        "input": str(input_path),
        "selected_count": len(selected_rows),
        "one_hop_count": len(one_hop_rows),
        "two_hop_count": len(two_hop_rows),
        "counts_match": len(selected_rows) == (len(one_hop_rows) + len(two_hop_rows)),
        "one_hop_rule_counts": _rule_counts(one_hop_rows),
        "two_hop_rule_counts": _rule_counts(two_hop_rows),
        "out_one": str(out_one),
        "out_two": str(out_two),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split PropLogic-MI jsonl into one-hop and two-hop files")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out_one", type=Path, required=True)
    parser.add_argument("--out_two", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    log_path = args.summary if args.summary is not None else args.out_one.with_suffix(".log")
    logger = setup_file_logger(__name__, resolve_log_path(output_path=log_path))

    all_rows = read_jsonl(args.input)
    selected_rows = list(all_rows)
    one_hop_rows, two_hop_rows = split_rows_by_hop(selected_rows)

    write_jsonl(args.out_one, one_hop_rows)
    write_jsonl(args.out_two, two_hop_rows)

    summary = build_summary(
        input_path=args.input,
        selected_rows=selected_rows,
        one_hop_rows=one_hop_rows,
        two_hop_rows=two_hop_rows,
        out_one=args.out_one,
        out_two=args.out_two,
    )

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    log_event(logger, summary)


if __name__ == "__main__":
    main()
