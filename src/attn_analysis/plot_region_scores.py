from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from src.progress import log_event, resolve_log_path, setup_file_logger

from .attn_region_ablation import _plot_bar


def _read_scores(csv_path: Path) -> tuple[List[int], List[float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Score csv is empty: {csv_path}")
    layers = [int(float(row["layer"])) for row in rows]
    scores = [float(row["dpd"]) for row in rows]
    return layers, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot attention region patching scores from saved csv")
    parser.add_argument("--scores_csv", type=Path, required=True)
    parser.add_argument("--output_png", type=Path, required=True)
    args = parser.parse_args()

    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_png.parent, filename="plot_region_scores.log"))
    layers, scores = _read_scores(args.scores_csv)
    _plot_bar(output_png=args.output_png, layers=layers, scores=scores)
    log_event(logger, {"scores_csv": str(args.scores_csv), "output_png": str(args.output_png)})


if __name__ == "__main__":
    main()
