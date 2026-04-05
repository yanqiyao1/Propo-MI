from __future__ import annotations

import argparse
from pathlib import Path

from src.progress import log_event, resolve_log_path, setup_file_logger

from .plot_data import read_scores_csv, resolve_score_label
from .plotting import plot_bar


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MLP region patching scores from saved csv")
    parser.add_argument("--scores_csv", type=Path, required=True)
    parser.add_argument("--output_png", type=Path, required=True)
    parser.add_argument(
        "--score_column",
        default="dpd",
        help="Column name in the score csv. Use 'dpd' for signed dPD or 'abs_dpd' for |dPD|.",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Optional y-axis label. Defaults to a label derived from --score_column.",
    )
    args = parser.parse_args()

    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_png.parent, filename="plot_region_scores.log"))
    layers, scores = read_scores_csv(args.scores_csv, score_column=args.score_column)
    ylabel = args.ylabel or resolve_score_label(args.score_column)
    plot_bar(output_png=args.output_png, layers=layers, scores=scores, ylabel=ylabel)
    log_event(
        logger,
        {
            "scores_csv": str(args.scores_csv),
            "output_png": str(args.output_png),
            "score_column": args.score_column,
            "ylabel": ylabel,
        },
    )


if __name__ == "__main__":
    main()
