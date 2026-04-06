from __future__ import annotations

import argparse
from pathlib import Path

from src.progress import log_event, resolve_log_path, setup_file_logger

from .plot_only import plot_taxonomy_line_chart_from_counts, read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step2 taxonomy chart from saved counts csv")
    parser.add_argument("--counts_csv", type=Path, required=True)
    parser.add_argument("--output_png", type=Path, required=True)
    args = parser.parse_args()

    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_png.parent, filename="plot_step2_taxonomy.log"))
    rows = read_csv(args.counts_csv)
    if not rows:
        raise ValueError(f"No rows found in counts csv: {args.counts_csv}")
    plot_taxonomy_line_chart_from_counts(rows, args.output_png)
    log_event(logger, {"counts_csv": str(args.counts_csv), "output_png": str(args.output_png)})


if __name__ == "__main__":
    main()
