from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from src.progress import log_event, resolve_log_path, setup_file_logger

from .common import read_csv
from .step1_discover_fast import _plot_layer_head_distribution


def _infer_plot_shape(classify_csv: Path, summary_json: Path | None) -> Tuple[int, int]:
    n_layers = 0
    n_heads = 0
    if summary_json is not None and summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        n_layers = int(payload.get("n_layers", 0))
        n_heads = int(payload.get("n_heads", 0))

    rows = read_csv(classify_csv)
    if not rows:
        raise ValueError(f"No rows found in classify csv: {classify_csv}")
    if n_layers <= 0:
        n_layers = max(int(float(row.get("layer", 0))) for row in rows) + 1
    if n_heads <= 0:
        n_heads = max(int(float(row.get("head", 0))) for row in rows) + 1
    return n_layers, n_heads


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step1 layer-head role distribution from saved classification files")
    parser.add_argument("--classify_csv", type=Path, required=True)
    parser.add_argument("--output_png", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    args = parser.parse_args()

    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_png.parent, filename="plot_step1_heads.log"))
    rows = read_csv(args.classify_csv)
    if not rows:
        raise ValueError(f"No rows found in classify csv: {args.classify_csv}")
    n_layers, n_heads = _infer_plot_shape(args.classify_csv, args.summary_json)
    _plot_layer_head_distribution(rows, args.output_png, n_layers=n_layers, n_heads=n_heads)
    log_event(
        logger,
        {
            "classify_csv": str(args.classify_csv),
            "summary_json": str(args.summary_json) if args.summary_json is not None else "",
            "output_png": str(args.output_png),
            "n_layers": n_layers,
            "n_heads": n_heads,
        },
    )


if __name__ == "__main__":
    main()
