from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

from src.progress import log_event, resolve_log_path, setup_file_logger

from .activation_patching_dataset import (
    _plot_layer_stage_simple,
    _plot_simple_comparison,
    _plot_simple_heatmap,
    aggregate_by_category_with_sample_avg,
    compute_simple_heatmap,
    excluded_simple_categories_for_results,
)
from .result_schema import infer_delta_score_n_layers


def _load_results(path: Path) -> List[Dict[str, object]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("results", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"Unsupported patching results payload: {type(payload).__name__}")


def _load_summary(path: Path | None) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported summary json: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot token-analysis simple figures from saved patching results")
    parser.add_argument("--input_pkl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--early_end", type=int, default=-1)
    parser.add_argument("--middle_end", type=int, default=-1)
    parser.add_argument("--n_layers", type=int, default=0)
    args = parser.parse_args()

    summary = _load_summary(args.summary_json)
    early_end = int(args.early_end) if args.early_end > 0 else int(summary.get("early_end", 14))
    middle_end = int(args.middle_end) if args.middle_end > 0 else int(summary.get("middle_end", 24))
    n_layers = int(args.n_layers) if args.n_layers > 0 else int(summary.get("n_layers", 0))

    results = _load_results(args.input_pkl)
    if not results:
        raise ValueError(f"No patching results found in {args.input_pkl}")
    if n_layers <= 0:
        n_layers = infer_delta_score_n_layers(results, fallback=0)
        if n_layers <= 0:
            raise ValueError("Unable to infer n_layers from results; please pass --n_layers")
    if not (0 < early_end < middle_end < n_layers):
        raise ValueError(f"Invalid layer boundaries: early_end={early_end}, middle_end={middle_end}, n_layers={n_layers}")

    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_dir, filename="plot_simple_analysis.log"))
    excluded_categories = excluded_simple_categories_for_results(results)
    stats_rows = aggregate_by_category_with_sample_avg(
        results,
        n_layers=n_layers,
        early_end=early_end,
        middle_end=middle_end,
        exclude_categories=excluded_categories,
    )
    heatmap_data, category_order = compute_simple_heatmap(
        results,
        n_layers=n_layers,
        exclude_categories=excluded_categories,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_simple_comparison(stats_rows, output_path=args.output_dir / "category_comparison_simple.png")
    _plot_layer_stage_simple(
        stats_rows,
        output_path=args.output_dir / "layer_stage_simple.png",
        early_end=early_end,
        middle_end=middle_end,
        n_layers=n_layers,
    )
    _plot_simple_heatmap(
        heatmap_data,
        category_order,
        output_path=args.output_dir / "heatmap_simple.png",
        early_end=early_end,
        middle_end=middle_end,
    )
    log_event(
        logger,
        {
            "input_pkl": str(args.input_pkl),
            "summary_json": str(args.summary_json) if args.summary_json is not None else "",
            "output_dir": str(args.output_dir),
            "n_layers": n_layers,
            "early_end": early_end,
            "middle_end": middle_end,
            "n_results": len(results),
            "excluded_categories": sorted(excluded_categories),
        },
    )


if __name__ == "__main__":
    main()
