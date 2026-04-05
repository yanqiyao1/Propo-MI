from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from src.progress import log_event, resolve_log_path, setup_file_logger

from .refined_plotting import plot_by_stage


def _load_stats_payload(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported stats json: {path}")
    return payload


def _plot_one(stats_json: Path, output_png: Path, default_title_suffix: str, default_ylabel: str) -> None:
    payload = _load_stats_payload(stats_json)
    stats = payload.get("stats")
    category_labels = payload.get("category_labels")
    metadata = payload.get("metadata")
    if not isinstance(stats, dict) or not isinstance(category_labels, dict) or not isinstance(metadata, dict):
        raise ValueError(f"Malformed refined stats payload: {stats_json}")
    category_order = metadata.get("category_order")
    if not isinstance(category_order, list):
        raise ValueError(f"Missing category_order in refined stats payload: {stats_json}")
    stage_bounds = metadata.get("stage_bounds", {})
    if not isinstance(stage_bounds, dict):
        raise ValueError(f"Missing stage_bounds in refined stats payload: {stats_json}")

    early_end = int(stage_bounds.get("early", [0, 0])[1]) + 1
    middle_end = int(stage_bounds.get("middle", [0, 0])[1]) + 1
    n_layers = int(stage_bounds.get("late", [0, 0])[1]) + 1
    title = str(payload.get("title", "Refined Token Analysis")).strip()
    if title:
        title = f"{title}: {default_title_suffix}"
    else:
        title = default_title_suffix

    plot_by_stage(
        stats=stats,
        category_order=category_order,
        title=title,
        save_path=output_png,
        early_end=early_end,
        middle_end=middle_end,
        n_layers=n_layers,
        ylabel=default_ylabel,
    )


def _maybe_default_signed_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.name.endswith(".json"):
        candidate = path.with_name(path.stem + "_signed.json")
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot refined token-analysis figures from saved stats json")
    parser.add_argument("--sum_stats_json", type=Path, default=None)
    parser.add_argument("--mean_stats_json", type=Path, default=None)
    parser.add_argument("--sum_signed_stats_json", type=Path, default=None)
    parser.add_argument("--mean_signed_stats_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.sum_stats_json is None and args.mean_stats_json is None:
        raise ValueError("At least one of --sum_stats_json or --mean_stats_json must be provided")

    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_dir, filename="plot_refined_analysis.log"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sum_signed_stats_json = args.sum_signed_stats_json or _maybe_default_signed_path(args.sum_stats_json)
    mean_signed_stats_json = args.mean_signed_stats_json or _maybe_default_signed_path(args.mean_stats_json)

    if args.sum_stats_json is not None:
        _plot_one(
            args.sum_stats_json,
            args.output_dir / "refined_by_stage_sum.png",
            default_title_suffix="Sum |dPD|",
            default_ylabel="Sum |dPD| (Mean +/- SEM)",
        )
    if args.mean_stats_json is not None:
        _plot_one(
            args.mean_stats_json,
            args.output_dir / "refined_by_stage_mean.png",
            default_title_suffix="Mean |dPD| per token",
            default_ylabel="Mean |dPD| (Mean +/- SEM)",
        )
    if sum_signed_stats_json is not None:
        _plot_one(
            sum_signed_stats_json,
            args.output_dir / "refined_by_stage_sum_signed.png",
            default_title_suffix="Sum dPD",
            default_ylabel="Sum dPD (Mean +/- SEM)",
        )
    if mean_signed_stats_json is not None:
        _plot_one(
            mean_signed_stats_json,
            args.output_dir / "refined_by_stage_mean_signed.png",
            default_title_suffix="Mean dPD per token",
            default_ylabel="Mean dPD (Mean +/- SEM)",
        )

    log_event(
        logger,
        {
            "sum_stats_json": str(args.sum_stats_json) if args.sum_stats_json is not None else "",
            "mean_stats_json": str(args.mean_stats_json) if args.mean_stats_json is not None else "",
            "sum_signed_stats_json": str(sum_signed_stats_json) if sum_signed_stats_json is not None else "",
            "mean_signed_stats_json": str(mean_signed_stats_json) if mean_signed_stats_json is not None else "",
            "output_dir": str(args.output_dir),
        },
    )


if __name__ == "__main__":
    main()
