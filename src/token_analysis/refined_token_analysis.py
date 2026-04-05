from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.progress import log_event, make_tqdm, resolve_log_path, setup_file_logger

from .refined_plotting import plot_by_stage
from .result_schema import get_delta_score_matrix, infer_delta_score_n_layers
from .refined_token_classifier import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_DISPLAY_NAMES,
    CATEGORY_ORDER,
    classify_tokens_refined,
)


EXPR_FIRST_EXCLUDED_REFINED_CATEGORIES = {"variable_in_expr", "operator", "expr_last"}


def _load_results(path: Path) -> List[Dict[str, object]]:
    with path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("results", "data", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise ValueError(f"Unsupported pickle payload type: {type(obj).__name__}")


def _infer_n_layers(results: Sequence[Dict[str, object]], fallback: int = 0) -> int:
    return infer_delta_score_n_layers(results, fallback=fallback)


def _stage_slices(early_end: int, middle_end: int, n_layers: int) -> Dict[str, Tuple[int, int]]:
    if not (0 < early_end < middle_end < n_layers):
        raise ValueError(
            f"Require 0 < early_end < middle_end < n_layers, got "
            f"{early_end}, {middle_end}, {n_layers}"
        )
    return {
        "early": (0, early_end),
        "middle": (early_end, middle_end),
        "late": (middle_end, n_layers),
    }


def _empty_stats(category_order: Sequence[str], stages: Iterable[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cat in category_order:
        out[cat] = {}
        for stage in stages:
            out[cat][stage] = {"mean": 0.0, "sem": 0.0, "n": 0}
    return out


def _finalize_stats(
    data: Dict[str, Dict[str, List[float]]],
    category_order: Sequence[str],
    stage_names: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats = _empty_stats(category_order, stage_names)
    for cat in category_order:
        for stage in stage_names:
            vals = np.array(data.get(cat, {}).get(stage, []), dtype=np.float64)
            if vals.size == 0:
                continue
            stats[cat][stage] = {
                "mean": float(vals.mean()),
                "sem": float(vals.std(ddof=0) / np.sqrt(vals.size)),
                "n": int(vals.size),
            }
    return stats


def _compute_totals(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    category_order: Sequence[str],
    stage_names: Sequence[str],
) -> Dict[str, float]:
    totals = {}
    for cat in category_order:
        totals[cat] = float(sum(float(stats[cat][st]["mean"]) for st in stage_names))
    return totals


def aggregate_refined(
    results: Sequence[Dict[str, object]],
    *,
    mode: str,
    early_end: int,
    middle_end: int,
    n_layers: int,
    include_derived_assignment: bool,
    strict: bool,
    progress_desc: str,
    exclude_categories: Sequence[str] | None = None,
    absolute: bool = True,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, object]]:
    if mode not in {"sum", "mean"}:
        raise ValueError(f"Unknown mode {mode!r}")

    stages = _stage_slices(early_end=early_end, middle_end=middle_end, n_layers=n_layers)
    stage_names = list(stages.keys())
    data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    excluded = set(exclude_categories or [])

    category_order = [c for c in CATEGORY_ORDER if (include_derived_assignment or c != "derived_assignment") and c not in excluded]
    prompt_order_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    category_token_counts: Counter[str] = Counter()
    warnings_preview: List[str] = []
    warning_count = 0
    skipped = 0
    used = 0

    for row in make_tqdm(results, total=len(results), desc=progress_desc, leave=False, disable=len(results) <= 1):
        str_tokens = row.get("str_tokens")
        delta_score = get_delta_score_matrix(row)
        sample = row.get("sample", {})
        if not isinstance(str_tokens, (list, tuple)):
            skipped += 1
            skip_reasons["missing_str_tokens"] += 1
            continue
        if delta_score is None:
            skipped += 1
            skip_reasons["missing_delta_score"] += 1
            continue

        arr = np.asarray(delta_score)
        if arr.ndim != 2:
            skipped += 1
            skip_reasons["delta_score_not_2d"] += 1
            continue

        if not isinstance(sample, dict):
            sample = {}

        categories, meta = classify_tokens_refined(
            str_tokens=str_tokens,
            sample=sample,
            include_derived_assignment=include_derived_assignment,
            strict=strict,
        )
        for w in meta.get("warnings", []):
            warning_count += 1
            if len(warnings_preview) < 20:
                warnings_preview.append(str(w))

        if not bool(meta.get("is_valid", False)):
            skipped += 1
            skip_reasons["invalid_classification"] += 1
            continue

        prompt_order_counts[str(meta.get("prompt_order_detected", "unknown"))] += 1
        used += 1

        pos_count = min(arr.shape[1], len(categories))
        layer_count = min(arr.shape[0], n_layers)
        if pos_count <= 0 or layer_count <= 0:
            skipped += 1
            used -= 1
            skip_reasons["empty_tensor_after_trim"] += 1
            continue

        value_arr = arr[:layer_count, :pos_count]
        if absolute:
            value_arr = np.abs(value_arr)

        cat_positions: Dict[str, List[int]] = defaultdict(list)
        for pos in range(pos_count):
            cat = categories[pos]
            if cat not in category_order:
                continue
            cat_positions[cat].append(pos)
            category_token_counts[cat] += 1

        for cat, positions in cat_positions.items():
            if not positions:
                continue
            for stage_name, (start, end) in stages.items():
                if start >= layer_count:
                    continue
                end_clamped = min(end, layer_count)
                per_pos = [float(value_arr[start:end_clamped, pos].mean()) for pos in positions]
                value = float(sum(per_pos)) if mode == "sum" else float(np.mean(per_pos))
                data[cat][stage_name].append(value)

    stats = _finalize_stats(data=data, category_order=category_order, stage_names=stage_names)
    totals = _compute_totals(stats=stats, category_order=category_order, stage_names=stage_names)

    metadata = {
        "mode": mode,
        "total_samples": len(results),
        "used_samples": used,
        "skipped_samples": skipped,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "prompt_order_counts": dict(sorted(prompt_order_counts.items())),
        "category_token_counts": dict(sorted(category_token_counts.items())),
        "warning_count": warning_count,
        "warnings_preview": warnings_preview,
        "category_order": category_order,
        "category_labels": {cat: CATEGORY_DISPLAY_NAMES.get(cat, cat) for cat in category_order},
        "category_descriptions": {cat: CATEGORY_DESCRIPTIONS.get(cat, "") for cat in category_order},
        "stage_bounds": {
            "early": [0, early_end - 1],
            "middle": [early_end, middle_end - 1],
            "late": [middle_end, n_layers - 1],
        },
        "absolute": bool(absolute),
        "totals": totals,
    }
    return stats, metadata


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _infer_prompt_orders(results: Sequence[Dict[str, object]]) -> set[str]:
    prompt_orders: set[str] = set()
    for row in results:
        sample = row.get("sample", {})
        if isinstance(sample, dict):
            value = str(sample.get("prompt_order", "")).strip()
            if value:
                prompt_orders.add(value)
    return prompt_orders


def _excluded_categories_for_results(results: Sequence[Dict[str, object]]) -> set[str]:
    prompt_orders = _infer_prompt_orders(results)
    if prompt_orders == {"expr_first"}:
        return set(EXPR_FIRST_EXCLUDED_REFINED_CATEGORIES)
    return set()


def _write_stats_csv(
    *,
    path: Path,
    stats: Dict[str, Dict[str, Dict[str, float]]],
    totals: Dict[str, float],
    category_order: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "category_label", "category_description", "stage", "mean", "sem", "n", "total"],
        )
        writer.writeheader()
        for cat in category_order:
            for stage in ("early", "middle", "late"):
                row = stats[cat][stage]
                writer.writerow(
                    {
                        "category": cat,
                        "category_label": CATEGORY_DISPLAY_NAMES.get(cat, cat),
                        "category_description": CATEGORY_DESCRIPTIONS.get(cat, ""),
                        "stage": stage,
                        "mean": row["mean"],
                        "sem": row["sem"],
                        "n": row["n"],
                        "total": totals[cat],
                    }
                )


def _print_table(
    *,
    stats: Dict[str, Dict[str, Dict[str, float]]],
    category_order: Sequence[str],
    mode: str,
    logger,
) -> None:
    log_event(logger, f"[{mode.upper()}] Category           | Early    | Middle   | Late     | Total")
    log_event(logger, "-" * 74)
    for cat in category_order:
        early = float(stats[cat]["early"]["mean"])
        middle = float(stats[cat]["middle"]["mean"])
        late = float(stats[cat]["late"]["mean"])
        total = early + middle + late
        label = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        log_event(logger, f"{label:20s} | {early:8.3f} | {middle:8.3f} | {late:8.3f} | {total:8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refined token-category analysis from patching_results.pkl")
    parser.add_argument("--input_pkl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Refined Token Analysis")
    parser.add_argument("--early_end", type=int, default=14)
    parser.add_argument("--middle_end", type=int, default=24)
    parser.add_argument(
        "--n_layers",
        type=int,
        default=0,
        help="Layer count override. Use 0 to auto-infer from input pkl.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--include-derived-assignment",
        "--include_derived_assignment",
        dest="include_derived_assignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-plots",
        "--save_plots",
        dest="save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-csv",
        "--save_csv",
        dest="save_csv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_dir, filename="refined_token_analysis.log"))

    rows = _load_results(args.input_pkl)
    excluded_categories = _excluded_categories_for_results(rows)

    resolved_n_layers = int(args.n_layers) if int(args.n_layers) > 0 else _infer_n_layers(rows, fallback=0)
    if resolved_n_layers <= 0:
        raise ValueError(
            "Unable to infer n_layers from input pkl. "
            "Please pass --n_layers explicitly."
        )

    sum_stats, sum_meta = aggregate_refined(
        rows,
        mode="sum",
        early_end=args.early_end,
        middle_end=args.middle_end,
        n_layers=resolved_n_layers,
        include_derived_assignment=args.include_derived_assignment,
        strict=args.strict,
        progress_desc="refined-sum",
        exclude_categories=excluded_categories,
        absolute=True,
    )
    mean_stats, mean_meta = aggregate_refined(
        rows,
        mode="mean",
        early_end=args.early_end,
        middle_end=args.middle_end,
        n_layers=resolved_n_layers,
        include_derived_assignment=args.include_derived_assignment,
        strict=args.strict,
        progress_desc="refined-mean",
        exclude_categories=excluded_categories,
        absolute=True,
    )
    signed_sum_stats, signed_sum_meta = aggregate_refined(
        rows,
        mode="sum",
        early_end=args.early_end,
        middle_end=args.middle_end,
        n_layers=resolved_n_layers,
        include_derived_assignment=args.include_derived_assignment,
        strict=args.strict,
        progress_desc="refined-sum-signed",
        exclude_categories=excluded_categories,
        absolute=False,
    )
    signed_mean_stats, signed_mean_meta = aggregate_refined(
        rows,
        mode="mean",
        early_end=args.early_end,
        middle_end=args.middle_end,
        n_layers=resolved_n_layers,
        include_derived_assignment=args.include_derived_assignment,
        strict=args.strict,
        progress_desc="refined-mean-signed",
        exclude_categories=excluded_categories,
        absolute=False,
    )

    category_order = list(sum_meta["category_order"])
    sum_totals = dict(sum_meta["totals"])
    mean_totals = dict(mean_meta["totals"])
    signed_sum_totals = dict(signed_sum_meta["totals"])
    signed_mean_totals = dict(signed_mean_meta["totals"])
    category_labels = {cat: CATEGORY_DISPLAY_NAMES.get(cat, cat) for cat in category_order}
    category_descriptions = {cat: CATEGORY_DESCRIPTIONS.get(cat, "") for cat in category_order}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(
        args.output_dir / "refined_stats_sum.json",
        {
            "title": args.title,
            "mode": "sum",
            "value_type": "abs_dpd",
            "stats": sum_stats,
            "totals": sum_totals,
            "category_labels": category_labels,
            "category_descriptions": category_descriptions,
            "excluded_categories": sorted(excluded_categories),
            "metadata": sum_meta,
        },
    )
    _save_json(
        args.output_dir / "refined_stats_mean.json",
        {
            "title": args.title,
            "mode": "mean",
            "value_type": "abs_dpd",
            "stats": mean_stats,
            "totals": mean_totals,
            "category_labels": category_labels,
            "category_descriptions": category_descriptions,
            "excluded_categories": sorted(excluded_categories),
            "metadata": mean_meta,
        },
    )
    _save_json(
        args.output_dir / "refined_stats_sum_signed.json",
        {
            "title": args.title,
            "mode": "sum",
            "value_type": "dpd",
            "stats": signed_sum_stats,
            "totals": signed_sum_totals,
            "category_labels": category_labels,
            "category_descriptions": category_descriptions,
            "excluded_categories": sorted(excluded_categories),
            "metadata": signed_sum_meta,
        },
    )
    _save_json(
        args.output_dir / "refined_stats_mean_signed.json",
        {
            "title": args.title,
            "mode": "mean",
            "value_type": "dpd",
            "stats": signed_mean_stats,
            "totals": signed_mean_totals,
            "category_labels": category_labels,
            "category_descriptions": category_descriptions,
            "excluded_categories": sorted(excluded_categories),
            "metadata": signed_mean_meta,
        },
    )
    _save_json(
        args.output_dir / "refined_metadata.json",
        {
            "input_pkl": str(args.input_pkl),
            "title": args.title,
            "n_rows": len(rows),
            "config": {
                "early_end": args.early_end,
                "middle_end": args.middle_end,
                "n_layers": resolved_n_layers,
                "strict": args.strict,
                "include_derived_assignment": args.include_derived_assignment,
            },
            "category_order": category_order,
            "category_labels": category_labels,
            "category_descriptions": category_descriptions,
            "excluded_categories": sorted(excluded_categories),
            "sum": sum_meta,
            "mean": mean_meta,
            "signed_sum": signed_sum_meta,
            "signed_mean": signed_mean_meta,
        },
    )

    if args.save_csv:
        _write_stats_csv(
            path=args.output_dir / "refined_stats_sum.csv",
            stats=sum_stats,
            totals=sum_totals,
            category_order=category_order,
        )
        _write_stats_csv(
            path=args.output_dir / "refined_stats_mean.csv",
            stats=mean_stats,
            totals=mean_totals,
            category_order=category_order,
        )
        _write_stats_csv(
            path=args.output_dir / "refined_stats_sum_signed.csv",
            stats=signed_sum_stats,
            totals=signed_sum_totals,
            category_order=category_order,
        )
        _write_stats_csv(
            path=args.output_dir / "refined_stats_mean_signed.csv",
            stats=signed_mean_stats,
            totals=signed_mean_totals,
            category_order=category_order,
        )

    if args.save_plots:
        plot_by_stage(
            stats=sum_stats,
            category_order=category_order,
            title=f"{args.title}: Sum |dPD|",
            save_path=args.output_dir / "refined_by_stage_sum.png",
            early_end=args.early_end,
            middle_end=args.middle_end,
            n_layers=resolved_n_layers,
            ylabel="Sum |dPD| (Mean +/- SEM)",
        )
        plot_by_stage(
            stats=mean_stats,
            category_order=category_order,
            title=f"{args.title}: Mean |dPD| per token",
            save_path=args.output_dir / "refined_by_stage_mean.png",
            early_end=args.early_end,
            middle_end=args.middle_end,
            n_layers=resolved_n_layers,
            ylabel="Mean |dPD| (Mean +/- SEM)",
        )
        plot_by_stage(
            stats=signed_sum_stats,
            category_order=category_order,
            title=f"{args.title}: Sum dPD",
            save_path=args.output_dir / "refined_by_stage_sum_signed.png",
            early_end=args.early_end,
            middle_end=args.middle_end,
            n_layers=resolved_n_layers,
            ylabel="Sum dPD (Mean +/- SEM)",
        )
        plot_by_stage(
            stats=signed_mean_stats,
            category_order=category_order,
            title=f"{args.title}: Mean dPD per token",
            save_path=args.output_dir / "refined_by_stage_mean_signed.png",
            early_end=args.early_end,
            middle_end=args.middle_end,
            n_layers=resolved_n_layers,
            ylabel="Mean dPD (Mean +/- SEM)",
        )

    _print_table(stats=sum_stats, category_order=category_order, mode="sum", logger=logger)
    _print_table(stats=mean_stats, category_order=category_order, mode="mean", logger=logger)
    log_event(
        logger,
        {
            "input_pkl": str(args.input_pkl),
            "rows": len(rows),
            "output_dir": str(args.output_dir),
            "used_samples_sum": sum_meta["used_samples"],
            "used_samples_mean": mean_meta["used_samples"],
            "excluded_categories": sorted(excluded_categories),
        },
    )


if __name__ == "__main__":
    main()
