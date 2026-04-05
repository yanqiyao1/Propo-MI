"""Step 2 - taxonomy counts + optional plotting from Step1 outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.progress import log_event, resolve_log_path, setup_file_logger

from .common import ROLE_COLOR, ROLE_LABEL, ROLE_ORDER, apply_paper_style, read_csv, resolve_role_label, write_csv, write_json


def build_taxonomy_count_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    n_layers: int,
    role_col: str = "role_label",
) -> List[Dict[str, object]]:
    layer_role_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        layer = int(float(row.get("layer", 0)))
        role = resolve_role_label(row, role_col=role_col)
        if role is None:
            continue
        layer_role_counts[layer][role] += 1

    count_rows: List[Dict[str, object]] = []
    for layer in range(n_layers):
        record: Dict[str, object] = {"layer": layer}
        total = 0
        for role in ROLE_ORDER:
            value = int(layer_role_counts[layer].get(role, 0))
            record[role] = value
            total += value
        record["total"] = total
        count_rows.append(record)
    return count_rows


def plot_taxonomy_line_chart_from_counts(
    count_rows: Sequence[Mapping[str, object]],
    output_png: Path,
) -> None:
    if not count_rows:
        raise ValueError("No taxonomy count rows to plot")

    layers = [int(float(row.get("layer", 0))) for row in count_rows]
    fact = [int(float(row.get("fact_retrieval", 0))) for row in count_rows]
    split = [int(float(row.get("splitting", 0))) for row in count_rows]
    trans = [int(float(row.get("transmission", 0))) for row in count_rows]
    n_layers = max(layers) + 1 if layers else 0

    apply_paper_style({"axes.grid": True, "grid.alpha": 0.22})
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(layers, fact, label=ROLE_LABEL["fact_retrieval"], marker="o", color=ROLE_COLOR["fact_retrieval"])
    ax.plot(layers, split, label=ROLE_LABEL["splitting"], marker="s", color=ROLE_COLOR["splitting"])
    ax.plot(layers, trans, label=ROLE_LABEL["transmission"], marker="^", color=ROLE_COLOR["transmission"])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head count")
    if layers:
        ax.set_xticks(np.arange(0, max(layers) + 1, max(1, n_layers // 10)))
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320)
    plt.close(fig)


def plot_taxonomy_line_chart(
    rows: Sequence[Mapping[str, object]],
    output_png: Path,
    n_layers: int,
    role_col: str = "role_label",
) -> None:
    count_rows = build_taxonomy_count_rows(rows, n_layers=n_layers, role_col=role_col)
    plot_taxonomy_line_chart_from_counts(count_rows, output_png)


def run_step2(
    classify_csv: Path,
    output_dir: Path,
    n_layers: int = 0,
    role_col: str = "role_label",
    save_plots: bool = True,
) -> Dict[str, object]:
    logger = setup_file_logger(__name__, resolve_log_path(output_dir=output_dir, filename="step2_taxonomy.log"))
    rows = read_csv(classify_csv)
    if not rows:
        raise ValueError(f"No rows in classify csv: {classify_csv}")

    if n_layers <= 0:
        n_layers = max(int(float(r.get("layer", 0))) for r in rows) + 1

    count_rows = build_taxonomy_count_rows(rows, n_layers=n_layers, role_col=role_col)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts_csv = output_dir / "head_taxonomy_counts.csv"
    write_csv(counts_csv, count_rows)

    png_path = output_dir / "head_taxonomy_line_chart.png"
    if save_plots:
        plot_taxonomy_line_chart_from_counts(count_rows, png_path)

    role_totals = {
        role: int(sum(int(float(row.get(role, 0))) for row in count_rows))
        for role in ROLE_ORDER
    }
    summary = {
        "n_layers": n_layers,
        "n_classified_heads": int(sum(role_totals.values())),
        "role_totals": role_totals,
        "counts_csv": str(counts_csv),
        "plot_png": str(png_path),
        "plot_generated": bool(save_plots),
    }
    write_json(output_dir / "summary.json", summary)
    log_event(logger, {"stage": "step2_done", "output_dir": str(output_dir), "plot_generated": bool(save_plots)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step2: taxonomy counts with optional line chart")
    parser.add_argument("--classify_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_layers", type=int, default=0)
    parser.add_argument("--role_col", default="role_label")
    parser.add_argument(
        "--save_plots",
        "--save-plots",
        dest="save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    run_step2(
        classify_csv=args.classify_csv,
        output_dir=args.output_dir,
        n_layers=args.n_layers,
        role_col=args.role_col,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
