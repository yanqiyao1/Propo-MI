from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.progress import log_event, resolve_log_path, setup_file_logger
from src.plot_style import apply_paper_style, stylize_axis, stylize_colorbar


METRICS = ("BMI", "BCR", "SBI")
BANDS = ("early", "middle", "late")
REGIONS = ("facts_region", "expression_region", "query_region")
REGION_LABELS = {
    "facts_region": "Facts",
    "expression_region": "Expression",
    "query_region": "Query",
}
METRIC_TITLES = {
    "BMI": "Band Mean Impact (BMI)",
    "BCR": "Band Contribution Ratio (BCR)",
    "SBI": "Signed Band Impact (SBI)",
}
CMAPS = {
    "BMI": "Blues",
    "BCR": "Oranges",
    "SBI": "RdBu_r",
}
STACK_COLORS = {
    "early": "#4C78A8",
    "middle": "#F58518",
    "late": "#54A24B",
}


def _read_metric_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Matrix csv is empty: {path}")

    row_map = {str(row["region"]): row for row in rows}
    matrix = np.zeros((len(REGIONS), len(BANDS)), dtype=np.float64)
    for i, region in enumerate(REGIONS):
        if region not in row_map:
            raise KeyError(f"Missing region {region!r} in matrix {path}")
        for j, band in enumerate(BANDS):
            matrix[i, j] = float(row_map[region][band])
    return matrix


def _load_hop_dir(base_dir: Path, label: str) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "label": label,
        "base_dir": str(base_dir),
        "matrices": {},
    }
    for metric in METRICS:
        path = base_dir / f"{metric}_matrix.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {metric} matrix csv: {path}")
        matrices = payload["matrices"]
        assert isinstance(matrices, dict)
        matrices[metric] = _read_metric_matrix(path)
    return payload


def _text_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return "black"
    norm = (value - vmin) / (vmax - vmin)
    return "white" if norm > 0.55 else "black"


def _plot_panel(hop_payloads: Sequence[Dict[str, object]], output_dir: Path, title: str = "") -> Tuple[Path, Path]:
    nrows = len(hop_payloads)
    ncols = len(METRICS)
    if nrows == 0:
        raise ValueError("No hop payloads provided")

    apply_paper_style(
        {
            "font.size": 13.5,
            "axes.titlesize": 16.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
        }
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    if title.strip():
        fig.suptitle(title, y=0.995)

    metric_ranges: Dict[str, Tuple[float, float]] = {}
    for metric in METRICS:
        arrays = [np.asarray(hp["matrices"][metric], dtype=np.float64) for hp in hop_payloads]
        arr = np.stack(arrays, axis=0)
        if metric == "SBI":
            vmax = float(np.max(np.abs(arr))) if arr.size else 1.0
            vmax = max(vmax, 1e-9)
            metric_ranges[metric] = (-vmax, vmax)
        elif metric == "BCR":
            metric_ranges[metric] = (0.0, 1.0)
        else:
            metric_ranges[metric] = (0.0, float(np.max(arr)) if arr.size else 1.0)

    last_images = {}
    for row_idx, hop_payload in enumerate(hop_payloads):
        label = str(hop_payload["label"])
        matrices = hop_payload["matrices"]
        assert isinstance(matrices, dict)

        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            matrix = np.asarray(matrices[metric], dtype=np.float64)
            vmin, vmax = metric_ranges[metric]
            image = ax.imshow(matrix, cmap=CMAPS[metric], vmin=vmin, vmax=vmax, aspect="auto")
            last_images[metric] = image

            if row_idx == 0:
                ax.set_title(METRIC_TITLES[metric])
            if col_idx == 0:
                ax.set_ylabel(label)

            ax.set_xticks(np.arange(len(BANDS)))
            ax.set_xticklabels([band.capitalize() for band in BANDS])
            ax.set_yticks(np.arange(len(REGIONS)))
            ax.set_yticklabels([REGION_LABELS[r] for r in REGIONS])

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = float(matrix[i, j])
                    text = f"{value:+.3f}" if metric == "SBI" else f"{value:.3f}"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color=_text_color(value, vmin=vmin, vmax=vmax),
                        fontsize=11.0,
                        fontweight="semibold",
                    )
            stylize_axis(ax)

    for col_idx, metric in enumerate(METRICS):
        cbar = fig.colorbar(last_images[metric], ax=axes[:, col_idx], fraction=0.045, pad=0.03)
        stylize_colorbar(cbar)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "band_metrics_panel.png"
    pdf_path = output_dir / "band_metrics_panel.pdf"
    fig.savefig(png_path, dpi=320)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _plot_bcr_stacked(hop_payloads: Sequence[Dict[str, object]], output_dir: Path) -> Tuple[Path, Path]:
    nrows = len(hop_payloads)
    apply_paper_style(
        {
            "font.size": 13.5,
            "axes.titlesize": 15.5,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
        }
    )

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8.2, 4.2 * nrows), squeeze=False)
    for row_idx, hop_payload in enumerate(hop_payloads):
        ax = axes[row_idx, 0]
        label = str(hop_payload["label"])
        matrices = hop_payload["matrices"]
        assert isinstance(matrices, dict)
        matrix = np.asarray(matrices["BCR"], dtype=np.float64)

        x = np.arange(len(REGIONS))
        bottom = np.zeros(len(REGIONS), dtype=np.float64)
        for band_idx, band in enumerate(BANDS):
            values = matrix[:, band_idx]
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=STACK_COLORS[band],
                edgecolor="white",
                linewidth=0.7,
                label=band.capitalize(),
            )
            bottom += values

        ax.set_title(f"{label}: BCR Composition by Region")
        ax.set_ylabel("Contribution Ratio")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS])
        ax.grid(axis="y", alpha=0.25)
        if row_idx == 0:
            ax.legend(loc="upper right", ncol=3)
        stylize_axis(ax)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "bcr_stacked.png"
    pdf_path = output_dir / "bcr_stacked.pdf"
    fig.savefig(png_path, dpi=320)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MLP region band metrics panel and BCR stacked charts")
    parser.add_argument("--one_hop_dir", type=Path, default=None)
    parser.add_argument("--two_hop_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--title", default="", help="Optional panel title; empty disables suptitle.")
    args = parser.parse_args()
    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_dir, filename="plot_band_metrics.log"))

    hop_payloads: List[Dict[str, object]] = []
    if args.one_hop_dir is not None:
        hop_payloads.append(_load_hop_dir(args.one_hop_dir, label="One-hop"))
    if args.two_hop_dir is not None:
        hop_payloads.append(_load_hop_dir(args.two_hop_dir, label="Two-hop"))
    if not hop_payloads:
        raise ValueError("At least one of --one_hop_dir or --two_hop_dir must be provided")

    panel_png, panel_pdf = _plot_panel(hop_payloads, output_dir=args.output_dir, title=args.title)
    bcr_png, bcr_pdf = _plot_bcr_stacked(hop_payloads, output_dir=args.output_dir)

    log_event(
        logger,
        {
            "output_dir": str(args.output_dir),
            "panel_png": str(panel_png),
            "panel_pdf": str(panel_pdf),
            "bcr_png": str(bcr_png),
            "bcr_pdf": str(bcr_pdf),
        },
    )


if __name__ == "__main__":
    main()
