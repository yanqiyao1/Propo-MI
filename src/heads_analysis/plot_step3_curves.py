from __future__ import annotations

import argparse
from pathlib import Path

from src.progress import log_event, resolve_log_path, setup_file_logger

from .plot_only import plot_pd_curves, read_csv


def _resolve_k_values(rows):
    return sorted({int(float(row.get("k", 0))) for row in rows if int(float(row.get("k", 0))) > 0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step3 validation curves from saved pd_curve_metrics.csv")
    parser.add_argument("--pd_curve_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--include_signed_ratio_plot",
        "--include-signed-ratio-plot",
        dest="include_signed_ratio_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate pd_signed_ratio_curve.png. Disabled by default.",
    )
    args = parser.parse_args()

    logger = setup_file_logger(__name__, resolve_log_path(output_dir=args.output_dir, filename="plot_step3_curves.log"))
    rows = read_csv(args.pd_curve_csv)
    if not rows:
        raise ValueError(f"No rows found in pd curve csv: {args.pd_curve_csv}")
    k_values = _resolve_k_values(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    abs_ratio_plot = args.output_dir / "pd_abs_ratio_curve.png"
    signed_ratio_plot = args.output_dir / "pd_signed_ratio_curve.png"
    dpd_shift_plot = args.output_dir / "dpd_shift_curve.png"

    plot_pd_curves(
        rows,
        abs_ratio_plot,
        y_mean_key="mean_abs_relative_dpd_mean",
        y_sem_key="mean_abs_relative_dpd_sem",
        ylabel="|dPD| / |PD_original|",
        k_values=k_values,
    )
    if args.include_signed_ratio_plot:
        plot_pd_curves(
            rows,
            signed_ratio_plot,
            y_mean_key="mean_signed_relative_dpd_mean",
            y_sem_key="mean_signed_relative_dpd_sem",
            ylabel="dPD / PD_original",
            k_values=k_values,
        )
    plot_pd_curves(
        rows,
        dpd_shift_plot,
        y_mean_key="mean_dpd_shift_mean",
        y_sem_key="mean_dpd_shift_sem",
        ylabel="dPD",
        k_values=k_values,
    )
    log_event(
        logger,
        {
            "pd_curve_csv": str(args.pd_curve_csv),
            "output_dir": str(args.output_dir),
            "k_values": k_values,
            "include_signed_ratio_plot": bool(args.include_signed_ratio_plot),
        },
    )


if __name__ == "__main__":
    main()
