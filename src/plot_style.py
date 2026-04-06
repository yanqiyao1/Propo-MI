from __future__ import annotations

from typing import Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar


# Centralized plotting style for paper-quality figures.
PAPER_RC_PARAMS: dict[str, object] = {
    "figure.dpi": 160,
    "savefig.dpi": 360,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 14.0,
    "font.weight": "semibold",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 17.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 15.0,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "xtick.major.width": 0.95,
    "ytick.major.width": 0.95,
    "xtick.major.size": 4.2,
    "ytick.major.size": 4.2,
    "legend.fontsize": 12.5,
    "legend.title_fontsize": 13.0,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#B8B8B8",
    "lines.linewidth": 2.2,
    "lines.markersize": 6.0,
    "patch.linewidth": 0.95,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style(overrides: Mapping[str, object] | None = None) -> None:
    params = dict(PAPER_RC_PARAMS)
    if overrides:
        params.update(dict(overrides))
    plt.rcParams.update(params)


def _set_label_weight(labels: Iterable[object], weight: str) -> None:
    for label in labels:
        if hasattr(label, "set_fontweight"):
            label.set_fontweight(weight)


def stylize_axis(
    ax: Axes,
    *,
    tick_weight: str = "semibold",
    label_weight: str = "bold",
    title_weight: str = "bold",
) -> None:
    ax.xaxis.label.set_fontweight(label_weight)
    ax.yaxis.label.set_fontweight(label_weight)
    ax.title.set_fontweight(title_weight)
    _set_label_weight(ax.get_xticklabels(), tick_weight)
    _set_label_weight(ax.get_yticklabels(), tick_weight)

    legend = ax.get_legend()
    if legend is not None:
        _set_label_weight(legend.get_texts(), tick_weight)
        legend_title = legend.get_title()
        if legend_title is not None:
            legend_title.set_fontweight(label_weight)


def stylize_colorbar(
    colorbar: Colorbar,
    *,
    tick_weight: str = "semibold",
    label_weight: str = "bold",
) -> None:
    colorbar.ax.yaxis.label.set_fontweight(label_weight)
    _set_label_weight(colorbar.ax.get_yticklabels(), tick_weight)
