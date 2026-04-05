from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt


# Centralized plotting style for paper-quality figures.
PAPER_RC_PARAMS: dict[str, object] = {
    "figure.dpi": 150,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 12.0,
    "font.weight": "regular",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 14.0,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12.0,
    "axes.labelweight": "semibold",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.fontsize": 10.5,
    "legend.title_fontsize": 11.0,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#B8B8B8",
    "lines.linewidth": 1.8,
    "lines.markersize": 5.0,
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

