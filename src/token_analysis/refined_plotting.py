from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.plot_style import apply_paper_style, stylize_axis
from .refined_token_classifier import CATEGORY_DISPLAY_NAMES


STAGE_COLORS = {"early": "#3498DB", "middle": "#9B59B6", "late": "#E67E22"}


def plot_by_stage(
    *,
    stats: Dict[str, Dict[str, Dict[str, float]]],
    category_order: Sequence[str],
    title: str,
    save_path: Path,
    early_end: int,
    middle_end: int,
    n_layers: int,
    ylabel: str,
) -> None:
    cats = [c for c in category_order if c in stats]
    cat_labels = [CATEGORY_DISPLAY_NAMES.get(c, c) for c in cats]
    stages = ["early", "middle", "late"]

    if not cats:
        raise ValueError("No categories available to plot")

    apply_paper_style(
        {
            "font.size": 14.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 14.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.5,
        }
    )

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    x = np.arange(len(cats))
    width = 0.25

    stage_labels = {
        "early": f"Early (L0-{early_end - 1})",
        "middle": f"Middle (L{early_end}-{middle_end - 1})",
        "late": f"Late (L{middle_end}-{n_layers - 1})",
    }

    for i, stage in enumerate(stages):
        means = [float(stats[c][stage]["mean"]) for c in cats]
        sems = [float(stats[c][stage]["sem"]) for c in cats]
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            label=stage_labels[stage],
            color=STAGE_COLORS[stage],
            alpha=0.85,
            edgecolor="black",
            linewidth=1.1,
            capsize=4,
        )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    stylize_axis(ax)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
