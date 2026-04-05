from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.plot_style import apply_paper_style
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
            "font.size": 12.5,
            "axes.titlesize": 14.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))
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
            linewidth=1.0,
            capsize=3,
        )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
