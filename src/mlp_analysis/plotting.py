from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.plot_style import apply_paper_style, stylize_axis


def plot_bar(output_png: Path, layers: Sequence[int], scores: Sequence[float], *, ylabel: str = "dPD") -> None:
    apply_paper_style(
        {
            "font.size": 14.5,
            "axes.titlesize": 16.5,
            "axes.labelsize": 15.0,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
        }
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.bar(layers, scores, color="#4C78A8", edgecolor="#1F1F1F", linewidth=0.9)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(ylabel)
    if layers:
        ax.set_xticks(np.arange(0, max(layers) + 1, 5))
    ax.grid(axis="y", alpha=0.28)
    stylize_axis(ax)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320)
    plt.close(fig)
