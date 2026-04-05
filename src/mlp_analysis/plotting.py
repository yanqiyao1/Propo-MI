from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.plot_style import apply_paper_style


def plot_bar(output_png: Path, layers: Sequence[int], scores: Sequence[float], *, ylabel: str = "dPD") -> None:
    apply_paper_style(
        {
            "font.size": 13.0,
            "axes.titlesize": 15.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
        }
    )

    plt.figure(figsize=(10, 6))
    plt.bar(layers, scores, color="C0")
    plt.xlabel("Layer Index")
    plt.ylabel(ylabel)
    if layers:
        plt.xticks(np.arange(0, max(layers) + 1, 5))
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
