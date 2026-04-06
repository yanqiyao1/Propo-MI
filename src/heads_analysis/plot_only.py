from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.plot_style import apply_paper_style, stylize_axis


ROLE_ORDER = ["fact_retrieval", "splitting", "transmission"]
ROLE_LABEL = {
    "fact_retrieval": "Fact-Retrieval",
    "splitting": "Splitting",
    "transmission": "Transmission",
}
ROLE_COLOR = {
    "fact_retrieval": "#D55E00",
    "splitting": "#0072B2",
    "transmission": "#009E73",
}
ROLE_SCORE_COLUMNS = {
    "fact_retrieval": "score_fact",
    "splitting": "score_split",
    "transmission": "score_trans",
}

STRATEGY_ORDER = [
    "single_fact_retrieval",
    "single_splitting",
    "single_transmission",
    "pair_fact_retrieval__splitting",
    "pair_fact_retrieval__transmission",
    "pair_splitting__transmission",
    "mixed_all",
    "random_outside_selected",
]
STRATEGY_LABEL = {
    "single_fact_retrieval": ROLE_LABEL["fact_retrieval"],
    "single_splitting": ROLE_LABEL["splitting"],
    "single_transmission": ROLE_LABEL["transmission"],
    "pair_fact_retrieval__splitting": f'{ROLE_LABEL["fact_retrieval"]} + {ROLE_LABEL["splitting"]}',
    "pair_fact_retrieval__transmission": f'{ROLE_LABEL["fact_retrieval"]} + {ROLE_LABEL["transmission"]}',
    "pair_splitting__transmission": f'{ROLE_LABEL["splitting"]} + {ROLE_LABEL["transmission"]}',
    "mixed_all": "Mixed-All",
    "random_outside_selected": "Random-Outside",
}
STRATEGY_COLOR = {
    "single_fact_retrieval": ROLE_COLOR["fact_retrieval"],
    "single_splitting": ROLE_COLOR["splitting"],
    "single_transmission": ROLE_COLOR["transmission"],
    "pair_fact_retrieval__splitting": "#7A4C99",
    "pair_fact_retrieval__transmission": "#B8860B",
    "pair_splitting__transmission": "#3B8EA5",
    "mixed_all": "#2F2F2F",
    "random_outside_selected": "#9A9A9A",
}
STRATEGY_MARKER = {
    "single_fact_retrieval": "o",
    "single_splitting": "s",
    "single_transmission": "^",
    "pair_fact_retrieval__splitting": "D",
    "pair_fact_retrieval__transmission": "P",
    "pair_splitting__transmission": "X",
    "mixed_all": "*",
    "random_outside_selected": "v",
}


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _coerce_role_score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def resolve_role_label(row: Mapping[str, object], role_col: str = "role_label") -> str | None:
    role_raw = str(row.get(role_col, "")).strip()
    if role_raw in ROLE_ORDER:
        return role_raw

    ranked = []
    for idx, role in enumerate(ROLE_ORDER):
        score = _coerce_role_score(row.get(ROLE_SCORE_COLUMNS[role]))
        if math.isfinite(score):
            ranked.append((score, -idx, role))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return str(ranked[0][2])


def plot_layer_head_distribution(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    n_layers: int,
    n_heads: int,
    role_col: str = "role_label",
) -> None:
    pts_by_role: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"x": [], "y": [], "s": []})
    layer_role_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        layer = int(float(row.get("layer", 0)))
        head = int(float(row.get("head", 0)))
        role = resolve_role_label(row, role_col=role_col)
        if role is None:
            continue
        impact = abs(float(row.get("mean_abs_dpd_shift", 0.0)))
        pts_by_role[role]["x"].append(float(head))
        pts_by_role[role]["y"].append(float(layer))
        pts_by_role[role]["s"].append(35.0 + 300.0 * impact)
        layer_role_counts[layer][role] += 1

    apply_paper_style(
        {
            "axes.grid": True,
            "grid.alpha": 0.20,
            "font.size": 14.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 14.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.0,
        }
    )
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(13.4, 6.8), gridspec_kw={"width_ratios": [1.45, 1.0]})

    gx, gy = np.meshgrid(np.arange(n_heads), np.arange(n_layers))
    ax_map.scatter(gx.ravel(), gy.ravel(), s=7, c="#DCDCDC", marker="s", alpha=0.35, linewidths=0)

    for role in ROLE_ORDER:
        xs, ys, ss = pts_by_role[role]["x"], pts_by_role[role]["y"], pts_by_role[role]["s"]
        if not xs:
            continue
        ax_map.scatter(
            xs,
            ys,
            s=ss,
            c=ROLE_COLOR[role],
            edgecolors="#1F1F1F",
            linewidths=0.45,
            alpha=0.92,
            marker="s",
            label=f"{ROLE_LABEL[role]} (n={len(xs)})",
            zorder=3,
        )

    ax_map.set_xlim(-0.6, n_heads - 0.4)
    ax_map.set_ylim(-0.6, n_layers - 0.4)
    ax_map.set_xlabel("Head Index")
    ax_map.set_ylabel("Layer")
    ax_map.set_xticks(np.arange(0, n_heads, max(1, n_heads // 8)))
    ax_map.set_yticks(np.arange(0, n_layers, max(1, n_layers // 10)))
    ax_map.legend(loc="upper left", frameon=True)
    stylize_axis(ax_map)

    layers_arr = np.arange(n_layers)
    bottom = np.zeros(n_layers, dtype=np.float64)
    for role in ROLE_ORDER:
        values = np.asarray([layer_role_counts[int(layer)].get(role, 0) for layer in layers_arr], dtype=np.float64)
        if values.sum() <= 0:
            continue
        ax_bar.barh(
            layers_arr,
            values,
            left=bottom,
            color=ROLE_COLOR[role],
            edgecolor="#2F2F2F",
            linewidth=0.35,
            alpha=0.88,
            label=ROLE_LABEL[role],
        )
        bottom += values

    ax_bar.set_ylim(-0.6, n_layers - 0.4)
    ax_bar.set_xlabel("# Heads in Layer")
    ax_bar.set_ylabel("Layer")
    ax_bar.set_yticks(np.arange(0, n_layers, max(1, n_layers // 10)))
    stylize_axis(ax_bar)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320)
    plt.close(fig)


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

    apply_paper_style(
        {
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 14.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 14.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.0,
        }
    )
    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    ax.plot(layers, fact, label=ROLE_LABEL["fact_retrieval"], marker="o", color=ROLE_COLOR["fact_retrieval"], linewidth=2.4, markersize=6.6)
    ax.plot(layers, split, label=ROLE_LABEL["splitting"], marker="s", color=ROLE_COLOR["splitting"], linewidth=2.4, markersize=6.4)
    ax.plot(layers, trans, label=ROLE_LABEL["transmission"], marker="^", color=ROLE_COLOR["transmission"], linewidth=2.4, markersize=6.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head count")
    if layers:
        ax.set_xticks(np.arange(0, max(layers) + 1, max(1, n_layers // 10)))
    ax.legend(loc="best", frameon=True)
    stylize_axis(ax)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320)
    plt.close(fig)


def plot_pd_curves(
    aggregated_rows: Sequence[Mapping[str, object]],
    output: Path,
    *,
    y_mean_key: str,
    y_sem_key: str,
    ylabel: str,
    k_values: Sequence[int],
) -> None:
    if not aggregated_rows:
        return

    apply_paper_style(
        {
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 14.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 14.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.0,
        }
    )
    fig, ax = plt.subplots(figsize=(11.6, 6.8))

    any_data = False
    for strategy_id in STRATEGY_ORDER:
        strat_rows = [row for row in aggregated_rows if str(row["strategy_id"]) == strategy_id]
        if not strat_rows:
            continue
        strat_rows = sorted(strat_rows, key=lambda row: int(row["k"]))
        xs = [int(row["k"]) for row in strat_rows]
        ys = [float(row[y_mean_key]) for row in strat_rows]
        sems = [float(row[y_sem_key]) for row in strat_rows]
        color = STRATEGY_COLOR.get(strategy_id, "#4D4D4D")
        marker = STRATEGY_MARKER.get(strategy_id, "o")
        linestyle = "--" if strategy_id == "random_outside_selected" else "-"
        ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle, linewidth=2.35, label=STRATEGY_LABEL.get(strategy_id, strategy_id))
        if any(sems):
            xs_arr = np.asarray(xs, dtype=np.float64)
            ys_arr = np.asarray(ys, dtype=np.float64)
            sem_arr = np.asarray(sems, dtype=np.float64)
            ax.fill_between(xs_arr, ys_arr - sem_arr, ys_arr + sem_arr, color=color, alpha=0.12)
        any_data = True

    if not any_data:
        plt.close(fig)
        return

    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("k (number of heads)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(k_values))
    ax.legend(loc="best", ncol=2, frameon=True)
    stylize_axis(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320)
    plt.close(fig)
