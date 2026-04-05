from __future__ import annotations

import csv
from pathlib import Path


SCORE_LABELS = {
    "dpd": "dPD",
    "abs_dpd": "|dPD|",
}


def resolve_score_label(score_column: str) -> str:
    return SCORE_LABELS.get(score_column, score_column)


def read_scores_csv(csv_path: Path, score_column: str = "dpd") -> tuple[list[int], list[float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Score csv is empty: {csv_path}")
    if score_column not in rows[0]:
        if score_column == "abs_dpd" and "dpd" in rows[0]:
            layers = [int(float(row["layer"])) for row in rows]
            scores = [abs(float(row["dpd"])) for row in rows]
            return layers, scores
        raise KeyError(f"Score column {score_column!r} not found in {csv_path}")
    layers = [int(float(row["layer"])) for row in rows]
    scores = [float(row[score_column]) for row in rows]
    return layers, scores
