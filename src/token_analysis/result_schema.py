from __future__ import annotations

from typing import Mapping, Sequence


DELTA_SCORE_KEYS = ("delta_dpd", "delta_ld")


def get_delta_score_matrix(row: Mapping[str, object]) -> object | None:
    for key in DELTA_SCORE_KEYS:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _matrix_shape(matrix: object) -> tuple[int, int] | None:
    shape = getattr(matrix, "shape", None)
    if isinstance(shape, Sequence) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])

    if isinstance(matrix, (list, tuple)):
        n_rows = len(matrix)
        if n_rows == 0:
            return 0, 0
        first_row = matrix[0]
        if isinstance(first_row, (list, tuple)):
            return n_rows, len(first_row)

    return None


def infer_delta_score_n_layers(results: Sequence[Mapping[str, object]], fallback: int = 0) -> int:
    n_layers = 0
    for row in results:
        matrix = get_delta_score_matrix(row)
        if matrix is None:
            continue
        shape = _matrix_shape(matrix)
        if shape is None:
            continue
        n_layers = max(n_layers, int(shape[0]))
    if n_layers > 0:
        return n_layers
    return int(fallback)
