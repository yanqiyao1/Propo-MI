from __future__ import annotations

import itertools
import random
from typing import Callable, Dict, Iterable, List, Tuple


def _resolve_candidate_vars(facts: Dict[str, bool], candidate_vars: Iterable[str]) -> List[str]:
    vars_list = sorted({str(v) for v in candidate_vars if v in facts})
    if not vars_list:
        vars_list = sorted(str(v) for v in facts)
    if not vars_list:
        raise ValueError("Cannot corrupt empty fact mapping")
    return vars_list


def find_label_flipping_fact_corruption(
    facts: Dict[str, bool],
    candidate_vars: Iterable[str],
    evaluate_label: Callable[[Dict[str, bool]], bool],
    clean_label: bool,
    rng: random.Random,
) -> Tuple[Dict[str, bool], Dict[str, object]]:
    vars_list = _resolve_candidate_vars(facts, candidate_vars)
    candidates: List[Tuple[int, List[str], Dict[str, bool]]] = []

    for values in itertools.product((False, True), repeat=len(vars_list)):
        corrupted = dict(facts)
        changed_vars: List[str] = []
        for var, value in zip(vars_list, values):
            corrupted[var] = value
            if facts[var] != value:
                changed_vars.append(var)
        if not changed_vars:
            continue
        if evaluate_label(corrupted) == clean_label:
            continue
        candidates.append((len(changed_vars), changed_vars, corrupted))

    if not candidates:
        raise ValueError("Unable to find a fact corruption that flips the label")

    min_distance = min(distance for distance, _, _ in candidates)
    minimal_candidates = [item for item in candidates if item[0] == min_distance]
    _, changed_vars, corrupted = rng.choice(minimal_candidates)
    meta = {
        "corruption": "label_flipping_fact_search",
        "changed_vars": list(changed_vars),
        "hamming_distance": len(changed_vars),
        "candidate_var_count": len(vars_list),
    }
    return corrupted, meta
