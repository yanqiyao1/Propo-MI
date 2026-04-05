from __future__ import annotations

import argparse
import math
import json
import random
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.progress import log_event, resolve_log_path, setup_file_logger

from .corruptions import find_label_flipping_fact_corruption
from .formatters import PROMPT_ENDING_CHOICES, build_depth_prompt, build_prompt
from .logic_ast import And, Const, Expr, Not, Or, Var, collect_vars, eval_expr, rename_vars, to_semi_natural, to_symbolic
from .template_rules import all_rule_names, build_one_hop_expr, get_rule_template


VAR_POOL = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in {"F", "T"}]


def _sample_facts(variables: Iterable[str], rng: random.Random) -> Dict[str, bool]:
    return {name: bool(rng.getrandbits(1)) for name in sorted(set(variables))}


def _iter_bool_envs(variables: Sequence[str]) -> Iterable[Dict[str, bool]]:
    vars_list = list(sorted(set(variables)))
    for values in product((False, True), repeat=len(vars_list)):
        yield {var: value for var, value in zip(vars_list, values)}


def _constant_expr_value(expr: Expr) -> bool | None:
    expr_vars = collect_vars(expr)
    envs = list(_iter_bool_envs(expr_vars)) if expr_vars else [{}]
    values = {eval_expr(expr, env) for env in envs}
    if len(values) != 1:
        return None
    return next(iter(values))


def _build_opposite_constant_expr(expr: Expr, desired_label: bool) -> Expr:
    expr_vars = collect_vars(expr)
    if not expr_vars:
        return Const(desired_label)
    anchor = Var(expr_vars[0])
    if desired_label:
        return Or(anchor, Not(anchor))
    return And(anchor, Not(anchor))


def _choose_fresh_vars(rng: random.Random, used: Iterable[str], count: int) -> List[str]:
    used_set = set(used)
    candidates = [v for v in VAR_POOL if v not in used_set]
    if len(candidates) < count:
        raise ValueError(f"Not enough variable symbols left. need={count}, available={len(candidates)}")
    return rng.sample(candidates, k=count)


def _remap_expr_variables(expr: Expr, rng: random.Random) -> Tuple[Expr, Dict[str, str]]:
    src_vars = collect_vars(expr)
    dst_vars = _choose_fresh_vars(rng, used=[], count=len(src_vars))
    mapping = {src: dst for src, dst in zip(src_vars, dst_vars)}
    return rename_vars(expr, mapping), mapping


def _record_one_hop(
    sample_id: str,
    rule_name: str,
    expr: Expr,
    rng: random.Random,
    prompt_order: str = "facts_first",
    prompt_ending: str = "answer_suffix",
) -> Dict[str, object]:
    template = get_rule_template(rule_name)
    expr_vars = collect_vars(expr)
    facts = _sample_facts(expr_vars, rng)
    label = eval_expr(expr, facts)
    corrupted_expr = expr
    try:
        corrupted_facts, corruption_meta = find_label_flipping_fact_corruption(
            facts,
            expr_vars,
            evaluate_label=lambda env: eval_expr(expr, env),
            clean_label=label,
            rng=rng,
        )
    except ValueError:
        const_value = _constant_expr_value(expr)
        if const_value is None:
            raise
        corrupted_facts = dict(facts)
        corrupted_expr = _build_opposite_constant_expr(expr, desired_label=not label)
        corruption_meta = {
            "corruption": "label_flipping_expr_fallback",
            "changed_vars": [],
            "hamming_distance": 0,
            "expr_changed": True,
            "clean_expr_constant": const_value,
        }
    corrupted_label = eval_expr(corrupted_expr, corrupted_facts)
    if corrupted_label == label:
        raise RuntimeError(f"Failed to flip label for one-hop sample {sample_id}")

    expr_symbolic = to_symbolic(expr)
    expr_semi = to_semi_natural(expr)
    corrupted_expr_symbolic = to_symbolic(corrupted_expr)
    corrupted_expr_semi = to_semi_natural(corrupted_expr)

    return {
        "id": sample_id,
        "hop": "one_hop",
        "prompt_order": prompt_order,
        "prompt_ending": prompt_ending,
        "rule": rule_name,
        "category": template.category,
        "formal_definition": template.formal_definition,
        "facts": facts,
        "expr_symbolic": expr_symbolic,
        "expr_semi_natural": expr_semi,
        "corrupted_expr_symbolic": corrupted_expr_symbolic,
        "corrupted_expr_semi_natural": corrupted_expr_semi,
        "query": "Evaluate proposition truth value",
        "label": label,
        "label_corrupted": corrupted_label,
        "corrupted_facts": corrupted_facts,
        "clean_prompt_symbolic": build_depth_prompt(
            hop="one_hop",
            facts=facts,
            query_expr_text=expr_symbolic,
            mode="nocot",
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        ),
        "corrupted_prompt_symbolic": build_depth_prompt(
            hop="one_hop",
            facts=corrupted_facts,
            query_expr_text=corrupted_expr_symbolic,
            mode="nocot",
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        ),
        "clean_prompt_symbolic_verbose": build_prompt(
            facts,
            expr_symbolic,
            mode="nocot",
            template_style="verbose",
            truth_style="full",
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        ),
        "corrupted_prompt_symbolic_verbose": build_prompt(
            corrupted_facts,
            corrupted_expr_symbolic,
            mode="nocot",
            template_style="verbose",
            truth_style="full",
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        ),
        "meta": {
            "corruption": corruption_meta,
        },
    }


def _build_two_hop_chain(rule_name: str, rng: random.Random) -> Dict[str, object]:
    base_expr = build_one_hop_expr(rule_name, rng)
    step1_expr, remap = _remap_expr_variables(base_expr, rng)
    step1_vars = collect_vars(step1_expr)

    mid_var, gate_var = _choose_fresh_vars(rng, used=step1_vars, count=2)
    step1_const = _constant_expr_value(step1_expr)
    if step1_const is True:
        use_and = True
    elif step1_const is False:
        use_and = False
    else:
        use_and = rng.random() < 0.5
    query_expr = And(Var(mid_var), Var(gate_var)) if use_and else Or(Var(mid_var), Var(gate_var))

    return {
        "step1_expr": step1_expr,
        "step1_vars": step1_vars,
        "mid_var": mid_var,
        "gate_var": gate_var,
        "query_expr": query_expr,
        "query_op": "and" if use_and else "or",
        "var_remap": remap,
        "step1_constant_value": step1_const,
    }


def _record_two_hop(
    sample_id: str,
    rule_name: str,
    rng: random.Random,
    prompt_order: str = "facts_first",
    prompt_ending: str = "answer_suffix",
) -> Dict[str, object]:
    template = get_rule_template(rule_name)
    chain = _build_two_hop_chain(rule_name, rng)

    step1_expr = chain["step1_expr"]
    step1_vars = chain["step1_vars"]
    mid_var = chain["mid_var"]
    gate_var = chain["gate_var"]
    query_expr = chain["query_expr"]
    query_op = chain["query_op"]
    var_remap = chain["var_remap"]

    facts = _sample_facts(list(step1_vars) + [gate_var], rng)
    step1_value = eval_expr(step1_expr, facts)
    env_clean = dict(facts)
    env_clean[mid_var] = step1_value
    label = eval_expr(query_expr, env_clean)

    def evaluate_two_hop(env_facts: Dict[str, bool]) -> bool:
        step1_value_eval = eval_expr(step1_expr, env_facts)
        env_full = dict(env_facts)
        env_full[mid_var] = step1_value_eval
        return eval_expr(query_expr, env_full)

    corrupted_facts, corruption_meta = find_label_flipping_fact_corruption(
        facts,
        list(step1_vars) + [gate_var],
        evaluate_label=evaluate_two_hop,
        clean_label=label,
        rng=rng,
    )
    step1_value_corrupt = eval_expr(step1_expr, corrupted_facts)
    env_corrupt = dict(corrupted_facts)
    env_corrupt[mid_var] = step1_value_corrupt
    corrupted_label = eval_expr(query_expr, env_corrupt)
    if corrupted_label == label:
        raise RuntimeError(f"Failed to flip label for two-hop sample {sample_id}")

    step1_symbolic = to_symbolic(step1_expr)
    query_symbolic = to_symbolic(query_expr)
    step1_semi = to_semi_natural(step1_expr)
    query_semi = to_semi_natural(query_expr)
    derived_symbolic = [(mid_var, step1_symbolic)]

    return {
        "id": sample_id,
        "hop": "two_hop",
        "prompt_order": prompt_order,
        "prompt_ending": prompt_ending,
        "rule": rule_name,
        "category": template.category,
        "formal_definition": template.formal_definition,
        "facts": facts,
        "expr_symbolic": query_symbolic,
        "expr_semi_natural": query_semi,
        "corrupted_expr_symbolic": query_symbolic,
        "corrupted_expr_semi_natural": query_semi,
        "intermediate_var": mid_var,
        "intermediate_expr_symbolic": step1_symbolic,
        "intermediate_expr_semi_natural": step1_semi,
        "query": "Evaluate proposition truth value",
        "label": label,
        "label_corrupted": corrupted_label,
        "corrupted_facts": corrupted_facts,
        "clean_prompt_symbolic": build_depth_prompt(
            hop="two_hop",
            facts=facts,
            query_expr_text=query_symbolic,
            mode="nocot",
            derived_steps=derived_symbolic,
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        ),
        "corrupted_prompt_symbolic": build_depth_prompt(
            hop="two_hop",
            facts=corrupted_facts,
            query_expr_text=query_symbolic,
            mode="nocot",
            derived_steps=derived_symbolic,
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        ),
        "clean_prompt_symbolic_verbose": build_prompt(
            facts=facts,
            expr_text=query_symbolic,
            mode="nocot",
            template_style="verbose",
            truth_style="full",
            derived_steps=derived_symbolic,
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        ),
        "corrupted_prompt_symbolic_verbose": build_prompt(
            facts=corrupted_facts,
            expr_text=query_symbolic,
            mode="nocot",
            template_style="verbose",
            truth_style="full",
            derived_steps=derived_symbolic,
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        ),
        "meta": {
            "corruption": corruption_meta,
            "two_hop_chain": {
                "query_op": query_op,
                "gate_var": gate_var,
                "var_remap": var_remap,
                "step1_constant_value": chain["step1_constant_value"],
            },
        },
    }


def _row_unique_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    # Uniqueness is defined by symbolic prompt content and hop/rule identity.
    return (
        str(row["hop"]),
        str(row["rule"]),
        str(row["clean_prompt_symbolic"]),
        str(row["corrupted_prompt_symbolic"]),
    )


def _expr_variant_count(rule_name: str) -> int:
    # This template family only has one or two symbolic variants per rule.
    # 16 probes are enough to recover the full variant set deterministically.
    variants = set()
    for seed in range(16):
        expr = build_one_hop_expr(rule_name, random.Random(seed))
        variants.add(to_symbolic(expr))
    return len(variants)


def _rule_var_count(rule_name: str) -> int:
    expr = build_one_hop_expr(rule_name, random.Random(0))
    return len(collect_vars(expr))


def _perm(n: int, k: int) -> int:
    # Python 3.10 has math.perm, keep helper for clarity.
    return int(math.perm(n, k))


def _rule_capacity(rule_name: str, hop: str) -> int:
    # Conservative upper bound under label-flipping corruption semantics:
    # one-hop: expr_variant * variable_remap * clean_facts * alternative_fact_assignments_upper_bound
    # two-hop: one-hop factors on step1 vars * choose(mid,gate) * gate fact *
    #          alternative_fact_assignments_upper_bound * query op
    n_vars = len(VAR_POOL)
    k = _rule_var_count(rule_name)
    s = _expr_variant_count(rule_name)
    base = s * _perm(n_vars, k)

    if hop == "one_hop":
        return base * (2**k) * max(1, (2**k) - 1)
    if hop == "two_hop":
        return base * _perm(n_vars - k, 2) * (2 ** (k + 1)) * max(1, (2 ** (k + 1)) - 1) * 2
    raise ValueError(f"Unknown hop {hop!r}")


def _hop_total_capacity(hop: str) -> int:
    return sum(_rule_capacity(rule, hop) for rule in all_rule_names())


def _ordered_rule_counts(rules: Sequence[str], counts: Dict[str, int]) -> Dict[str, int]:
    return {rule: int(counts.get(rule, 0)) for rule in rules}


def _allocate_even_rule_totals(
    *,
    total_count: int,
    rules: Sequence[str],
    total_capacities: Dict[str, int],
    rng: random.Random,
) -> Dict[str, int]:
    rules_list = list(rules)
    if total_count <= 0:
        return _ordered_rule_counts(rules_list, {})
    if total_count < len(rules_list):
        raise ValueError(
            f"target_count={total_count} cannot cover all {len(rules_list)} rules. "
            f"Use target_count >= {len(rules_list)}."
        )

    counts = {rule: 0 for rule in rules_list}
    shuffled_rules = list(rules_list)
    rng.shuffle(shuffled_rules)

    for rule in shuffled_rules:
        if total_capacities[rule] < 1:
            raise ValueError(f"Rule {rule!r} cannot supply even a single unique row under current capacities")
        counts[rule] = 1

    remaining = total_count - len(rules_list)
    while remaining > 0:
        progressed = False
        for rule in shuffled_rules:
            if remaining == 0:
                break
            if counts[rule] >= total_capacities[rule]:
                continue
            counts[rule] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise ValueError(f"Requested target_count {total_count} exceeds total per-rule capacity constraints")

    return _ordered_rule_counts(rules_list, counts)


def _allocate_hop_targets_per_rule(
    *,
    rule_totals: Dict[str, int],
    one_hop_target: int,
    one_hop_capacities: Dict[str, int],
    two_hop_capacities: Dict[str, int],
    rng: random.Random,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int | float | bool]]:
    rules = list(rule_totals)
    total_count = sum(rule_totals.values())
    if total_count == 0:
        empty = _ordered_rule_counts(rules, {})
        return empty, empty, {
            "requested_one_hop_target": one_hop_target,
            "resolved_one_hop_target": 0,
            "requested_one_hop_ratio": 0.0,
            "resolved_one_hop_ratio": 0.0,
            "min_feasible_one_hop": 0,
            "max_feasible_one_hop": 0,
            "one_hop_target_adjusted": bool(one_hop_target),
        }
    if not (0 <= one_hop_target <= total_count):
        raise ValueError(f"one_hop_target must be in [0, {total_count}], got {one_hop_target}")

    lower_bounds: Dict[str, int] = {}
    upper_bounds: Dict[str, int] = {}
    for rule in rules:
        total_for_rule = rule_totals[rule]
        lower = max(0, total_for_rule - two_hop_capacities[rule])
        upper = min(total_for_rule, one_hop_capacities[rule])
        if lower > upper:
            raise ValueError(
                f"Unable to allocate hop counts for rule {rule!r}: lower_bound={lower}, upper_bound={upper}"
            )
        lower_bounds[rule] = lower
        upper_bounds[rule] = upper

    min_one_hop = sum(lower_bounds.values())
    max_one_hop = sum(upper_bounds.values())
    resolved_one_hop_target = min(max(one_hop_target, min_one_hop), max_one_hop)

    one_hop_counts = dict(lower_bounds)
    remaining = resolved_one_hop_target - min_one_hop
    exact_ratio = resolved_one_hop_target / total_count
    tie_break_order = list(rules)
    rng.shuffle(tie_break_order)
    tie_break_priority = {rule: len(tie_break_order) - idx for idx, rule in enumerate(tie_break_order)}

    while remaining > 0:
        candidates = [rule for rule in rules if one_hop_counts[rule] < upper_bounds[rule]]
        if not candidates:
            raise RuntimeError("Failed to allocate one-hop targets despite feasibility check")
        chosen_rule = max(
            candidates,
            key=lambda rule: (
                (rule_totals[rule] * exact_ratio) - one_hop_counts[rule],
                upper_bounds[rule] - one_hop_counts[rule],
                tie_break_priority[rule],
            ),
        )
        one_hop_counts[chosen_rule] += 1
        remaining -= 1

    two_hop_counts = {rule: rule_totals[rule] - one_hop_counts[rule] for rule in rules}
    allocation_meta: Dict[str, int | float | bool] = {
        "requested_one_hop_target": one_hop_target,
        "resolved_one_hop_target": resolved_one_hop_target,
        "requested_one_hop_ratio": one_hop_target / total_count,
        "resolved_one_hop_ratio": resolved_one_hop_target / total_count,
        "min_feasible_one_hop": min_one_hop,
        "max_feasible_one_hop": max_one_hop,
        "one_hop_target_adjusted": resolved_one_hop_target != one_hop_target,
    }
    return _ordered_rule_counts(rules, one_hop_counts), _ordered_rule_counts(rules, two_hop_counts), allocation_meta


def _generate_unique_row_for_rule(
    *,
    hop: str,
    rule: str,
    sample_index: int,
    rng: random.Random,
    prompt_order: str,
    prompt_ending: str,
) -> Dict[str, object]:
    if hop == "one_hop":
        one_hop_base = build_one_hop_expr(rule, rng)
        one_hop_expr, one_hop_remap = _remap_expr_variables(one_hop_base, rng)
        sample_id = f"{rule}_one_u{sample_index}"
        row = _record_one_hop(
            sample_id,
            rule,
            one_hop_expr,
            rng,
            prompt_order=prompt_order,
            prompt_ending=prompt_ending,
        )
        row["meta"]["var_remap"] = one_hop_remap
        return row
    if hop == "two_hop":
        sample_id = f"{rule}_two_u{sample_index}"
        return _record_two_hop(sample_id, rule, rng, prompt_order=prompt_order, prompt_ending=prompt_ending)
    raise ValueError(f"Unknown hop {hop!r}")


def _generate_unique_hop_rows(
    *,
    hop: str,
    per_rule_targets: Dict[str, int],
    rng: random.Random,
    max_attempts_multiplier: int,
    prompt_order: str,
    prompt_ending: str,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    rules = all_rule_names()
    normalized_targets = _ordered_rule_counts(rules, per_rule_targets)
    target_count = sum(normalized_targets.values())
    if target_count <= 0:
        return [], normalized_targets

    capacities = {rule: _rule_capacity(rule, hop) for rule in rules}
    for rule, target in normalized_targets.items():
        if target < 0:
            raise ValueError(f"Per-rule target for {rule!r} must be >= 0, got {target}")
        if target > capacities[rule]:
            raise ValueError(
                f"Requested {target} unique rows for {hop}/{rule}, but theoretical max is {capacities[rule]}"
            )

    accepted: List[Dict[str, object]] = []
    seen = set()
    accepted_by_rule: Counter[str] = Counter()
    id_counters = defaultdict(int)
    rule_order = [rule for rule in rules if normalized_targets[rule] > 0]
    rng.shuffle(rule_order)

    for rule in rule_order:
        target_for_rule = normalized_targets[rule]
        attempts = 0
        max_attempts = max(target_for_rule * max_attempts_multiplier, target_for_rule + 1)
        while accepted_by_rule[rule] < target_for_rule and attempts < max_attempts:
            attempts += 1
            row = _generate_unique_row_for_rule(
                hop=hop,
                rule=rule,
                sample_index=id_counters[rule],
                rng=rng,
                prompt_order=prompt_order,
                prompt_ending=prompt_ending,
            )
            key = _row_unique_key(row)
            if key in seen:
                continue
            seen.add(key)
            id_counters[rule] += 1
            accepted_by_rule[rule] += 1
            accepted.append(row)

        if accepted_by_rule[rule] < target_for_rule:
            raise RuntimeError(
                f"Failed to generate enough unique rows for {hop}/{rule}: "
                f"requested={target_for_rule}, accepted={accepted_by_rule[rule]}, attempts={attempts}. "
                "Try reducing target_count or increasing --max_attempts_multiplier."
            )

    return accepted, _ordered_rule_counts(rules, accepted_by_rule)


def generate_records(
    per_rule_per_hop: int,
    seed: int,
    prompt_order: str = "facts_first",
    prompt_ending: str = "answer_suffix",
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for rule_name in all_rule_names():
        for idx in range(per_rule_per_hop):
            one_hop_base = build_one_hop_expr(rule_name, rng)
            one_hop, one_hop_remap = _remap_expr_variables(one_hop_base, rng)
            row = _record_one_hop(
                f"{rule_name}_one_{idx}",
                rule_name,
                one_hop,
                rng,
                prompt_order=prompt_order,
                prompt_ending=prompt_ending,
            )
            row["meta"]["var_remap"] = one_hop_remap
            rows.append(row)
            rows.append(_record_two_hop(f"{rule_name}_two_{idx}", rule_name, rng, prompt_order=prompt_order, prompt_ending=prompt_ending))

    rng.shuffle(rows)
    return rows


def generate_unique_records(
    *,
    target_count: int,
    seed: int,
    one_hop_ratio: float,
    max_attempts_multiplier: int,
    prompt_order: str = "facts_first",
    prompt_ending: str = "answer_suffix",
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if target_count <= 0:
        raise ValueError("target_count must be > 0 for unique generation")
    if not (0.0 <= one_hop_ratio <= 1.0):
        raise ValueError("one_hop_ratio must be in [0, 1]")

    rules = all_rule_names()
    one_target = int(round(target_count * one_hop_ratio))
    two_target = target_count - one_target

    one_hop_capacities = {rule: _rule_capacity(rule, "one_hop") for rule in rules}
    two_hop_capacities = {rule: _rule_capacity(rule, "two_hop") for rule in rules}
    total_capacities = {rule: one_hop_capacities[rule] + two_hop_capacities[rule] for rule in rules}

    one_cap = sum(one_hop_capacities.values())
    two_cap = sum(two_hop_capacities.values())
    total_cap = sum(total_capacities.values())
    if one_target > one_cap:
        raise ValueError(f"Requested one_hop target {one_target} exceeds max capacity {one_cap}")
    if two_target > two_cap:
        raise ValueError(f"Requested two_hop target {two_target} exceeds max capacity {two_cap}")
    if target_count > total_cap:
        raise ValueError(f"Requested target_count {target_count} exceeds max combined capacity {total_cap}")

    rng = random.Random(seed)
    rule_totals = _allocate_even_rule_totals(
        total_count=target_count,
        rules=rules,
        total_capacities=total_capacities,
        rng=rng,
    )
    one_targets_by_rule, two_targets_by_rule, hop_allocation_meta = _allocate_hop_targets_per_rule(
        rule_totals=rule_totals,
        one_hop_target=one_target,
        one_hop_capacities=one_hop_capacities,
        two_hop_capacities=two_hop_capacities,
        rng=rng,
    )

    one_rows, one_by_rule = _generate_unique_hop_rows(
        hop="one_hop",
        per_rule_targets=one_targets_by_rule,
        rng=rng,
        max_attempts_multiplier=max_attempts_multiplier,
        prompt_order=prompt_order,
        prompt_ending=prompt_ending,
    )
    two_rows, two_by_rule = _generate_unique_hop_rows(
        hop="two_hop",
        per_rule_targets=two_targets_by_rule,
        rng=rng,
        max_attempts_multiplier=max_attempts_multiplier,
        prompt_order=prompt_order,
        prompt_ending=prompt_ending,
    )

    rows = one_rows + two_rows
    rng.shuffle(rows)
    overall_rule_counts = {rule: one_by_rule[rule] + two_by_rule[rule] for rule in rules}
    actual_one_target = sum(one_by_rule.values())
    actual_two_target = sum(two_by_rule.values())
    meta = {
        "target_count": target_count,
        "requested_one_hop_target": one_target,
        "requested_two_hop_target": two_target,
        "one_hop_target": actual_one_target,
        "two_hop_target": actual_two_target,
        "one_hop_capacity_max": one_cap,
        "two_hop_capacity_max": two_cap,
        "combined_capacity_max": total_cap,
        "rule_counts": overall_rule_counts,
        "rule_balance_span": max(overall_rule_counts.values()) - min(overall_rule_counts.values()),
        "one_hop_rule_counts": one_by_rule,
        "two_hop_rule_counts": two_by_rule,
        "hop_allocation": hop_allocation_meta,
    }
    return rows, meta


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PropLogic-MI style dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per_rule_per_hop", type=int, default=200)
    parser.add_argument(
        "--target_count",
        type=int,
        default=0,
        help="If >0, generate exactly this many unique rows with full rule coverage and near-even rule balancing.",
    )
    parser.add_argument(
        "--one_hop_ratio",
        type=float,
        default=0.5,
        help="Used with --target_count. Fraction of rows allocated to one_hop.",
    )
    parser.add_argument(
        "--max_attempts_multiplier",
        type=int,
        default=200,
        help="Used with --target_count. Max attempts per desired unique row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt_order",
        choices=["facts_first", "expr_first"],
        default="facts_first",
        help="Prompt layout for compact prompts. facts_first keeps previous default.",
    )
    parser.add_argument(
        "--prompt_ending",
        choices=list(PROMPT_ENDING_CHOICES),
        default="answer_suffix",
        help="Prompt ending style. terminal_is removes the answer instruction and leaves the prompt ending at the final 'is'.",
    )
    args = parser.parse_args()
    logger = setup_file_logger(__name__, resolve_log_path(output_path=args.output))

    if args.prompt_ending == "terminal_is" and args.prompt_order != "facts_first":
        raise ValueError("--prompt_ending terminal_is currently requires --prompt_order facts_first")

    if args.target_count > 0:
        rows, uniq_meta = generate_unique_records(
            target_count=args.target_count,
            seed=args.seed,
            one_hop_ratio=args.one_hop_ratio,
            max_attempts_multiplier=args.max_attempts_multiplier,
            prompt_order=args.prompt_order,
            prompt_ending=args.prompt_ending,
        )
    else:
        rows = generate_records(
            per_rule_per_hop=args.per_rule_per_hop,
            seed=args.seed,
            prompt_order=args.prompt_order,
            prompt_ending=args.prompt_ending,
        )
        uniq_meta = None

    _write_jsonl(args.output, rows)

    unique_keys = {_row_unique_key(r) for r in rows}
    log_event(
        logger,
        {
            "output": str(args.output),
            "count": len(rows),
            "per_rule_per_hop": args.per_rule_per_hop,
            "target_count": args.target_count,
            "seed": args.seed,
            "prompt_order": args.prompt_order,
            "prompt_ending": args.prompt_ending,
            "unique_rows": len(unique_keys),
            "duplicate_rows": len(rows) - len(unique_keys),
            "unique_generation": uniq_meta,
        },
    )


if __name__ == "__main__":
    main()
