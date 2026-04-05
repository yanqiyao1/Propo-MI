from __future__ import annotations

import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def balanced_sample_by_rule(
    rows: Sequence[Mapping[str, object]],
    max_samples: int,
    rule_key: str = "rule",
) -> List[Dict[str, object]]:
    rows_list = [dict(row) for row in rows]
    if max_samples <= 0 or len(rows_list) <= max_samples:
        return rows_list

    groups: "OrderedDict[str, List[Dict[str, object]]]" = OrderedDict()
    for row in rows_list:
        rule_name = str(row.get(rule_key, ""))
        if rule_name not in groups:
            groups[rule_name] = []
        groups[rule_name].append(row)

    ordered_rules = sorted(groups.keys(), key=lambda name: (len(groups[name]), name))
    cursor = {name: 0 for name in ordered_rules}
    active_rules = [name for name in ordered_rules if groups[name]]
    selected: List[Dict[str, object]] = []

    while len(selected) < max_samples and active_rules:
        next_active: List[str] = []
        for rule_name in active_rules:
            idx = cursor[rule_name]
            if idx < len(groups[rule_name]) and len(selected) < max_samples:
                selected.append(groups[rule_name][idx])
                cursor[rule_name] = idx + 1
            if cursor[rule_name] < len(groups[rule_name]):
                next_active.append(rule_name)
            if len(selected) >= max_samples:
                break
        active_rules = next_active

    return selected


def count_by_field(items: Sequence[Any], field: str = "rule") -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        if isinstance(item, Mapping):
            value = item.get(field, "")
        else:
            value = getattr(item, field, "")
        counter[str(value)] += 1
    return dict(sorted(counter.items()))
