from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int


def _normalize(items: Sequence[dict]) -> List[HeadRef]:
    out: List[HeadRef] = []
    for item in items:
        out.append(HeadRef(layer=int(item["layer"]), head=int(item["head"])))
    return out


DEFAULT_TAXONOMY: Dict[str, List[HeadRef]] = {
    "splitting": [],
    "transmission": [],
    "fact_retrieval": [],
}


def load_taxonomy(path: Path | None) -> Dict[str, List[HeadRef]]:
    if path is None:
        return DEFAULT_TAXONOMY
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {k: _normalize(v) for k, v in obj.items()}


def get_head_set(name: str, taxonomy: Dict[str, List[HeadRef]]) -> List[HeadRef]:
    if name not in taxonomy:
        raise KeyError(f"Head set {name!r} not found in taxonomy")
    return taxonomy[name]
