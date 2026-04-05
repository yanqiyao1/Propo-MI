from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm


def resolve_log_path(*, output_path: Path | None = None, output_dir: Path | None = None, filename: str = "run.log") -> Path:
    if output_dir is not None:
        return output_dir / filename
    if output_path is not None:
        if output_path.suffix:
            return output_path.with_suffix(".log")
        return output_path.parent / f"{output_path.name}.log"
    raise ValueError("Either output_path or output_dir must be provided")


def setup_file_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, payload: Any) -> None:
    if isinstance(payload, (dict, list, tuple)):
        logger.info(json.dumps(payload, ensure_ascii=False))
    else:
        logger.info(str(payload))



def make_tqdm(
    iterable: Iterable[Any] | None = None,
    *,
    total: int | None = None,
    desc: str | None = None,
    leave: bool = False,
    disable: bool = False,
):
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave, disable=disable)
