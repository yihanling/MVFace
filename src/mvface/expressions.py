"""Canonical FaceScape expression table."""

from __future__ import annotations

import json
from pathlib import Path

EXPRESSIONS_JSON = Path(__file__).resolve().parent / "assets" / "expressions.json"


def load_expressions(path: str | Path | None = None) -> dict[int, str]:
    p = Path(path) if path is not None else EXPRESSIONS_JSON
    return {int(k): v for k, v in json.loads(p.read_text()).items()}


EXPRESSIONS: dict[int, str] = load_expressions()
EXPRESSION_IDS: list[int] = sorted(EXPRESSIONS)


def stem(exp_id: int) -> str:
    try:
        return EXPRESSIONS[int(exp_id)]
    except KeyError:
        raise KeyError(f"unknown expression id {exp_id}; "
                       f"known ids are {EXPRESSION_IDS}") from None


def parse_expression_spec(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(EXPRESSION_IDS)
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    unknown = sorted(out - set(EXPRESSION_IDS))
    if unknown:
        raise ValueError(f"unknown expression ids {unknown}; "
                         f"known ids are {EXPRESSION_IDS}")
    return sorted(out)
