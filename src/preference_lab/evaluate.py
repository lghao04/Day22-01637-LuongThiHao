from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

from .schemas import PreferenceExample


def deterministic_score(text: str) -> float:
    """Return a reproducible CPU-only proxy score based on lexical information."""
    words = re.findall(r"\b\w+\b", text.casefold())
    if not words:
        return 0.0
    diversity = len(set(words)) / len(words)
    return math.log1p(len(words)) + diversity


def pairwise_accuracy(
    examples: Sequence[PreferenceExample],
    chosen_scores: Sequence[float],
    rejected_scores: Sequence[float],
    *,
    tie_value: float = 0.0,
) -> float:
    """Return pairwise accuracy, assigning ``tie_value`` credit to tied scores."""
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError("examples, chosen_scores, and rejected_scores must have equal lengths")
    if not 0 <= tie_value <= 1:
        raise ValueError("tie_value must be between 0 and 1")
    if not all(math.isfinite(score) for score in (*chosen_scores, *rejected_scores)):
        raise ValueError("scores must contain only finite values")
    if not examples:
        return 0.0

    credit = sum(
        1.0 if chosen > rejected else tie_value if chosen == rejected else 0.0
        for chosen, rejected in zip(chosen_scores, rejected_scores, strict=True)
    )
    return credit / len(examples)


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
