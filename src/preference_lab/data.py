from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample, normalize_text

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){9,15}(?!\d)")


def _contains_pii(example: PreferenceExample) -> bool:
    text = f"{example.prompt} {example.chosen} {example.rejected}"
    return bool(_EMAIL_PATTERN.search(text) or _PHONE_PATTERN.search(text))


def load_jsonl(
    path: str | Path,
    *,
    allow_duplicate_prompts: bool = False,
    check_pii: bool = False,
) -> list[PreferenceExample]:
    """Load and validate preference examples from a JSONL file."""
    source = Path(path)
    examples: list[PreferenceExample] = []
    seen_prompts: dict[str, int] = {}
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                example = PreferenceExample.model_validate(raw)
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(f"{source}: line {line_number}: {error}") from error

            prompt_key = normalize_text(example.prompt)
            if not allow_duplicate_prompts and prompt_key in seen_prompts:
                first_line = seen_prompts[prompt_key]
                raise ValueError(
                    f"{source}: line {line_number}: duplicate prompt (first seen on line {first_line})"
                )
            if check_pii and _contains_pii(example):
                raise ValueError(
                    f"{source}: line {line_number}: possible email address or phone number"
                )

            seen_prompts.setdefault(prompt_key, line_number)
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    *,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Deterministically split prompt groups so they cannot leak across sets."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        return [], []

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(normalize_text(example.prompt), []).append(example)

    prompt_keys = list(groups)
    random.Random(seed).shuffle(prompt_keys)
    if len(prompt_keys) == 1:
        return list(groups[prompt_keys[0]]), []

    validation_group_count = round(len(prompt_keys) * validation_ratio)
    validation_group_count = min(max(1, validation_group_count), len(prompt_keys) - 1)
    validation_keys = set(prompt_keys[:validation_group_count])
    train = [item for key in prompt_keys if key not in validation_keys for item in groups[key]]
    validation = [item for key in prompt_keys if key in validation_keys for item in groups[key]]
    return train, validation
