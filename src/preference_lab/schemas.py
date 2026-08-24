from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


def normalize_text(value: str) -> str:
    """Normalize text for duplicate comparisons without changing stored content."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", normalized).split())


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected", mode="before")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: ValidationInfo) -> str:
        chosen = info.data.get("chosen")
        if not isinstance(chosen, str):
            return rejected

        normalized_chosen = normalize_text(chosen)
        normalized_rejected = normalize_text(rejected)
        similarity = SequenceMatcher(None, normalized_chosen, normalized_rejected).ratio()
        if normalized_chosen == normalized_rejected or (
            min(len(normalized_chosen), len(normalized_rejected)) >= 20 and similarity >= 0.97
        ):
            raise ValueError("chosen and rejected must be meaningfully different")
        return rejected
