from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import deterministic_score, pairwise_accuracy
from .losses import dpo_loss, orpo_loss
from .schemas import PreferenceExample


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    output_dir: str | Path = "outputs"

    def __post_init__(self) -> None:
        if self.method not in {"dpo", "orpo", "mock"}:
            raise ValueError("method must be one of: dpo, orpo, mock")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.lambda_orpo < 0:
            raise ValueError("lambda_orpo must be non-negative")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")


class PreferenceTrainer:
    """Deterministic CPU trainer used to exercise the preference pipeline locally."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, examples: Sequence[PreferenceExample]) -> dict[str, float]:
        """Score examples, calculate the configured objective, and save metrics."""
        if not examples:
            raise ValueError("training requires at least one preference example")

        chosen_score_values = [deterministic_score(example.chosen) for example in examples]
        rejected_score_values = [deterministic_score(example.rejected) for example in examples]
        chosen_scores = np.asarray(chosen_score_values, dtype=np.float64)
        rejected_scores = np.asarray(rejected_score_values, dtype=np.float64)
        chosen_logps = -np.logaddexp(0.0, -chosen_scores)
        rejected_logps = -np.logaddexp(0.0, -rejected_scores)

        if self.config.method == "dpo":
            reference = np.full_like(chosen_logps, -np.log(2.0))
            loss = dpo_loss(
                chosen_logps, rejected_logps, reference, reference, beta=self.config.beta
            )
        elif self.config.method == "orpo":
            loss = orpo_loss(
                -chosen_logps,
                chosen_logps,
                rejected_logps,
                lambda_orpo=self.config.lambda_orpo,
            )
        else:
            loss = float(np.mean(-chosen_logps))

        metrics = {
            "example_count": float(len(examples)),
            "final_loss": loss,
            "pairwise_accuracy": pairwise_accuracy(
                examples, chosen_score_values, rejected_score_values
            ),
        }
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        return metrics
