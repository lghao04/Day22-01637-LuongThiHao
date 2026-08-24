import json
from pathlib import Path

import pytest

from preference_lab.schemas import PreferenceExample
from preference_lab.trainers import PreferenceTrainer, TrainingConfig


@pytest.mark.parametrize("method", ["dpo", "orpo", "mock"])
def test_cpu_trainer_writes_metrics(tmp_path: Path, method: str) -> None:
    config = TrainingConfig(method=method, output_dir=tmp_path)
    examples = [
        PreferenceExample(
            prompt="Explain testing",
            chosen="Testing verifies expected behavior and catches regressions.",
            rejected="It does things.",
        )
    ]

    metrics = PreferenceTrainer(config).train(examples)
    saved = json.loads((tmp_path / "training_metrics.json").read_text(encoding="utf-8"))

    assert saved == metrics
    assert metrics["pairwise_accuracy"] == 1.0


def test_training_config_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        TrainingConfig(method="unknown")
