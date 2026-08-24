from pathlib import Path

import pytest

from preference_lab.evaluate import deterministic_score, pairwise_accuracy, write_metrics
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_handles_ties_explicitly() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [1.0], [1.0], tie_value=0.5) == 0.5


def test_pairwise_accuracy_rejects_mismatched_lengths() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match="equal lengths"):
        pairwise_accuracy(examples, [], [1.0])


def test_deterministic_score_is_reproducible() -> None:
    text = "A clear response with several distinct words."
    assert deterministic_score(text) == deterministic_score(text)
    assert deterministic_score("") == 0.0


def test_write_metrics_creates_json_file(tmp_path: Path) -> None:
    output = write_metrics({"pairwise_accuracy": 0.75}, tmp_path / "nested")
    assert output.read_text(encoding="utf-8").endswith("}\n") is False
    assert '"pairwise_accuracy": 0.75' in output.read_text(encoding="utf-8")
