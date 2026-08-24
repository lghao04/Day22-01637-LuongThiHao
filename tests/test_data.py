from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) >= 2
    assert examples[0].chosen != examples[0].rejected


def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_split_is_deterministic_and_keeps_prompt_groups_together() -> None:
    examples = [
        PreferenceExample(prompt="Repeated", chosen="A useful answer", rejected="Bad"),
        PreferenceExample(prompt="repeated", chosen="Another useful answer", rejected="Worse"),
        PreferenceExample(prompt="Different", chosen="Correct response", rejected="Incorrect"),
    ]

    first = split_by_prompt(examples, validation_ratio=0.5, seed=7)
    second = split_by_prompt(examples, validation_ratio=0.5, seed=7)

    assert first == second
    assert not ({item.prompt.casefold() for item in first[0]} & {"repeated"}) or not (
        {item.prompt.casefold() for item in first[1]} & {"repeated"}
    )


def test_loader_reports_line_number_for_invalid_json(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_text('{"prompt": "missing end"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1"):
        load_jsonl(source)


def test_loader_rejects_normalized_duplicate_prompt(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.jsonl"
    source.write_text(
        '{"prompt":"Hello!","chosen":"Detailed answer","rejected":"No"}\n'
        '{"prompt":" hello ","chosen":"A second answer","rejected":"Wrong"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"line 2: duplicate prompt"):
        load_jsonl(source)


def test_loader_optionally_rejects_pii(tmp_path: Path) -> None:
    source = tmp_path / "pii.jsonl"
    source.write_text(
        '{"prompt":"Email me at learner@example.com","chosen":"Safe answer","rejected":"No"}\n',
        encoding="utf-8",
    )

    assert len(load_jsonl(source)) == 1
    with pytest.raises(ValueError, match="possible email address"):
        load_jsonl(source, check_pii=True)
