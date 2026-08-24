import pytest
from pydantic import ValidationError

from preference_lab.schemas import PreferenceExample


def test_text_fields_are_stripped() -> None:
    example = PreferenceExample(prompt=" prompt ", chosen=" good ", rejected=" bad ")
    assert (example.prompt, example.chosen, example.rejected) == ("prompt", "good", "bad")


def test_responses_must_be_meaningfully_different() -> None:
    with pytest.raises(ValidationError, match="meaningfully different"):
        PreferenceExample(prompt="p", chosen="The SAME answer!", rejected=" the same answer ")


def test_whitespace_only_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(prompt="   ", chosen="good", rejected="bad")
