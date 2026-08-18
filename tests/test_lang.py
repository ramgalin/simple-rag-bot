"""Answer-language detection.

The candidate list is a setting rather than a constant because it is the whole
fix: with all 75 languages enabled, "кто основал стоицизм?" was detected as
Serbian and the bot answered a Russian user in Serbian.
"""

import pytest
from lingua import Language

from ragbot.lang import detect_language, parse_languages


def test_parses_a_comma_separated_list():
    assert parse_languages("English,Russian") == [Language.ENGLISH, Language.RUSSIAN]


def test_tolerates_whitespace_and_casing():
    assert parse_languages(" english ,  RUSSIAN ") == [Language.ENGLISH, Language.RUSSIAN]


def test_rejects_an_unknown_language_by_name():
    with pytest.raises(ValueError, match="Klingon"):
        parse_languages("English,Klingon")


def test_rejects_a_single_language():
    """lingua needs something to choose between, and so does the feature."""
    with pytest.raises(ValueError, match="at least two"):
        parse_languages("Russian")


def test_rejects_an_empty_setting():
    with pytest.raises(ValueError):
        parse_languages("")


@pytest.mark.parametrize(
    "question",
    [
        "кто основал стоицизм?",          # was detected as Serbian across all 75
        "а где он преподавал?",
        "что такое принцип полезности?",
        "какие проекты по амбисонику у нас есть?",
    ],
)
def test_short_russian_questions_are_detected_as_russian(question):
    assert detect_language(question) == "Russian"


@pytest.mark.parametrize(
    "question",
    [
        "who founded Stoicism?",
        "what is the principle of utility?",
        "where did he teach?",
    ],
)
def test_english_questions_are_detected_as_english(question):
    assert detect_language(question) == "English"


def test_falls_back_to_english_when_undetectable():
    assert detect_language("") == "English"
