"""Lone surrogates reach us from stdin under a POSIX locale, where Python decodes
with surrogateescape. They cannot be encoded back to UTF-8, so they must never be
allowed to travel as far as the HTTP client."""

import pytest

from ragbot.textio import strip_surrogates


def test_plain_text_is_untouched():
    assert strip_surrogates("какие проекты по амбисонику?") == "какие проекты по амбисонику?"


def test_lone_surrogate_is_replaced():
    # b"\xd1" is the lead byte of a two-byte Cyrillic character
    broken = b"\xd1".decode("utf-8", "surrogateescape")
    assert strip_surrogates(broken) == "?"


def test_result_is_always_encodable():
    broken = "а" + b"\xd1".decode("utf-8", "surrogateescape") + "б"
    with pytest.raises(UnicodeEncodeError):
        broken.encode("utf-8")
    strip_surrogates(broken).encode("utf-8")  # must not raise


def test_surrounding_text_survives():
    broken = "проект" + b"\xd1".decode("utf-8", "surrogateescape")
    assert strip_surrogates(broken) == "проект?"


def test_empty_string():
    assert strip_surrogates("") == ""
