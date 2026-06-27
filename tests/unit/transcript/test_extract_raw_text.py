"""Tests for raw transcript text extraction."""

import pytest

from app.transcript import extract_raw_text_from_transcript


@pytest.mark.unit
def test_extract_raw_text_from_top_level_text() -> None:
    """Top-level text field is returned when present."""
    transcript = {"text": "Hello world.", "segments": [{"text": "ignored"}]}
    assert extract_raw_text_from_transcript(transcript) == "Hello world."


@pytest.mark.unit
def test_extract_raw_text_from_segments() -> None:
    """Segment texts are joined when top-level text is missing."""
    transcript = {
        "segments": [
            {"text": "Hello"},
            {"text": "world."},
        ]
    }
    assert extract_raw_text_from_transcript(transcript) == "Hello world."


@pytest.mark.unit
def test_extract_raw_text_empty() -> None:
    """Empty transcript yields an empty string."""
    assert extract_raw_text_from_transcript({}) == ""
