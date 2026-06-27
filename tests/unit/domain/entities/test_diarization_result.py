"""Unit tests for DiarizationResult domain entity."""

import pandas as pd
import pytest

from app.domain.entities.diarization_result import DiarizationResult


@pytest.mark.unit
def test_unique_speaker_labels_and_count() -> None:
    """Speaker labels and count are derived from diarization segments."""
    result = DiarizationResult(
        segments=pd.DataFrame(
            [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
                {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
                {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
            ]
        )
    )

    assert result.speaker_count() == 2
    assert result.unique_speaker_labels() == ["SPEAKER_00", "SPEAKER_01"]
