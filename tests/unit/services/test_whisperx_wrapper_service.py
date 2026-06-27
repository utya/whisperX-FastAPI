"""Unit tests for the speech-to-text background pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import TaskCancelledError
from app.core.task_cancellation import clear_cancellation, request_cancellation
from app.domain.entities.diarization_result import DiarizationResult
from app.schemas import (
    AlignmentParams,
    ASROptions,
    ComputeType,
    Device,
    DiarizationParams,
    InterpolateMethod,
    SpeechToTextProcessingParams,
    TaskEnum,
    TaskStage,
    TaskStatus,
    VADOptions,
    WhisperModel,
    WhisperModelParams,
)
from app.services.whisperx_wrapper_service import (
    _ensure_not_cancelled,
    process_audio_common,
)


def _build_params(identifier: str = "task-123") -> SpeechToTextProcessingParams:
    """Build minimal speech-to-text params for pipeline tests."""
    audio = np.zeros(16000, dtype=np.float32)
    return SpeechToTextProcessingParams(
        audio=audio,
        identifier=identifier,
        vad_options=VADOptions(vad_onset=0.5, vad_offset=0.363),
        asr_options=ASROptions(
            beam_size=5,
            best_of=5,
            patience=1,
            length_penalty=1,
            temperatures=[0.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=None,
            suppress_tokens=[-1],
            suppress_numerals=True,
            hotwords=None,
        ),
        whisper_model_params=WhisperModelParams(
            language="en",
            model=WhisperModel.tiny,
            device=Device.cpu,
            device_index=0,
            compute_type=ComputeType.int8,
            task=TaskEnum.TRANSCRIBE,
            threads=0,
            batch_size=8,
            chunk_size=20,
        ),
        alignment_params=AlignmentParams(
            align_model=None,
            interpolate_method=InterpolateMethod.nearest,
            return_char_alignments=False,
        ),
        diarization_params=DiarizationParams(min_speakers=1, max_speakers=2),
    )


@pytest.mark.unit
def test_ensure_not_cancelled_raises_for_registry_entry() -> None:
    """Registry cancellation is detected before the next pipeline stage."""
    repository = MagicMock()
    task_id = "cancel-me"
    request_cancellation(task_id)
    try:
        with pytest.raises(TaskCancelledError):
            _ensure_not_cancelled(repository, task_id)
    finally:
        clear_cancellation(task_id)


@pytest.mark.unit
@patch("app.services.whisperx_wrapper_service.release_gpu_resources")
@patch("app.services.whisperx_wrapper_service.SyncSessionLocal")
def test_process_audio_common_persists_partial_text_after_transcribe(
    mock_session_local: MagicMock,
    mock_release_gpu: MagicMock,
) -> None:
    """Partial text and diarizing stage are saved after transcribe completes."""
    session = MagicMock()
    mock_session_local.return_value = session
    repository = MagicMock()
    repository.get_by_id.return_value = None

    call_order: list[str] = []

    def track_transcribe(*args: object, **kwargs: object) -> dict[str, object]:
        call_order.append("transcribe")
        return {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world."}],
        }

    def track_diarize(*args: object, **kwargs: object) -> DiarizationResult:
        call_order.append("diarize")
        return DiarizationResult(
            segments=pd.DataFrame(
                [
                    {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                    {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
                ]
            ),
            speaker_embeddings=None,
        )

    def track_align(*args: object, **kwargs: object) -> dict[str, object]:
        call_order.append("align")
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world.",
                    "words": [
                        {
                            "word": "Hello",
                            "start": 0.0,
                            "end": 0.5,
                            "score": 0.9,
                        },
                        {
                            "word": "world.",
                            "start": 0.5,
                            "end": 1.0,
                            "score": 0.9,
                        },
                    ],
                }
            ],
            "word_segments": [],
        }

    def track_combine(*args: object, **kwargs: object) -> dict[str, object]:
        call_order.append("combine")
        return {"segments": [{"text": "Hello world.", "speaker": "SPEAKER_00"}]}

    transcription_svc = MagicMock()
    transcription_svc.transcribe.side_effect = track_transcribe
    alignment_svc = MagicMock()
    alignment_svc.align.side_effect = track_align
    diarization_svc = MagicMock()
    diarization_svc.diarize.side_effect = track_diarize
    speaker_svc = MagicMock()
    speaker_svc.assign_speakers.side_effect = track_combine

    with patch(
        "app.services.whisperx_wrapper_service.SyncSQLAlchemyTaskRepository",
        return_value=repository,
    ):
        process_audio_common(
            _build_params(),
            transcription_service=transcription_svc,
            alignment_service=alignment_svc,
            diarization_service=diarization_svc,
            speaker_service=speaker_svc,
        )

    partial_update = next(
        call
        for call in repository.update.call_args_list
        if call.kwargs["update_data"].get("partial_text") == "Hello world."
    )
    assert (
        partial_update.kwargs["update_data"]["current_stage"]
        == TaskStage.diarizing.value
    )

    speaker_update = next(
        call
        for call in repository.update.call_args_list
        if call.kwargs["update_data"].get("partial_speaker_count") == 2
    )
    assert (
        speaker_update.kwargs["update_data"]["current_stage"]
        == TaskStage.aligning.value
    )
    assert speaker_update.kwargs["update_data"]["partial_speakers"] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]

    assert call_order == ["transcribe", "diarize", "align", "combine"]


@pytest.mark.unit
@patch("app.services.audio_processing_service._identify_and_store_speakers")
@patch("app.services.whisperx_wrapper_service.release_gpu_resources")
@patch("app.services.whisperx_wrapper_service.SyncSessionLocal")
def test_process_audio_common_persists_identified_speaker_labels(
    mock_session_local: MagicMock,
    mock_release_gpu: MagicMock,
    mock_identify: MagicMock,
) -> None:
    """Identified speaker names are exposed before the final result."""
    session = MagicMock()
    mock_session_local.return_value = session
    repository = MagicMock()
    repository.get_by_id.return_value = None

    transcription_svc = MagicMock()
    transcription_svc.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello."}],
    }
    alignment_svc = MagicMock()
    alignment_svc.align.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello.",
                "words": [
                    {"word": "Hello.", "start": 0.0, "end": 1.0, "score": 0.9},
                ],
            }
        ],
        "word_segments": [],
    }

    diarization_result = DiarizationResult(
        segments=pd.DataFrame([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]),
        speaker_embeddings={"SPEAKER_00": [0.1, 0.2]},
    )
    diarization_svc = MagicMock()
    diarization_svc.diarize.return_value = diarization_result

    identified_result = DiarizationResult(
        segments=pd.DataFrame([{"start": 0.0, "end": 1.0, "speaker": "Ivan"}]),
        speaker_embeddings={"Ivan": [0.1, 0.2]},
    )
    mock_identify.return_value = identified_result

    speaker_svc = MagicMock()
    speaker_svc.assign_speakers.return_value = {
        "segments": [{"text": "Hello.", "speaker": "Ivan"}]
    }

    params = _build_params()
    params.diarization_params.identify_speakers = True

    with patch(
        "app.services.whisperx_wrapper_service.SyncSQLAlchemyTaskRepository",
        return_value=repository,
    ):
        process_audio_common(
            params,
            transcription_service=transcription_svc,
            alignment_service=alignment_svc,
            diarization_service=diarization_svc,
            speaker_service=speaker_svc,
        )

    speaker_update = next(
        call
        for call in repository.update.call_args_list
        if call.kwargs["update_data"].get("partial_speaker_count") == 1
    )
    assert speaker_update.kwargs["update_data"]["partial_speakers"] == ["Ivan"]
    mock_identify.assert_called_once()


@pytest.mark.unit
@patch("app.services.whisperx_wrapper_service.release_gpu_resources")
@patch("app.services.whisperx_wrapper_service.SyncSessionLocal")
def test_process_audio_common_stops_when_cancelled_after_transcribe(
    mock_session_local: MagicMock,
    mock_release_gpu: MagicMock,
) -> None:
    """Pipeline stops after transcribe when cancellation is requested."""
    session = MagicMock()
    mock_session_local.return_value = session
    repository = MagicMock()
    repository.get_by_id.return_value = MagicMock(status=TaskStatus.cancelled)

    transcription_svc = MagicMock()
    transcription_svc.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Stop here."}],
    }
    alignment_svc = MagicMock()
    diarization_svc = MagicMock()

    request_cancellation("task-cancel")

    with patch(
        "app.services.whisperx_wrapper_service.SyncSQLAlchemyTaskRepository",
        return_value=repository,
    ):
        try:
            process_audio_common(
                _build_params("task-cancel"),
                transcription_service=transcription_svc,
                alignment_service=alignment_svc,
                diarization_service=diarization_svc,
                speaker_service=MagicMock(),
            )
        finally:
            clear_cancellation("task-cancel")

    diarization_svc.diarize.assert_not_called()
    alignment_svc.align.assert_not_called()


@pytest.mark.unit
@patch("app.services.whisperx_wrapper_service.release_gpu_resources")
@patch("app.core.gpu_semaphore.get_gpu_semaphore")
@patch("app.services.whisperx_wrapper_service.SyncSessionLocal")
def test_process_audio_common_releases_gpu_resources_on_cuda(
    mock_session_local: MagicMock,
    mock_get_gpu_semaphore: MagicMock,
    mock_release_gpu: MagicMock,
) -> None:
    """CUDA deployments always clear GPU memory in the worker finally block."""
    session = MagicMock()
    mock_session_local.return_value = session
    repository = MagicMock()
    repository.get_by_id.return_value = None
    mock_get_gpu_semaphore.return_value = MagicMock()

    transcription_svc = MagicMock()
    transcription_svc.transcribe.side_effect = RuntimeError("CUDA out of memory")

    params = _build_params("gpu-task")
    params.whisper_model_params.device = Device.cuda
    params.whisper_model_params.compute_type = ComputeType.float16

    with patch(
        "app.services.whisperx_wrapper_service.SyncSQLAlchemyTaskRepository",
        return_value=repository,
    ):
        process_audio_common(
            params,
            transcription_service=transcription_svc,
            alignment_service=MagicMock(),
            diarization_service=MagicMock(),
            speaker_service=MagicMock(),
        )

    mock_release_gpu.assert_called_once()
