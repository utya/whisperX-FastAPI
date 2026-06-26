"""Tests for GPU cleanup helper."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.gpu_cleanup import release_gpu_resources


@pytest.mark.unit
@patch("app.core.gpu_cleanup.torch")
@patch("app.core.gpu_cleanup.gc")
def test_release_gpu_resources_calls_cuda_empty_cache(
    mock_gc: MagicMock,
    mock_torch: MagicMock,
) -> None:
    """CUDA cache is cleared when CUDA is available."""
    mock_torch.cuda.is_available.return_value = True

    release_gpu_resources()

    mock_gc.collect.assert_called_once()
    mock_torch.cuda.empty_cache.assert_called_once()


@pytest.mark.unit
@patch("app.core.gpu_cleanup.torch")
@patch("app.core.gpu_cleanup.gc")
def test_release_gpu_resources_skips_cuda_when_unavailable(
    mock_gc: MagicMock,
    mock_torch: MagicMock,
) -> None:
    """CUDA cache is not touched when CUDA is unavailable."""
    mock_torch.cuda.is_available.return_value = False

    release_gpu_resources()

    mock_gc.collect.assert_called_once()
    mock_torch.cuda.empty_cache.assert_not_called()
