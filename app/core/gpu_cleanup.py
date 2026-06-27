"""GPU memory cleanup utilities."""

import gc

import torch


def release_gpu_resources() -> None:
    """Release Python and CUDA memory after GPU work or failures."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
