"""Random seeds and device selection."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device() -> torch.device:
    """CUDA if available, otherwise CPU (as in the dissertation scripts)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
