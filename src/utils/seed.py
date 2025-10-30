"""Utilities to set global random seeds for full reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy and PyTorch (CPU & CUDA) deterministically.

    Args:
        seed: The integer used to initialize random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic (may reduce speed on some ops)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
