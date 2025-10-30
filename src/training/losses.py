"""Criterion (loss function) factory."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _tensor_or_none(x: Iterable[float] | None) -> torch.Tensor | None:
    """Convert a Python list to a float tensor on CPU, or return None."""
    if x is None:
        return None
    t = torch.tensor(list(x), dtype=torch.float32)
    return t


def build_criterion(cfg: dict) -> nn.Module:
    """Build a PyTorch loss function from a config dictionary.

    Supported:
        - cross_entropy: nn.CrossEntropyLoss (supports class weights)
        - focal (stub): raises NotImplementedError for now

    Args:
        cfg: Configuration with a 'train.loss' section.

    Returns:
        An `nn.Module` loss instance.
    """
    loss_cfg = dict(cfg.get("train", {}).get("loss", {}) or {})
    name = str(loss_cfg.get("name", "cross_entropy")).lower()

    if name == "cross_entropy":
        weight_list = loss_cfg.get("weight", None)
        weight = _tensor_or_none(weight_list)
        return nn.CrossEntropyLoss(weight=weight)

    if name == "focal":
        raise NotImplementedError("Focal loss is not implemented yet.")

    raise ValueError(f"Unknown loss: {name}")
