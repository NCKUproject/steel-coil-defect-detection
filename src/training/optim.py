"""Optimizer and scheduler factories."""

from __future__ import annotations

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


def build_optimizer(model: torch.nn.Module, cfg: dict) -> Optimizer:
    """Build an optimizer from 'train.optimizer' config."""
    opt_cfg = dict(cfg.get("train", {}).get("optimizer", {}) or {})
    name = str(opt_cfg.get("name", "adam")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if name == "adam":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True
        )

    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: Optimizer, cfg: dict) -> _LRScheduler | None:
    """Build an LR scheduler from 'train.scheduler' config. Returns None if disabled."""
    sch_cfg = dict(cfg.get("train", {}).get("scheduler", {}) or {})
    name = sch_cfg.get("name", None)
    if not name:
        return None
    name = str(name).lower()

    if name == "step":
        step_size = int(sch_cfg.get("step_size", 5))
        gamma = float(sch_cfg.get("gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "onecycle":
        max_lr = float(sch_cfg.get("max_lr", 1e-3))
        total_steps = int(sch_cfg.get("total_steps", 0))
        if total_steps <= 0:
            raise ValueError("OneCycle requires 'total_steps' > 0.")
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

    raise ValueError(f"Unknown scheduler: {name}")
