"""MLP classifier for flattened images (baseline)."""

from __future__ import annotations

import torch.nn as nn

from src.models.registry import register


class MLP(nn.Module):
    """A simple multilayer perceptron for image classification.

    Notes:
        Input expects a flattened tensor of shape (B, in_dim).
        You should handle resize/gray/flatten in your Dataset/Transforms.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden: list[int], dropout: float = 0.2):
        """Initialize the MLP."""
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returning logits of shape (B, num_classes)."""
        return self.net(x)


@register("mlp")
def build_mlp(in_dim: int, num_classes: int, hidden: list[int], dropout: float = 0.2) -> nn.Module:
    """Factory function used by the registry to create an MLP."""
    return MLP(in_dim=in_dim, num_classes=num_classes, hidden=hidden, dropout=dropout)
