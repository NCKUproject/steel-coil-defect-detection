"""VGG16 transfer-learning model with a configurable classifier head."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.models.registry import register


def _load_vgg16(pretrained: bool) -> nn.Module:
    """Create a VGG16-BN backbone, optionally with pretrained ImageNet weights."""
    if pretrained:
        try:
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1
            model = models.vgg16_bn(weights=weights)
        except AttributeError:
            model = models.vgg16_bn(pretrained=True)
    else:
        if hasattr(models, "VGG16_BN_Weights"):
            model = models.vgg16_bn(weights=None)
        else:
            model = models.vgg16_bn(pretrained=False)
    return model


def _build_classifier(
    in_features: int, num_classes: int, hidden: int, dropout: float
) -> nn.Sequential:
    """Return a compact classifier head starting from the flattened feature size."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, num_classes),
    )


@register("vgg16")
def build_vgg16(
    num_classes: int,
    pretrained: bool = True,
    freeze_features: bool = False,
    dropout: float = 0.5,
    classifier_hidden: int = 512,
) -> nn.Module:
    """Build a VGG16-BN model with a small classifier head.

    Notes:
        The original VGG16-BN classifier expects a flattened input of 25088
        elements (7x7x512). The custom head must start from this size.
    """
    model = _load_vgg16(pretrained=pretrained)

    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False

    # The first linear layer of the original classifier takes 25088 inputs.
    in_features = model.classifier[0].in_features

    # Replace the entire classifier with a compact head that starts at 25088.
    model.classifier = _build_classifier(
        in_features=in_features,
        num_classes=num_classes,
        hidden=classifier_hidden,
        dropout=dropout,
    )
    return model
