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
        model = (
            models.vgg16_bn(weights=None)
            if hasattr(models, "VGG16_BN_Weights")
            else models.vgg16_bn(pretrained=False)
        )
    return model


def _build_classifier(
    in_features: int, num_classes: int, hidden: int, dropout: float
) -> nn.Sequential:
    """Return a compact classifier head for VGG features."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
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

    Args:
        num_classes: Number of classes.
        pretrained: Whether to load ImageNet weights.
        freeze_features: If True, freeze convolutional feature extractor.
        dropout: Dropout rate used in the classifier head.
        classifier_hidden: Hidden dimension of the classifier head.
    """
    model = _load_vgg16(pretrained=pretrained)

    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier = _build_classifier(in_features, num_classes, classifier_hidden, dropout)
    return model
