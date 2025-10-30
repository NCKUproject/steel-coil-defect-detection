"""Composable image transforms for training and evaluation."""

from __future__ import annotations

from typing import Literal

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.utils.constants import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _norm_block(scheme: Literal["imagenet", "none"]) -> list:
    """Return a normalization block according to the given scheme.

    Args:
        scheme: Normalization scheme. Supports "imagenet" or "none".

    Returns:
        A list of torchvision transforms composing the normalization step.
    """
    if scheme == "imagenet":
        return [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return []


def _augment_block(cfg: dict) -> list:
    """Create an augmentation block based on config values.

    Args:
        cfg: A dictionary-like object under `data.augment` in YAML.

    Returns:
        A list of torchvision transforms composing the augmentation step.
    """
    aug = []
    if not cfg.get("enabled", False):
        return aug

    hflip_p = float(cfg.get("hflip_prob", 0.0))
    vflip_p = float(cfg.get("vflip_prob", 0.0))
    rot_deg = float(cfg.get("rotation_deg", 0.0))
    brightness = float(cfg.get("brightness", 0.0))
    contrast = float(cfg.get("contrast", 0.0))
    saturation = float(cfg.get("saturation", 0.0))
    hue = float(cfg.get("hue", 0.0))
    use_blur = bool(cfg.get("gaussian_blur", False))

    if hflip_p > 0:
        aug.append(transforms.RandomHorizontalFlip(p=hflip_p))
    if vflip_p > 0:
        aug.append(transforms.RandomVerticalFlip(p=vflip_p))
    if rot_deg > 0:
        aug.append(
            transforms.RandomRotation(degrees=rot_deg, interpolation=InterpolationMode.BILINEAR)
        )
    if any(v > 0 for v in (brightness, contrast, saturation, hue)):
        aug.append(
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            )
        )
    if use_blur:
        aug.append(transforms.GaussianBlur(kernel_size=3))

    return aug


def build_transforms_from_cfg(cfg: dict, phase: str = "train"):
    """Build torchvision transforms based on configuration and phase.

    Args:
        cfg: Configuration dictionary parsed from base.yaml.
        phase: Either "train" or "test".

    Returns:
        torchvision.transforms.Compose: A composed transformation pipeline.
    """
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    # Accept both keys: "image_size" and "img_size".
    img_size = int(data_cfg.get("image_size") or data_cfg.get("img_size") or 224)
    model_type = str(model_cfg.get("name", "cnn")).lower()

    # === Transform for MLP models (flattened grayscale input) ===
    if model_type == "mlp":
        mlp_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # flatten into 1D tensor
            transforms.Lambda(lambda x: x.view(-1)),
        ]
        return transforms.Compose(mlp_transforms)

    # === Transform for CNN-based models (RGB input) ===
    if phase == "train":
        train_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        return transforms.Compose(train_transforms)

    test_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(test_transforms)


def build_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    normalize: Literal["imagenet", "none"] = "imagenet",
    phase: Literal["train", "test"] = "train",
    augment_enabled: bool = False,
    augment_params: dict | None = None,
) -> transforms.Compose:
    """Build a torchvision transform pipeline using explicit arguments.

    This helper mirrors `build_transforms_from_cfg` but is convenient for scripts
    that do not load a YAML configuration.

    Args:
        image_size: Target square size for resizing.
        normalize: Normalization scheme name, "imagenet" or "none".
        phase: Either "train" or "test".
        augment_enabled: Whether to enable the augmentation block.
        augment_params: Optional dictionary to override augmentation parameters.

    Returns:
        A composed torchvision transform ready to be applied to PIL images.
    """
    cfg = {
        "data": {
            "image_size": image_size,
            "normalize": normalize,
            "augment": {"enabled": augment_enabled, **(augment_params or {})},
        }
    }
    return build_transforms_from_cfg(cfg, phase=phase)
