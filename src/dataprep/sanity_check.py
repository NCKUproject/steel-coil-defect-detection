"""Quick dataset sanity checks: shape, label histogram, and image grid preview."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.dataprep.dataset import create_dataloader_from_cfg
from src.utils.constants import IMAGENET_MEAN, IMAGENET_STD


def _to_numpy_img(t: torch.Tensor, model_type: str, image_size: int) -> "np.ndarray":
    """Convert a tensor to a NumPy image for matplotlib.

    Supports:
        - CNN path: tensors shaped (3,H,W) normalized by ImageNet stats
        - Grayscale path: tensors shaped (1,H,W)
        - MLP path: flattened tensors shaped (H*W,) to be reshaped to (H,W)

    Args:
        t: Image tensor.
        model_type: Model name, e.g., "mlp" or "vgg16".
        image_size: Target square size used in transforms.

    Returns:
        A NumPy array in HxW (grayscale) or HxWx3 (RGB) suitable for imshow.
    """
    if t.dim() == 1:
        if model_type == "mlp":
            arr = t.view(image_size, image_size).cpu().numpy()
            return arr
        raise ValueError("Received a flattened tensor for a non-MLP model.")

    if t.dim() == 3:
        c, h, w = t.shape
        if c == 3:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            x = (t * std + mean).clamp(0, 1)
            return x.permute(1, 2, 0).cpu().numpy()
        if c == 1:
            return t.squeeze(0).cpu().numpy()
        raise ValueError(f"Unsupported channel count: {c}")

    raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")


def _save_grid(
    images: Sequence[torch.Tensor],
    labels: Sequence[int],
    class_names: Sequence[str],
    out: Path,
    model_type: str,
    image_size: int,
) -> None:
    """Save a small grid of images for quick inspection.

    Args:
        images: Sequence of image tensors.
        labels: Integer labels aligned with images.
        class_names: List mapping label indices to names.
        out: Output PNG path.
        model_type: Model name to guide rendering behavior.
        image_size: Square size used to unflatten MLP inputs.
    """
    import math

    n = min(len(images), 16)
    cols = 4
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        img_np = _to_numpy_img(images[i], model_type=model_type, image_size=image_size)
        if img_np.ndim == 2:
            ax.imshow(img_np, cmap="gray")
        else:
            ax.imshow(img_np)
        ax.set_title(class_names[labels[i]], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run sanity checks using configs/base.yaml and save artifacts under outputs/preview/."""
    cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))
    model_type = str(cfg.get("model", {}).get("name", "cnn")).lower()
    image_size = int(
        cfg.get("data", {}).get("image_size") or cfg.get("data", {}).get("img_size", 224)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build loader from cfg; auto shuffle for train
    loader, class_names = create_dataloader_from_cfg(cfg, phase="train")

    # Fetch one batch
    # (B,3,H,W), (B,)
    xb, yb = next(iter(loader))
    print("Batch images:", tuple(xb.shape))
    print("Batch labels:", tuple(yb.shape), "min/max:", int(yb.min()), int(yb.max()))
    print("Num classes:", len(class_names), class_names[:3], "...")

    # Label histogram
    bincount = torch.bincount(yb, minlength=len(class_names))
    print("Label histogram (this batch):", bincount.tolist())

    # Save preview grid
    out_dir = Path("outputs/preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_grid(
        list(xb),
        list(yb.tolist()),
        class_names,
        out_dir / "train_batch_preview.png",
        model_type=model_type,
        image_size=image_size,
    )
    print("Saved:", out_dir / "train_batch_preview.png")
    print("Device:", device)


if __name__ == "__main__":
    main()
