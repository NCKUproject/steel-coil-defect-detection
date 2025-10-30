"""Dataset and DataLoader builders for the steel-coil classification project."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dataprep.transforms import build_transforms, build_transforms_from_cfg
from src.utils.constants import (
    CLASSES_TXT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    TEST_CSV,
    TRAIN_CSV,
)


def _read_class_names(path: Path) -> list[str]:
    """Read class names (one per line) from a text file.

    Args:
        path: Path to the classes text file.

    Returns:
        A list of class names in the order defined by the file.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


class SteelDataset(Dataset):
    """Classification dataset reading samples from a CSV file.

    The CSV must contain two columns: ``path`` and ``label``.
    Images are loaded with PIL and converted to RGB. Transforms are applied
    on-the-fly (see ``transforms`` argument).

    Args:
        csv_path: Path to the CSV file (with columns ``path`` and ``label``).
        classes_txt: Path to the classes file used to define label order.
        transforms: A composed torchvision transform to apply to PIL images.

    Raises:
        AssertionError: If the CSV yields no valid samples after validation.
    """

    def __init__(self, csv_path: Path, classes_txt: Path, transforms):
        """Initialize the dataset."""
        self.df = pd.read_csv(csv_path)
        self.classes = _read_class_names(classes_txt)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transforms = transforms

        # Remove rows whose file does not exist or whose label is unknown.
        self.df = self.df[
            self.df["path"].apply(lambda p: Path(p).exists()) & self.df["label"].isin(self.classes)
        ].reset_index(drop=True)

        assert not self.df.empty, f"No valid samples found in {csv_path}"

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        """Return a single (image_tensor, label_index) pair.

        Args:
            idx: Zero-based index into the dataset.

        Returns:
            A tuple of (image_tensor, label_index).
        """
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        label_name = row["label"]
        label_idx = self.class_to_idx[label_name]

        # Convert to RGB to be compatible with pretrained CNNs.
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transforms(img)
        return img_tensor, label_idx


def create_dataloader_from_cfg(cfg: dict, phase: str = "train") -> Tuple[DataLoader, list[str]]:
    """Create a DataLoader by reading paths and hyperparameters from a YAML config.

    The function expects the following keys in the config (with sensible defaults):
        data.train_csv / data.test_csv
        data.image_size, data.normalize, data.augment
        runtime.num_workers (optional)
        train.batch_size (for train), eval.batch_size (for test) – fallback to DEFAULT_BATCH_SIZE

    Args:
        cfg: Parsed YAML configuration.
        phase: Either ``"train"`` or ``"test"``.

    Returns:
        A pair of (dataloader, class_names).
    """
    assert phase in {"train", "test"}
    data_cfg = dict(cfg.get("data", {}))

    # optional section; we still default below
    paths = dict(cfg.get("paths", {}))

    csv_path = Path(
        data_cfg.get(
            "train_csv" if phase == "train" else "test_csv",
            paths.get(
                "train_csv" if phase == "train" else "test_csv",
                TRAIN_CSV if phase == "train" else TEST_CSV,
            ),
        )
    )
    classes_txt = Path(paths.get("classes_txt", CLASSES_TXT))

    # Build transforms from config.
    tfm = build_transforms_from_cfg(cfg, phase=phase)

    # Batch-size and workers.
    if phase == "train":
        batch_size = int(cfg.get("train", {}).get("batch_size", DEFAULT_BATCH_SIZE))
    else:
        batch_size = int(
            cfg.get("eval", {}).get(
                "batch_size", cfg.get("train", {}).get("batch_size", DEFAULT_BATCH_SIZE)
            )
        )
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))

    ds = SteelDataset(csv_path=csv_path, classes_txt=classes_txt, transforms=tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, ds.classes


def create_dataloader_explicit(
    csv_path: str = TRAIN_CSV,
    classes_txt: str = CLASSES_TXT,
    phase: str = "train",
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    normalize: str = "imagenet",
    augment_enabled: bool = False,
    augment_params: dict | None = None,
) -> Tuple[DataLoader, list[str]]:
    """Create a DataLoader using explicit arguments (no YAML required).

    Args:
        csv_path: Path to the CSV listing samples and labels.
        classes_txt: Path to the classes file.
        phase: ``"train"`` or ``"test"``.
        image_size: Target size for resizing (square).
        batch_size: Batch size used by the DataLoader.
        num_workers: Number of worker processes for background data loading.
        normalize: Normalization scheme ("imagenet" or "none").
        augment_enabled: Whether to enable augmentation.
        augment_params: Optional dictionary with augmentation parameters.

    Returns:
        A pair of (dataloader, class_names).
    """
    tfm = build_transforms(
        image_size=image_size,
        normalize=normalize,
        phase=phase,
        augment_enabled=augment_enabled,
        augment_params=augment_params,
    )
    ds = SteelDataset(csv_path=Path(csv_path), classes_txt=Path(classes_txt), transforms=tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, ds.classes
