"""Generate stratified 8:2 train/test CSV splits from class-folder dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.constants import RANDOM_SEED

# Comments go above the code they describe:
# Root folders for input dataset and output CSV files.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan_dataset() -> pd.DataFrame:
    """Scan class-folder dataset and return a DataFrame with (path, label)."""
    rows: list[dict[str, str]] = []
    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for p in class_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"path": p.as_posix(), "label": label})
    return pd.DataFrame(rows)


def main() -> None:
    """Create stratified 8:2 train/test CSV files using the global RANDOM_SEED."""
    df = scan_dataset()
    assert not df.empty, f"Dataset is empty under {DATA_DIR}"
    print("Total samples:", len(df))
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_SEED
    )
    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)
    print("Saved:", SPLIT_DIR / "train.csv", "and", SPLIT_DIR / "test.csv")


if __name__ == "__main__":
    main()
