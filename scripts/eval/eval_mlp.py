"""Evaluate the best MLP checkpoint on the test split and save reports."""

from __future__ import annotations

import csv
import importlib
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataprep.dataset import create_dataloader_from_cfg
from src.models.registry import build

# Named constants for output artifacts.
BEST_CKPT_NAME: str = "best_mlp.pt"
CONFUSION_PNG_NAME: str = "mlp_confusion_matrix.png"
CLASS_REPORT_CSV_NAME: str = "mlp_classification_report.csv"
DEFAULT_CONFIG_PATH: str = "configs/base.yaml"


def _safe_load_checkpoint(path: Path) -> dict:
    """Load a PyTorch checkpoint with forward and backward compatibility.

    Tries to use the safer `weights_only=True` flag if available. Falls back
    to the legacy behavior for older PyTorch versions.
    """
    try:
        state = torch.load(path, weights_only=True, map_location="cpu")
    except TypeError:
        state = torch.load(path, map_location="cpu")
    return state


def _build_mlp_from_ckpt_cfg(
    ckpt_cfg: dict, num_classes: int, device: torch.device
) -> torch.nn.Module:
    """Construct an MLP using configuration values stored in the checkpoint.

    This function ensures that the input dimensionality and hidden layers match
    the training configuration used to produce the checkpoint.
    """
    importlib.import_module("src.models.mlp")

    data_cfg = ckpt_cfg.get("data", {}) or {}
    img_size = int(data_cfg.get("image_size") or data_cfg.get("img_size") or 128)
    in_dim = img_size * img_size

    mlp_cfg = ckpt_cfg.get("model", {}).get("mlp", {}) or {}
    hidden = list(mlp_cfg.get("hidden", [512, 256]))
    dropout = float(mlp_cfg.get("dropout", 0.2))

    model = build("mlp", in_dim=in_dim, num_classes=num_classes, hidden=hidden, dropout=dropout)
    model = model.to(device)
    return model


def _build_test_loader_from_ckpt_cfg(runtime_cfg: dict, ckpt_cfg: dict):
    """Create a test DataLoader using the checkpoint's data settings.

    The function copies the current runtime config and overrides fields that
    affect tensor shapes and transforms, such as `model.name` and `data.image_size`,
    to match the checkpoint. This prevents shape mismatches at evaluation time.
    """
    cfg_for_loader = deepcopy(runtime_cfg)
    cfg_for_loader.setdefault("model", {})
    cfg_for_loader["model"]["name"] = "mlp"

    ckpt_data = ckpt_cfg.get("data", {}) or {}
    image_size = int(ckpt_data.get("image_size") or ckpt_data.get("img_size") or 128)
    cfg_for_loader.setdefault("data", {})
    cfg_for_loader["data"]["image_size"] = image_size

    loader, class_names = create_dataloader_from_cfg(cfg_for_loader, phase="test")
    return loader, class_names


def _gather_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return arrays of ground-truth labels and predictions."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu())
            targets.append(yb.cpu())
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    return y_true, y_pred


def _plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    """Save a confusion matrix heatmap image to the given path."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_classification_report(report: dict, accuracy: float, out_csv: Path) -> None:
    """Write a classification report dictionary and accuracy to a CSV file."""
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for key, vals in report.items():
            if (
                isinstance(vals, dict)
                and {"precision", "recall", "f1-score", "support"} <= vals.keys()
            ):
                writer.writerow(
                    [key, vals["precision"], vals["recall"], vals["f1-score"], vals["support"]]
                )
        writer.writerow([])
        writer.writerow(["accuracy", accuracy])


def main() -> None:
    """Evaluate MLP on the test split and export metrics under outputs/metrics/."""
    cfg = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(cfg.get("paths", {}).get("outputs", "outputs"))
    model_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / BEST_CKPT_NAME
    state = _safe_load_checkpoint(ckpt_path)
    ckpt_cfg = state.get("config", cfg)

    test_loader, class_names = _build_test_loader_from_ckpt_cfg(cfg, ckpt_cfg)
    num_classes = len(class_names)

    model = _build_mlp_from_ckpt_cfg(ckpt_cfg, num_classes=num_classes, device=device)
    model.load_state_dict(state["model_state"])

    y_true, y_pred = _gather_predictions(model, test_loader, device)
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    _plot_confusion_matrix(cm, class_names, metrics_dir / CONFUSION_PNG_NAME)
    _write_classification_report(report, acc, metrics_dir / CLASS_REPORT_CSV_NAME)

    print(f"Test accuracy: {acc:.4f}")
    print("Saved:", metrics_dir / CONFUSION_PNG_NAME)
    print("Saved:", metrics_dir / CLASS_REPORT_CSV_NAME)


if __name__ == "__main__":
    main()
