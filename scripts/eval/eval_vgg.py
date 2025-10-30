"""Evaluate the best VGG16 checkpoint on the test split and save reports."""

from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataprep.dataset import create_dataloader_from_cfg
from src.models.registry import build

BEST_CKPT = "best_vgg16.pt"
CONFUSION_PNG = "vgg16_confusion_matrix.png"
CLASS_REPORT_CSV = "vgg16_classification_report.csv"


def _plot_confusion(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    """Save a confusion matrix heatmap."""
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


def main() -> None:
    """Evaluate VGG16 on test set and export metrics."""
    cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    importlib.import_module("src.models.vgg")

    test_loader, class_names = create_dataloader_from_cfg(cfg, phase="test")
    num_classes = len(class_names)

    vgg_cfg = cfg.get("model", {}).get("vgg16", {}) or {}
    model = build(
        "vgg16",
        num_classes=num_classes,
        pretrained=bool(vgg_cfg.get("pretrained", True)),
        freeze_features=bool(vgg_cfg.get("freeze_features", False)),
        dropout=float(vgg_cfg.get("dropout", 0.5)),
        classifier_hidden=int(vgg_cfg.get("classifier_hidden", 512)),
    )

    ckpt_path = Path(cfg.get("paths", {}).get("outputs", "outputs")) / "models" / BEST_CKPT
    try:
        state = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model = model.to(device)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu())
            targets.append(yb.cpu())

    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    out_root = Path(cfg.get("paths", {}).get("outputs", "outputs"))
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    _plot_confusion(cm, class_names, metrics_dir / CONFUSION_PNG)

    import csv

    with (metrics_dir / CLASS_REPORT_CSV).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for k, v in report.items():
            if isinstance(v, dict) and {"precision", "recall", "f1-score", "support"} <= v.keys():
                writer.writerow([k, v["precision"], v["recall"], v["f1-score"], v["support"]])
        writer.writerow([])
        writer.writerow(["accuracy", acc])

    print(f"Test accuracy: {acc:.4f}")
    print("Saved:", metrics_dir / CONFUSION_PNG)
    print("Saved:", metrics_dir / CLASS_REPORT_CSV)


if __name__ == "__main__":
    main()
