"""Train a VGG16 transfer-learning model for steel-coil classification."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.dataprep.dataset import create_dataloader_from_cfg
from src.models.registry import build
from src.training.losses import build_criterion
from src.training.optim import build_optimizer, build_scheduler

BEST_CKPT = "best_vgg16.pt"
CURVE_PNG = "vgg16_accuracy_curve.png"


def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate accuracy, macro precision and macro recall."""
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
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, _f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, float(precision), float(recall)


def main() -> None:
    """Train VGG16 using config values and save artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    importlib.import_module("src.models.vgg")

    train_loader, class_names = create_dataloader_from_cfg(cfg, phase="train")
    test_loader, _ = create_dataloader_from_cfg(cfg, phase="test")
    num_classes = len(class_names)

    vgg_cfg = cfg.get("model", {}).get("vgg16", {}) or {}
    model = build(
        "vgg16",
        num_classes=num_classes,
        pretrained=bool(vgg_cfg.get("pretrained", True)),
        freeze_features=bool(vgg_cfg.get("freeze_features", False)),
        dropout=float(vgg_cfg.get("dropout", 0.5)),
        classifier_hidden=int(vgg_cfg.get("classifier_hidden", 512)),
    ).to(device)

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    out_root = Path(cfg.get("paths", {}).get("outputs", "outputs"))
    model_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg.get("train", {}).get("epochs", 10))
    best_acc = -1.0
    hist_tr, hist_te = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        if scheduler is not None:
            scheduler.step()

        tr_acc, tr_prec, tr_rec = evaluate(model, train_loader, device)
        te_acc, te_prec, te_rec = evaluate(model, test_loader, device)
        hist_tr.append(tr_acc)
        hist_te.append(te_acc)

        avg_loss = running / max(1, len(train_loader))
        print(
            f"Epoch {epoch:02d}/{epochs} | loss {avg_loss:.4f} | "
            f"train acc {tr_acc:.3f} prec {tr_prec:.3f} rec {tr_rec:.3f} | "
            f"test acc {te_acc:.3f} prec {te_prec:.3f} rec {te_rec:.3f}"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {"model_state": model.state_dict(), "config": cfg, "classes": class_names},
                model_dir / BEST_CKPT,
            )

    fig = plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hist_tr) + 1), hist_tr, label="Train Acc")
    plt.plot(range(1, len(hist_te) + 1), hist_te, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(metrics_dir / CURVE_PNG, dpi=150)
    plt.close(fig)

    print("Saved:", model_dir / BEST_CKPT)
    print("Saved:", metrics_dir / CURVE_PNG)


if __name__ == "__main__":
    main()
