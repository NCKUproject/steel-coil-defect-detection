"""Train an MLP baseline for steel-coil classification."""

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
from src.utils.constants import RANDOM_SEED
from src.utils.seed import set_global_seed

# ensure @register("mlp")
importlib.import_module("src.models.mlp")

importlib.import_module("src.models.mlp")


def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate accuracy, precision and recall on a given dataloader."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, _f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, float(precision), float(recall)


def main() -> None:
    """Train the MLP model using parameters from configs/base.yaml and save artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    set_global_seed(RANDOM_SEED)

    cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, class_names = create_dataloader_from_cfg(cfg, phase="train")
    test_loader, _ = create_dataloader_from_cfg(cfg, phase="test")
    num_classes = len(class_names)

    img_size = int(
        cfg.get("data", {}).get("image_size") or cfg.get("data", {}).get("img_size") or 128
    )
    in_dim = img_size * img_size
    mlp_cfg = cfg.get("model", {}).get("mlp", {}) or {}
    hidden = list(mlp_cfg.get("hidden", [512, 256]))
    dropout = float(mlp_cfg.get("dropout", 0.2))

    model = build("mlp", in_dim=in_dim, num_classes=num_classes, hidden=hidden, dropout=dropout).to(
        device
    )

    epochs = int(cfg.get("train", {}).get("epochs", 10))

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    out_root = Path(cfg.get("paths", {}).get("outputs", "outputs"))
    model_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    history_train, history_test = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        if scheduler is not None:
            scheduler.step()
        train_acc, train_prec, train_rec = evaluate(model, train_loader, device)
        test_acc, test_prec, test_rec = evaluate(model, test_loader, device)
        history_train.append(train_acc)
        history_test.append(test_acc)

        avg_loss = running_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch:02d}/{epochs} | loss {avg_loss:.4f} | "
            f"train acc {train_acc:.3f} prec {train_prec:.3f} rec {train_rec:.3f} | "
            f"test acc {test_acc:.3f} prec {test_prec:.3f} rec {test_rec:.3f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {"model_state": model.state_dict(), "config": cfg, "classes": class_names},
                model_dir / "best_mlp.pt",
            )

    fig = plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history_train) + 1), history_train, label="Train Acc")
    plt.plot(range(1, len(history_test) + 1), history_test, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(metrics_dir / "mlp_accuracy_curve.png", dpi=150)
    plt.close(fig)

    print("Saved:", model_dir / "best_mlp.pt")
    print("Saved:", metrics_dir / "mlp_accuracy_curve.png")


if __name__ == "__main__":
    main()
