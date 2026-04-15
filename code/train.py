"""
CSC3218-style CIFAR-10 training: metrics, curves, confusion matrix, early stopping,
LR scheduling, weight decay, and sample prediction figures.

Run: python train.py
Outputs go to ./results/ by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data import CIFAR10_CLASSES, get_dataloaders
from model import CIFAR10CNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def plot_curves(
    history: dict,
    out_dir: Path,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_loss_accuracy.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: tuple[str, ...], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_sample_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_path: Path,
    n_show: int = 16,
    mean: tuple[float, ...] = (0.4914, 0.4822, 0.4465),
    std: tuple[float, ...] = (0.2470, 0.2435, 0.2616),
) -> None:
    model.eval()
    images: list[torch.Tensor] = []
    trues: list[int] = []
    preds: list[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = logits.argmax(dim=1)
        for i in range(x.size(0)):
            if len(images) >= n_show:
                break
            images.append(x[i].cpu())
            trues.append(int(y[i].item()))
            preds.append(int(p[i].item()))
        if len(images) >= n_show:
            break

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_2d(axes)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx < len(images):
            img = images[idx] * std_t + mean_t
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)
            ok = trues[idx] == preds[idx]
            color = "green" if ok else "red"
            ax.set_title(
                f"T:{CIFAR10_CLASSES[trues[idx]]}\nP:{CIFAR10_CLASSES[preds[idx]]}",
                fontsize=8,
                color=color,
            )
        ax.axis("off")
    fig.suptitle("Sample predictions (T=true, P=predicted)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 CNN (CSC3218)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization (AdamW)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (epochs)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--rotation", action="store_true", help="Optional train-time rotation")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        augment_train=not args.no_augment,
        use_rotation=args.rotation,
    )

    model = CIFAR10CNN(num_classes=10, dropout_p=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state: dict | None = None
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save(best_state, out_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val acc improvement for {args.patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on splits
    tr_loss, tr_acc, _, _ = evaluate(model, train_loader, device, criterion)
    va_loss, va_acc, _, _ = evaluate(model, val_loader, device, criterion)
    te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, device, criterion)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(CIFAR10_CLASSES),
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    summary = {
        "train_accuracy": float(tr_acc),
        "val_accuracy": float(va_acc),
        "test_accuracy": float(te_acc),
        "test_loss": float(te_loss),
        "test_precision_macro": float(prec_macro),
        "test_recall_macro": float(rec_macro),
        "test_f1_macro": float(f1_macro),
        "epochs_ran": len(history["train_loss"]),
        "seconds": round(time.time() - start, 1),
        "device": str(device),
        "hyperparameters": vars(args),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n=== Final metrics ===")
    print(f"Train accuracy: {tr_acc:.4f}")
    print(f"Val accuracy:   {va_acc:.4f}")
    print(f"Test accuracy:  {te_acc:.4f}")
    print(f"Test precision (macro): {prec_macro:.4f}")
    print(f"Test recall (macro):    {rec_macro:.4f}")
    print(f"Test F1 (macro):        {f1_macro:.4f}")
    print("\nPer-class report:\n", report)

    plot_curves(history, out_dir)
    plot_confusion(cm, CIFAR10_CLASSES, out_dir / "confusion_matrix_test.png")
    plot_sample_predictions(model, test_loader, device, out_dir / "sample_predictions.png")

    print(f"\nSaved figures and metrics under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
