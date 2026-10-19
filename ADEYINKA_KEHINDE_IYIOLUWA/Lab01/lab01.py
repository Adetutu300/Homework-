#!/usr/bin/env python3
"""Lab 01 — Backpropagation and three-linear-layer MNIST classification.

Run:
    python lab01.py

The official MNIST dataset is downloaded automatically on first run.
No local datasets, model weights, or other project files are needed.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Generate figures on Colab, servers, and desktops.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Same experiment settings as the Lab 01 report.
SEED = 42
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Paths are relative to this source file, not a user's private computer or Colab.
PROJECT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ThreeLinearNN(nn.Module):
    """Three linear layers: 784 -> 256 -> 128 -> 10, with two ReLUs."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(images, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Raw logits for CrossEntropyLoss.


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            batch_size = labels.size(0)
            total_loss += loss_fn(logits, labels).item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size
    return total_loss / total, 100.0 * correct / total, correct, total


def save_plot(history, metric, destination):
    epochs = list(range(1, len(history[f"training_{metric}"]) + 1))
    title = metric.capitalize()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history[f"training_{metric}"], "o-", label=f"Training {title}")
    ax.plot(epochs, history[f"validation_{metric}"], "s-", label=f"Validation {title}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)" if metric == "accuracy" else "Cross-Entropy Loss")
    ax.set_title(f"Training and Validation {title}")
    ax.grid(True, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def smoke_test() -> None:
    """Check that the model, forward pass, and backpropagation run."""
    set_seed(SEED)
    model = ThreeLinearNN()
    sample = torch.randn(4, 1, 28, 28)
    targets = torch.tensor([0, 1, 2, 3])
    logits = model(sample)
    assert logits.shape == (4, 10), f"Unexpected model output: {logits.shape}"
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    print("SMOKE TEST PASSED: 3 linear layers, output shape (4, 10), gradients OK")


def main():
    parser = argparse.ArgumentParser(description="Lab 01: three-linear-layer MNIST")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data",
                        help="Directory for the automatically downloaded MNIST dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs",
                        help="Directory where graphs and results are saved")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick forward/backward test without downloading MNIST")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")

    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 66)
    print("LAB 01: BACKPROPAGATION AND THREE-LINEAR-LAYER MNIST CLASSIFICATION")
    print(f"Device: {device} | Seed: {SEED} | Epochs: {args.epochs}")
    print("Architecture: 784 -> 256 -> 128 -> 10")
    print("=" * 66)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    print("Loading MNIST (downloads automatically on first run)...")
    full_training_set = datasets.MNIST(
        root=str(args.data_dir), train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=str(args.data_dir), train=False, download=True, transform=transform
    )
    train_set, validation_set = random_split(
        full_training_set,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)

    model = ThreeLinearNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {
        "training_loss": [], "validation_loss": [],
        "training_accuracy": [], "validation_accuracy": [],
    }
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()  # Backpropagation via automatic differentiation.
            optimizer.step()
            batch = labels.size(0)
            running_loss += loss.item() * batch
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total
        val_loss, val_accuracy, _, _ = evaluate(
            model, validation_loader, loss_fn, device
        )
        row = {
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_loss": val_loss,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
        }
        rows.append(row)
        for metric in ("loss", "accuracy"):
            history[f"training_{metric}"].append(row[f"training_{metric}"])
            history[f"validation_{metric}"].append(row[f"validation_{metric}"])
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"train acc {train_accuracy:.2f}% | val acc {val_accuracy:.2f}%")

    # Evaluate the official 10,000-image test set after training.
    test_loss, test_accuracy, test_correct, test_total = evaluate(
        model, test_loader, loss_fn, device
    )
    print("\n" + "=" * 66)
    print("FINAL MNIST RESULT")
    print(f"Overall Test Accuracy: {test_accuracy:.2f}%")
    print(f"Correct classifications: {test_correct}/{test_total}")
    print(f"Test Cross-Entropy Loss: {test_loss:.5f}")
    print("=" * 66)

    summary = {
        "seed": SEED,
        "architecture": "784 -> 256 -> 128 -> 10",
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": test_total,
        "batch_size": BATCH_SIZE,
        "epochs": args.epochs,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "test_accuracy_percent": round(test_accuracy, 2),
        "test_correct": test_correct,
        "test_total": test_total,
        "test_loss": test_loss,
        "history": history,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "epoch_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "FINAL_TEST_ACCURACY.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(f"Overall Test Accuracy: {test_accuracy:.2f}%\n"
                f"Correct classifications: {test_correct}/{test_total}\n")
    save_plot(history, "loss", args.output_dir / "training_validation_loss.png")
    save_plot(history, "accuracy", args.output_dir / "training_validation_accuracy.png")
    torch.save(model.state_dict(), args.output_dir / "trained_model.pt")
    print(f"\nSaved evaluation metrics, graphs, and weights to: {args.output_dir}")


if __name__ == "__main__":
    main()
