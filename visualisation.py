"""
MLCV Project 1 — visualisation.py
Generates and saves training diagnostic figures and run parameters.
All outputs saved to figures/YYYY-MM-DD_HH-MM-SS/

TODO: adjust for test data (cm using val)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns


def _make_run_dir():
    """Create a timestamped output directory under figures/."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir   = os.path.join("figures", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_params(run_dir, model, optimiser, criterion, scheduler, config):
    """
    Save model architecture and training hyperparameters to a .txt file.

    Args:
        run_dir   : output directory
        model     : CNN instance
        optimiser : torch optimiser
        criterion : loss function
        scheduler : lr scheduler
        config    : dict of hyperparameters from config.py
    """
    path = os.path.join(run_dir, "run_params.txt")
    with open(path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("RUN PARAMETERS\n")
        f.write("=" * 50 + "\n\n")

        f.write("── Hyperparameters ──\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")

        f.write("\n── Model Architecture ──\n")
        f.write(str(model) + "\n")
        f.write(f"\n  Trainable parameters: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

        f.write("\n── Optimiser ──\n")
        f.write(str(optimiser) + "\n")

        f.write("\n── Loss Function ──\n")
        f.write(str(criterion) + "\n")

        f.write("\n── Scheduler ──\n")
        f.write(str(scheduler) + "\n")

    print(f"  Run parameters saved to {path}")


def plot_training_curves(run_dir, train_losses, val_losses,
                         train_accs, val_accs):
    """
    Plot and save training and validation loss and accuracy curves.

    Args:
        run_dir      : output directory
        train_losses : list of train loss per epoch
        val_losses   : list of val loss per epoch
        train_accs   : list of train accuracy per epoch
        val_accs     : list of val accuracy per epoch
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses,   label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train")
    ax2.plot(epochs, [a * 100 for a in val_accs],   label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curves saved to {path}")


def plot_confusion_matrix(run_dir, model, dataloader, classes, device):
    """
    Run inference on the val set and plot a confusion matrix heatmap.

    Args:
        run_dir    : output directory
        model      : trained CNN instance (already in memory)
        dataloader : val DataLoader
        classes    : list of class name strings
        device     : torch.device
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs   = imgs.to(device)
            preds  = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(run_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {path}")


def plot_class_distribution(run_dir, dataset):
    """
    Save class distribution histogram from WasteData.analyse().

    Args:
        run_dir : output directory
        dataset : WasteData instance
    """
    fig  = dataset.analyse()
    path = os.path.join(run_dir, "class_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Class distribution saved to {path}")


def run_all(model, optimiser, criterion, scheduler, config,
            train_losses, val_losses, train_accs, val_accs,
            dl_val, dataset, device):
    """
    Generate all visualisations and save run parameters for a training run.
    Call this once after trainer.fit() completes in main.py.

    Args:
        model        : trained CNN instance
        optimiser    : torch optimiser
        criterion    : loss function
        scheduler    : lr scheduler
        config       : dict of hyperparameters
        train_losses : list of train loss per epoch
        val_losses   : list of val loss per epoch
        train_accs   : list of train accuracy per epoch
        val_accs     : list of val accuracy per epoch
        dl_val       : validation DataLoader
        dataset      : WasteData instance (for class distribution)
        device       : torch.device
    """
    print("\nSaving visualisations...")
    run_dir = _make_run_dir()

    save_run_params(run_dir, model, optimiser, criterion, scheduler, config)
    plot_training_curves(run_dir, train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(run_dir, model, dl_val, dataset.classes, device)
    plot_class_distribution(run_dir, dataset)

    print(f"\nAll outputs saved to {run_dir}/")