"""
MLCV Project 1 — dataset.py
Handles transforms, dataset loading, and dataloaders for waste classification.

"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


#step two
def get_transforms(img_size, train=False, rotation=False, light=False,
                   motion=False, texture=False):
    """
    Build a transform pipeline using torchvision Compose.

    Base (always applied): Resize, ToTensor.
    Augmentations (train only, opt-in via flags):
        rotation — flips and rotation
        light    — colour jitter
        motion   — gaussian blur
        texture  — random grayscale

    Note: data is pre-normalised on disk; Normalize() is intentionally omitted.
    Note: RandomErasing and RandomResizedCrop omitted — both require tensor
          input and PIL input respectively; safe composition needs restructuring.
    """
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    augmentations = []

    if train:
        if rotation:
            augmentations += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ]
        if light:
            augmentations += [
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4,
                        saturation=0.4, hue=0.1)
                ], p=0.7),
            ]
        if motion:
            augmentations += [
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.3),
            ]
        if texture:
            augmentations += [
                transforms.RandomGrayscale(p=0.1),
            ]

    return transforms.Compose(base + augmentations)

#step 3 (apply transforms to simulate domain shift)
def get_covariate_transforms(img_size):
    """
    apply non-random stronger transforms to val set 
    TODO
    """

#step 1 
class WasteData(Dataset):
    """
    Loads waste classification images from a directory.
    Expects ImageFolder layout: data/<class_name>/<image_files>
    """

    def __init__(self, data_dir, transform=None):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {data_dir.resolve()}\n"
                "Expected layout: data/<class_name>/<image_files>"
            )
        self.data    = ImageFolder(root=str(data_dir), transform=transform)
        self.classes = self.data.classes   # list[str]
        self.targets = self.data.targets   # list[int]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def analyse(self):
        """Plot and return class distribution histogram."""
        counts  = Counter(self.targets)
        values  = [counts[i] for i in range(len(self.classes))]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(self.classes, values)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

#step 1 (weighted random sampler selects mini batch from approx uniform distribution across classes) 
def get_dataloader(dataset, batch_size, train=False, num_workers=8):
    """
    Returns a DataLoader.

    Training:   WeightedRandomSampler for IID mini-batches,
                correcting class imbalance by sampling inversely
                proportional to class frequency.
    Validation: sequential, no sampler.

    Args:
        dataset     : WasteData instance
        batch_size  : mini-batch size
        train       : True for weighted sampling, False for sequential
        num_workers : parallel data loading workers (match to CPU cores)
    """
    if train:
        labels  = torch.tensor(dataset.targets)
        counts  = torch.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )