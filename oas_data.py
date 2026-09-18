"""Seeded CIFAR subsets with disjoint training, validation and test indices."""

import numpy as np


def split_indices(length, batch_size, split, seed=42):
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = np.random.default_rng(seed).permutation(length)
    # Train and validation partition the SAME permutation of the training set.
    if split == "train":
        indices = indices[:int(0.8 * length)]
    elif split == "validation":
        indices = indices[int(0.8 * length):]
    if batch_size > len(indices):
        raise ValueError("Requested batch exceeds the chosen split")
    return indices[:batch_size]


def load_cifar_batch(batch_size, img_size, split="train", seed=42, root="./data"):
    # Optional heavyweight dependency: core tests and synthetic runs do not
    # require torchvision or a network connection.
    try:
        from torchvision.datasets import CIFAR10
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install requirements-gui.txt to use CIFAR-10") from exc
    dataset = CIFAR10(root=root, train=(split != "test"), download=True)
    indices = split_indices(len(dataset), batch_size, split, seed)
    images = [np.asarray(dataset[int(i)][0].convert("L").resize(
        (img_size, img_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
        for i in indices]
    return np.stack(images)
