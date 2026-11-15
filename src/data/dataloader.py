
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import HARWindowDataset, load_har_data
from src.data.normalization import fit_normalization, apply_normalization


def load_split(split_path, split_index=0):
    with open(split_path, "r") as f:
        splits = json.load(f)

    # Support both a list of splits and a dictionary of splits
    if isinstance(splits, list):
        split = splits[split_index]
    elif "splits" in splits:
        split = splits["splits"][split_index]
    else:
        split = splits[str(split_index)]

    return split


def create_dataloaders(
    data_path,
    split_path,
    split_index=0,
    batch_size=128,
):
    X, y, subject_id = load_har_data(data_path)
    split = load_split(split_path, split_index)

    train_ids = np.asarray(split["train"])
    val_ids = np.asarray(split["val"])
    test_ids = np.asarray(split["test"])

    train_mask = np.isin(subject_id, train_ids)
    val_mask = np.isin(subject_id, val_ids)
    test_mask = np.isin(subject_id, test_ids)

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("One or more splits contain no windows.")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Fit ONLY on source training data
    mean, std = fit_normalization(X_train)

    X_train = apply_normalization(X_train, mean, std)
    X_val = apply_normalization(X_val, mean, std)
    X_test = apply_normalization(X_test, mean, std)

    train_dataset = HARWindowDataset(X_train, y_train)
    val_dataset = HARWindowDataset(X_val, y_val)
    test_dataset = HARWindowDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Class weights calculated from source training labels only
    counts = np.bincount(y_train, minlength=5)
    if np.any(counts == 0):
        raise ValueError("A class is missing from source training data.")

    weights = len(y_train) / (5 * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    normalization = {
        "mean": mean,
        "std": std,
    }

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": class_weights,
        "normalization": normalization,
    }
