
import numpy as np
import torch
from torch.utils.data import Dataset


def load_har_data(file_path):
    """
    Load a processed HAR dataset from an NPZ file.
    """

    data = np.load(file_path, allow_pickle=False)

    required = ["X", "y", "subject_id"]

    for key in required:
        if key not in data.files:
            raise ValueError(f"Missing required array: {key}")

    X = data["X"]
    y = data["y"]
    subject_id = data["subject_id"]

    # Validate data dimensions
    if X.ndim != 3 or X.shape[1:] != (50, 3):
        raise ValueError(f"Expected X shape (N, 50, 3), got {X.shape}")

    if not (len(X) == len(y) == len(subject_id)):
        raise ValueError("X, y and subject_id lengths do not match")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")

    if not np.isin(y, [0, 1, 2, 3, 4]).all():
        raise ValueError("Labels must be between 0 and 4")

    return (
        X.astype(np.float32),
        y.astype(np.int64),
        subject_id
    )


class HARWindowDataset(Dataset):
    """
    PyTorch Dataset for Human Activity Recognition windows.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
