
import numpy as np


def fit_normalization(X_train):
    """
    Calculate per-channel mean and standard deviation
    using source training data only.

    X_train shape: (N, 50, 3)
    """
    X_train = np.asarray(X_train, dtype=np.float32)

    if X_train.ndim != 3 or X_train.shape[1:] != (50, 3):
        raise ValueError(
            f"Expected X_train shape (N, 50, 3), got {X_train.shape}"
        )

    if len(X_train) == 0:
        raise ValueError("Training data cannot be empty.")

    if not np.isfinite(X_train).all():
        raise ValueError("Training data contains NaN or infinite values.")

    # Calculate statistics separately for each accelerometer axis
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)

    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(X, mean, std):
    """
    Apply source-training statistics to any dataset.
    Never calculate new statistics on target data.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (50, 3):
        raise ValueError(
            f"Expected X shape (N, 50, 3), got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError("Input data contains NaN or infinite values.")

    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    if np.any(std <= 0):
        raise ValueError("Standard deviation must be positive.")

    return ((X - mean) / std).astype(np.float32)
