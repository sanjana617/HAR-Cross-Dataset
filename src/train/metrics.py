
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate HAR evaluation metrics.

    Returns:
        acc, macro_f1, weighted_f1, kappa
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on empty labels.")

    if len(y_true) != len(y_pred):
        raise ValueError("True and predicted labels must have equal length.")

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2, 3, 4],
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2, 3, 4],
                average="weighted",
                zero_division=0,
            )
        ),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
