
import numpy as np

GRAVITY = 9.80665
WINDOW_SIZE = 50
NUM_CHANNELS = 3


def _validate_input(X):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim not in (2, 3):
        raise ValueError("Input must have shape (50, 3) or (N, 50, 3).")

    if X.ndim == 2:
        X = X[np.newaxis, ...]

    if X.shape[1:] != (WINDOW_SIZE, NUM_CHANNELS):
        raise ValueError("Expected windows with shape (N, 50, 3).")

    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values.")

    return X


def unit_mismatch(X):
    """Convert acceleration values from g to m/s²."""
    X = _validate_input(X)
    return X * GRAVITY


def time_stretch(X, factor):
    """
    Stretch or compress each window in time.

    factor < 1: compress, then edge-pad.
    factor > 1: stretch, then center-crop.
    """
    X = _validate_input(X)

    if factor not in (0.75, 1.25):
        raise ValueError("Supported factors are 0.75 and 1.25.")

    target_length = round(WINDOW_SIZE * factor)
    old_positions = np.linspace(0, 1, WINDOW_SIZE)
    new_positions = np.linspace(0, 1, target_length)

    output = []

    for window in X:
        resized = np.stack([
            np.interp(new_positions, old_positions, window[:, axis])
            for axis in range(NUM_CHANNELS)
        ], axis=1)

        if target_length < WINDOW_SIZE:
            total_padding = WINDOW_SIZE - target_length
            left = total_padding // 2
            right = total_padding - left

            resized = np.pad(
                resized,
                ((left, right), (0, 0)),
                mode="edge"
            )

        elif target_length > WINDOW_SIZE:
            start = (target_length - WINDOW_SIZE) // 2
            resized = resized[start:start + WINDOW_SIZE]

        output.append(resized)

    return np.asarray(output, dtype=np.float32)


def random_rotation(X, seed=0):
    """Apply a reproducible random 3D rotation to each window."""
    X = _validate_input(X)
    rng = np.random.default_rng(seed)

    output = []

    for window in X:
        # Generate a uniformly distributed random rotation.
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)

        w, x, y, z = q

        rotation = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

        output.append(window @ rotation.T)

    return np.asarray(output, dtype=np.float32)


def apply_perturbation(X, ablation, level=None, seed=0):
    """Apply one of the protocol-defined perturbations."""
    if ablation == "unit_mismatch":
        return unit_mismatch(X)

    if ablation == "time_stretch":
        if level is None:
            raise ValueError("Time stretch requires level 0.75 or 1.25.")
        return time_stretch(X, float(level))

    if ablation == "random_rotation":
        return random_rotation(X, seed=seed)

    if ablation == "control":
        return _validate_input(X).copy()

    raise ValueError(f"Unknown perturbation: {ablation}")
