
"""
Cross-dataset feature shift analysis for Human Activity Recognition.

For each directed dataset pair, this script computes:
- Mean KL divergence across handcrafted features
- Mean Jensen-Shannon divergence across handcrafted features
- Maximum Mean Discrepancy (RBF kernel)
- Proxy A-distance using a domain classifier
- Class-distribution Jensen-Shannon divergence

Expected input files:
data/processed/uci.npz
data/processed/wisdm.npz
data/processed/pamap2_hand.npz

Each NPZ must contain X, y, and subject_id.
X shape: (number_of_windows, 50, 3)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = {
    0: "walking",
    1: "upstairs",
    2: "downstairs",
    3: "sitting",
    4: "standing",
}

DOMAINS = ["uci", "wisdm", "pamap2_hand"]

RESULT_COLUMNS = [
    "source",
    "target",
    "activity",
    "source_windows",
    "target_windows",
    "mean_KL",
    "mean_JSD",
    "MMD",
    "proxy_A_distance",
    "class_distribution_JSD",
]


def load_dataset(path):
    """Load and validate a processed HAR dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        for key in ["X", "y", "subject_id"]:
            if key not in data.files:
                raise ValueError(f"{path} is missing required array: {key}")

        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64)
        subject_id = data["subject_id"]

    if X.ndim != 3 or X.shape[1:] != (50, 3):
        raise ValueError(f"{path}: X must have shape (N, 50, 3)")

    if len(X) != len(y) or len(y) != len(subject_id):
        raise ValueError(f"{path}: X, y and subject_id lengths do not match")

    if not np.isfinite(X).all():
        raise ValueError(f"{path}: X contains NaN or infinite values")

    if not np.isin(y, list(CLASS_NAMES)).all():
        raise ValueError(f"{path}: unexpected activity labels")

    return X, y


def extract_features(X):
    """
    Extract handcrafted features from each 50-sample,
    three-axis accelerometer window.

    Returns one feature vector per window.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3 or X.shape[1:] != (50, 3):
        raise ValueError("X must have shape (N, 50, 3)")

    if len(X) == 0:
        return np.empty((0, 14), dtype=np.float64)

    features = []

    # Time-domain features for each accelerometer axis.
    mean_values = np.mean(X, axis=1)
    variance_values = np.var(X, axis=1)
    skew_values = skew(X, axis=1, bias=False)
    kurtosis_values = kurtosis(X, axis=1, bias=False)

    features.extend([
        mean_values,
        variance_values,
        skew_values,
        kurtosis_values,
    ])

    # Frequency-domain features.
    fft_values = np.abs(np.fft.rfft(X, axis=1))
    frequencies = np.fft.rfftfreq(50, d=1.0 / 20.0)

    dominant_frequency = frequencies[
        np.argmax(fft_values[:, 1:, :], axis=1) + 1
    ]

    spectral_energy = np.sum(fft_values ** 2, axis=1)

    features.extend([
        dominant_frequency,
        spectral_energy,
    ])

    # Signal magnitude area.
    magnitude = np.sqrt(np.sum(X ** 2, axis=2))
    sma = np.mean(magnitude, axis=1)

    # Mean gravity vector approximation.
    gravity_vector = np.mean(X, axis=1)

    features.extend([
        sma[:, None],
        gravity_vector,
    ])

    result = np.concatenate(features, axis=1)

    # Replace undefined skewness/kurtosis values from constant signals.
    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return result


def histogram_divergences(source, target, bins=20):
    """
    Compute directional KL(source || target) and
    symmetric Jensen-Shannon divergence for each feature.

    Histogram bins are shared between source and target.
    """
    if len(source) < 2 or len(target) < 2:
        return np.nan, np.nan

    kl_values = []
    jsd_values = []

    for column in range(source.shape[1]):
        source_values = source[:, column]
        target_values = target[:, column]

        pooled = np.concatenate([source_values, target_values])

        low = np.min(pooled)
        high = np.max(pooled)

        # A constant feature has no distribution difference.
        if np.isclose(low, high):
            kl_values.append(0.0)
            jsd_values.append(0.0)
            continue

        edges = np.linspace(low, high, bins + 1)

        p, _ = np.histogram(source_values, bins=edges)
        q, _ = np.histogram(target_values, bins=edges)

        # Add smoothing to avoid zero probabilities.
        p = p.astype(np.float64) + 1e-10
        q = q.astype(np.float64) + 1e-10

        p /= p.sum()
        q /= q.sum()

        kl = entropy(p, q)

        midpoint = 0.5 * (p + q)
        jsd = 0.5 * entropy(p, midpoint) + 0.5 * entropy(q, midpoint)

        kl_values.append(kl)
        jsd_values.append(jsd)

    return float(np.mean(kl_values)), float(np.mean(jsd_values))


def calculate_mmd(source, target, max_samples=1000, seed=0):
    """Calculate biased MMD using an RBF kernel."""
    if len(source) < 2 or len(target) < 2:
        return np.nan

    rng = np.random.default_rng(seed)

    if len(source) > max_samples:
        source = source[
            rng.choice(len(source), max_samples, replace=False)
        ]

    if len(target) > max_samples:
        target = target[
            rng.choice(len(target), max_samples, replace=False)
        ]

    # Standardize using the pooled feature distribution.
    combined = np.vstack([source, target])
    scaler = StandardScaler()
    combined = scaler.fit_transform(combined)

    source = combined[:len(source)]
    target = combined[len(source):]

    distances = cdist(combined, combined, metric="sqeuclidean")
    positive_distances = distances[distances > 0]

    if len(positive_distances) == 0:
        return 0.0

    median_distance = np.median(positive_distances)

    if median_distance <= 0:
        return 0.0

    gamma = 1.0 / (2.0 * median_distance)

    K_ss = np.exp(-gamma * cdist(source, source, "sqeuclidean"))
    K_tt = np.exp(-gamma * cdist(target, target, "sqeuclidean"))
    K_st = np.exp(-gamma * cdist(source, target, "sqeuclidean"))

    mmd_squared = (
        np.mean(K_ss)
        + np.mean(K_tt)
        - 2.0 * np.mean(K_st)
    )

    return float(max(mmd_squared, 0.0))


def calculate_proxy_a_distance(source, target, seed=0):
    """
    Estimate proxy A-distance using a logistic-regression
    classifier trained to distinguish source from target.
    """
    if len(source) < 4 or len(target) < 4:
        return np.nan

    X = np.vstack([source, target])

    domain_labels = np.concatenate([
        np.zeros(len(source), dtype=np.int64),
        np.ones(len(target), dtype=np.int64),
    ])

    if len(np.unique(domain_labels)) != 2:
        return np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        domain_labels,
        test_size=0.25,
        random_state=seed,
        stratify=domain_labels,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )

    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    error = 1.0 - accuracy

    # Error is oriented so that A-distance remains non-negative.
    error = min(error, 1.0 - error)

    return float(2.0 * (1.0 - 2.0 * error))


def class_distribution_jsd(source_y, target_y):
    """Calculate JSD between the five activity distributions."""
    source_counts = np.bincount(source_y, minlength=5).astype(float)
    target_counts = np.bincount(target_y, minlength=5).astype(float)

    source_prob = source_counts / source_counts.sum()
    target_prob = target_counts / target_counts.sum()

    midpoint = 0.5 * (source_prob + target_prob)

    return float(
        0.5 * entropy(source_prob, midpoint)
        + 0.5 * entropy(target_prob, midpoint)
    )


def compare_domains(source_X, source_y, target_X, target_y,
                    source_name, target_name):
    """Compare two datasets and return overall and activity-wise rows."""

    rows = []

    source_features = extract_features(source_X)
    target_features = extract_features(target_X)

    def make_row(source_f, target_f, activity, source_labels=None,
                 target_labels=None):

        if len(source_f) < 2 or len(target_f) < 2:
            return None

        mean_kl, mean_jsd = histogram_divergences(
            source_f, target_f
        )

        mmd = calculate_mmd(source_f, target_f)

        proxy_a = calculate_proxy_a_distance(
            source_f, target_f
        )

        if source_labels is not None and target_labels is not None:
            class_jsd = class_distribution_jsd(
                source_labels, target_labels
            )
        else:
            class_jsd = np.nan

        return {
            "source": source_name,
            "target": target_name,
            "activity": activity,
            "source_windows": len(source_f),
            "target_windows": len(target_f),
            "mean_KL": mean_kl,
            "mean_JSD": mean_jsd,
            "MMD": mmd,
            "proxy_A_distance": proxy_a,
            "class_distribution_JSD": class_jsd,
        }

    # Overall dataset-level comparison.
    overall = make_row(
        source_features,
        target_features,
        "all",
        source_y,
        target_y,
    )

    if overall is not None:
        rows.append(overall)

    # Compare matching activities separately.
    for activity_id, activity_name in CLASS_NAMES.items():

        source_mask = source_y == activity_id
        target_mask = target_y == activity_id

        source_activity = source_features[source_mask]
        target_activity = target_features[target_mask]

        activity_row = make_row(
            source_activity,
            target_activity,
            activity_name,
        )

        if activity_row is not None:
            rows.append(activity_row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset HAR feature shift analysis"
    )

    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed NPZ datasets",
    )

    parser.add_argument(
        "--output",
        default="results/analysis/shift_metrics.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    datasets = {}

    # Load every available dataset.
    for domain in DOMAINS:
        dataset_path = data_dir / f"{domain}.npz"

        if dataset_path.exists():
            print(f"Loading {domain}...")
            datasets[domain] = load_dataset(dataset_path)
        else:
            print(f"Skipping {domain}: file not found")

    if len(datasets) < 2:
        raise RuntimeError(
            "At least two processed datasets are required."
        )

    all_rows = []
    names = list(datasets.keys())

    # IMPORTANT: all directed comparisons.
    # A -> B and B -> A are separate comparisons.
    for source_name in names:
        for target_name in names:

            if source_name == target_name:
                continue

            print(f"Comparing {source_name} -> {target_name}")

            source_X, source_y = datasets[source_name]
            target_X, target_y = datasets[target_name]

            pair_rows = compare_domains(
                source_X,
                source_y,
                target_X,
                target_y,
                source_name,
                target_name,
            )

            all_rows.extend(pair_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nShift analysis complete.")
    print(f"Dataset pairs analysed: {len(names) * (len(names) - 1)}")
    print(f"Rows written: {len(all_rows)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
