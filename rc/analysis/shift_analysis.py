
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = [
    "walking",
    "upstairs",
    "downstairs",
    "sitting",
    "standing",
]

FEATURE_NAMES = (
    [f"{stat}_{axis}" for stat in ["mean", "variance", "skewness", "kurtosis"]
     for axis in ["x", "y", "z"]]
    + [f"dominant_frequency_{axis}" for axis in ["x", "y", "z"]]
    + [f"spectral_energy_{axis}" for axis in ["x", "y", "z"]]
    + ["signal_magnitude_area"]
    + [f"gravity_vector_{axis}" for axis in ["x", "y", "z"]]
)


def extract_features(X, sampling_rate=20):
    """Extract handcrafted features from (N, 50, 3) HAR windows."""
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3 or X.shape[1:] != (50, 3):
        raise ValueError("X must have shape (N, 50, 3).")

    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values.")

    features = []

    for window in X:
        row = []

        # Statistical features for each axis
        row.extend(np.mean(window, axis=0))
        row.extend(np.var(window, axis=0))
        row.extend(skew(window, axis=0, bias=False))
        row.extend(kurtosis(window, axis=0, bias=False))

        # Frequency-domain features
        centered = window - np.mean(window, axis=0)
        spectrum = np.abs(np.fft.rfft(centered, axis=0))
        frequencies = np.fft.rfftfreq(
            window.shape[0], d=1.0 / sampling_rate
        )

        dominant = frequencies[np.argmax(spectrum[1:], axis=0) + 1]
        energy = np.sum(spectrum ** 2, axis=0) / window.shape[0]

        row.extend(dominant)
        row.extend(energy)

        # Signal magnitude area
        magnitude = np.sqrt(np.sum(window ** 2, axis=1))
        row.append(np.mean(magnitude))

        # Mean gravity vector
        row.extend(np.mean(window, axis=0))

        features.append(row)

    result = np.asarray(features, dtype=np.float64)

    # Replace undefined skewness/kurtosis from constant signals
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def histogram_divergences(source, target, bins=20):
    """Average KL and Jensen-Shannon divergence over features."""
    kl_values = []
    js_values = []

    for i in range(source.shape[1]):
        combined = np.concatenate([source[:, i], target[:, i]])

        edges = np.histogram_bin_edges(combined, bins=bins)

        p = np.histogram(source[:, i], bins=edges)[0].astype(float)
        q = np.histogram(target[:, i], bins=edges)[0].astype(float)

        # Small smoothing value avoids log(0)
        p = (p + 1e-10) / (p.sum() + len(p) * 1e-10)
        q = (q + 1e-10) / (q.sum() + len(q) * 1e-10)

        kl = np.sum(p * np.log(p / q))

        midpoint = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / midpoint))
        js += 0.5 * np.sum(q * np.log(q / midpoint))

        kl_values.append(kl)
        js_values.append(js)

    return float(np.mean(kl_values)), float(np.mean(js_values))


def calculate_mmd(source, target, max_samples=2000, seed=0):
    """Calculate RBF-kernel MMD using the median-distance heuristic."""
    rng = np.random.default_rng(seed)

    if len(source) > max_samples:
        source = source[rng.choice(len(source), max_samples, replace=False)]

    if len(target) > max_samples:
        target = target[rng.choice(len(target), max_samples, replace=False)]

    scaler = StandardScaler()
    combined = scaler.fit_transform(np.vstack([source, target]))

    source = combined[:len(source)]
    target = combined[len(source):]

    # Median heuristic for RBF bandwidth
    sample = combined[:min(len(combined), 500)]
    distances = np.sum(
        (sample[:, None, :] - sample[None, :, :]) ** 2,
        axis=2,
    )
    nonzero = distances[distances > 0]

    if len(nonzero) == 0:
        return 0.0

    bandwidth = np.median(nonzero)

    Kss = np.exp(
        -np.sum((source[:, None] - source[None, :]) ** 2, axis=2)
        / (2 * bandwidth)
    )
    Ktt = np.exp(
        -np.sum((target[:, None] - target[None, :]) ** 2, axis=2)
        / (2 * bandwidth)
    )
    Kst = np.exp(
        -np.sum((source[:, None] - target[None, :]) ** 2, axis=2)
        / (2 * bandwidth)
    )

    return float(Kss.mean() + Ktt.mean() - 2 * Kst.mean())


def calculate_proxy_a_distance(source, target, seed=0):
    """Estimate proxy A-distance using a domain classifier."""
    X = np.vstack([source, target])
    domain_labels = np.concatenate([
        np.zeros(len(source)),
        np.ones(len(target)),
    ])

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

    error = np.mean(classifier.predict(X_test) != y_test)

    # Use the better orientation of the binary classifier
    error = min(error, 1.0 - error)

    return float(2 * (1 - 2 * error))


def class_distribution_jsd(source_y, target_y):
    """Jensen-Shannon divergence between five activity distributions."""
    p = np.bincount(source_y, minlength=5).astype(float)
    q = np.bincount(target_y, minlength=5).astype(float)

    p /= p.sum()
    q /= q.sum()

    midpoint = 0.5 * (p + q)

    def kl_divergence(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return float(
        0.5 * kl_divergence(p, midpoint)
        + 0.5 * kl_divergence(q, midpoint)
    )


def compare_domains(
    X_source,
    y_source,
    X_target,
    y_target,
    source_name="source",
    target_name="target",
    seed=0,
):
    """Compare two domains overall and for each activity."""
    source_features = extract_features(X_source)
    target_features = extract_features(X_target)

    results = []

    groups = [("all", np.ones(len(y_source), dtype=bool),
               np.ones(len(y_target), dtype=bool))]

    for class_id, class_name in enumerate(CLASS_NAMES):
        groups.append((
            class_name,
            y_source == class_id,
            y_target == class_id,
        ))

    for group_name, source_mask, target_mask in groups:
        source = source_features[source_mask]
        target = target_features[target_mask]

        if len(source) < 2 or len(target) < 2:
            continue

        kl, js = histogram_divergences(source, target)

        results.append({
            "source": source_name,
            "target": target_name,
            "activity": group_name,
            "source_windows": len(source),
            "target_windows": len(target),
            "mean_KL": kl,
            "mean_JSD": js,
            "MMD": calculate_mmd(source, target, seed=seed),
            "proxy_A_distance": calculate_proxy_a_distance(
                source, target, seed=seed
            ),
            "class_distribution_JSD": (
                class_distribution_jsd(y_source, y_target)
                if group_name == "all" else ""
            ),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/raw/shift_analysis.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    domains = {}
    for name in ["uci", "wisdm", "pamap2_hand"]:
        path = data_dir / f"{name}.npz"

        if not path.exists():
            print(f"Skipping {name}: dataset not found.")
            continue

        with np.load(path, allow_pickle=False) as data:
            domains[name] = (data["X"], data["y"])

    if len(domains) < 2:
        raise RuntimeError("At least two processed datasets are required.")

    results = []
    names = list(domains)

    for i, source_name in enumerate(names):
        for target_name in names[i + 1:]:
            Xs, ys = domains[source_name]
            Xt, yt = domains[target_name]

            results.extend(compare_domains(
                Xs, ys, Xt, yt,
                source_name, target_name,
            ))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} comparisons to {output}")


if __name__ == "__main__":
    main()
