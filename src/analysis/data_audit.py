
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis


CLASS_NAMES = {
    0: "walking",
    1: "upstairs",
    2: "downstairs",
    3: "sitting",
    4: "standing",
}

SAMPLING_RATE = 20
WINDOW_SIZE = 50


def audit_dataset(file_path, domain):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with np.load(file_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"]
        subject_id = data["subject_id"]
        meta = json.loads(str(data["meta"]))

    if X.ndim != 3 or X.shape[1:] != (50, 3):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if not (len(X) == len(y) == len(subject_id)):
        raise ValueError("Dataset array lengths do not match.")

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    if not np.isin(y, list(CLASS_NAMES)).all():
        raise ValueError("Dataset contains unexpected labels.")

    # Overall per-axis statistics
    flat = X.reshape(-1, 3)

    axis_mean = np.mean(flat, axis=0)
    axis_variance = np.var(flat, axis=0)
    axis_std = np.std(flat, axis=0)
    axis_skewness = skew(flat, axis=0, bias=False)
    axis_kurtosis = kurtosis(flat, axis=0, bias=False)

    # Signal magnitude
    magnitude = np.linalg.norm(X, axis=2)

    # Signal Magnitude Area: mean absolute acceleration
    # summed across the three axes and averaged over time.
    sma_per_window = np.mean(np.sum(np.abs(X), axis=2), axis=1)

    # Frequency analysis after removing each window's DC component
    centered = X - X.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1))

    frequencies = np.fft.rfftfreq(
        WINDOW_SIZE, d=1 / SAMPLING_RATE
    )

    spectral_power = spectrum ** 2
    spectral_energy = float(np.mean(np.sum(spectral_power, axis=1)))

    # Ignore the zero-frequency component when finding dominant frequency
    mean_spectrum = np.mean(spectral_power, axis=(0, 2))
    dominant_frequency = float(
        frequencies[np.argmax(mean_spectrum[1:]) + 1]
    )

    report = {
        "domain": domain,
        "windows": int(len(X)),
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "subjects": int(len(np.unique(subject_id))),
        "subject_ids": sorted(
            np.unique(subject_id).astype(int).tolist()
        ),
        "subject_window_counts": {
            str(int(s)): int(np.sum(subject_id == s))
            for s in np.unique(subject_id)
        },
        "metadata": meta,
        "amplitude": {
            "min": float(X.min()),
            "max": float(X.max()),
            "mean": float(X.mean()),
            "std": float(X.std()),
        },
        "axis_statistics": {
            "mean": axis_mean.tolist(),
            "variance": axis_variance.tolist(),
            "std": axis_std.tolist(),
            "skewness": axis_skewness.tolist(),
            "kurtosis": axis_kurtosis.tolist(),
        },
        "mean_signal_magnitude_g": float(magnitude.mean()),
        "signal_magnitude_area_g": float(sma_per_window.mean()),
        "spectral_energy": spectral_energy,
        "dominant_frequency_hz": dominant_frequency,
        "class_distribution": {},
        "activity_checks": {},
        "class_statistics": {},
    }

    for class_id, class_name in CLASS_NAMES.items():
        mask = y == class_id
        class_X = X[mask]

        report["class_distribution"][class_name] = int(mask.sum())

        if len(class_X) == 0:
            continue

        class_flat = class_X.reshape(-1, 3)
        class_magnitude = np.linalg.norm(class_X, axis=2)

        class_centered = (
            class_X - class_X.mean(axis=1, keepdims=True)
        )
        class_spectrum = np.abs(
            np.fft.rfft(class_centered, axis=1)
        ) ** 2

        class_energy = float(
            np.mean(np.sum(class_spectrum, axis=1))
        )

        class_mean_spectrum = np.mean(
            class_spectrum, axis=(0, 2)
        )

        class_dominant_frequency = float(
            frequencies[
                np.argmax(class_mean_spectrum[1:]) + 1
            ]
        )

        class_sma = np.mean(
            np.sum(np.abs(class_X), axis=2),
            axis=1
        )

        report["activity_checks"][class_name] = {
            "windows": int(len(class_X)),
            "mean_magnitude_g": float(class_magnitude.mean()),
            "mean_axis_values_g": class_X.mean(
                axis=(0, 1)
            ).tolist(),
        }

        report["class_statistics"][class_name] = {
            "axis_mean": np.mean(class_flat, axis=0).tolist(),
            "axis_variance": np.var(class_flat, axis=0).tolist(),
            "axis_skewness": skew(
                class_flat, axis=0, bias=False
            ).tolist(),
            "axis_kurtosis": kurtosis(
                class_flat, axis=0, bias=False
            ).tolist(),
            "signal_magnitude_area_g": float(class_sma.mean()),
            "spectral_energy": class_energy,
            "dominant_frequency_hz": class_dominant_frequency,
        }

    return report


def write_audit_report(reports, output_path="docs/data_audit.md"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# HAR Dataset Audit\n\n")
        f.write(
            "All statistics below are calculated on raw, "
            "un-normalized acceleration values.\n\n"
        )

        for report in reports:
            f.write(f"## {report['domain'].upper()}\n\n")

            f.write(f"- Windows: {report['windows']}\n")
            f.write(f"- Shape: `{report['shape']}`\n")
            f.write(f"- Data type: {report['dtype']}\n")
            f.write(f"- Subjects: {report['subjects']}\n")
            f.write(
                f"- Subject IDs: `{report['subject_ids']}`\n"
            )
            f.write(
                f"- Amplitude range: "
                f"{report['amplitude']['min']:.4f} to "
                f"{report['amplitude']['max']:.4f} g\n"
            )
            f.write(
                f"- Overall mean: "
                f"{report['amplitude']['mean']:.4f} g\n"
            )
            f.write(
                f"- Overall standard deviation: "
                f"{report['amplitude']['std']:.4f} g\n"
            )
            f.write(
                f"- Mean signal magnitude: "
                f"{report['mean_signal_magnitude_g']:.4f} g\n"
            )
            f.write(
                f"- Signal magnitude area: "
                f"{report['signal_magnitude_area_g']:.4f} g\n"
            )
            f.write(
                f"- Overall spectral energy: "
                f"{report['spectral_energy']:.4f}\n"
            )
            f.write(
                f"- Dominant frequency: "
                f"{report['dominant_frequency_hz']:.2f} Hz\n\n"
            )

            f.write("### Per-axis statistics\n\n")
            f.write(
                "| Axis | Mean | Variance | Std | Skewness | Kurtosis |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|\n")

            stats = report["axis_statistics"]
            for i, axis in enumerate(["X", "Y", "Z"]):
                f.write(
                    f"| {axis} | "
                    f"{stats['mean'][i]:.4f} | "
                    f"{stats['variance'][i]:.4f} | "
                    f"{stats['std'][i]:.4f} | "
                    f"{stats['skewness'][i]:.4f} | "
                    f"{stats['kurtosis'][i]:.4f} |\n"
                )

            f.write("\n### Class distribution\n\n")
            for name, count in report["class_distribution"].items():
                f.write(f"- {name}: {count}\n")

            f.write("\n### Activity checks\n\n")
            for name, values in report["activity_checks"].items():
                f.write(
                    f"- {name}: {values['windows']} windows, "
                    f"mean magnitude "
                    f"{values['mean_magnitude_g']:.4f} g\n"
                )

            f.write("\n### Class-wise signal statistics\n\n")
            for name, values in report["class_statistics"].items():
                f.write(f"**{name}**\n\n")
                f.write(
                    f"- Axis means: `{values['axis_mean']}`\n"
                )
                f.write(
                    f"- Axis variances: `{values['axis_variance']}`\n"
                )
                f.write(
                    f"- Axis skewness: `{values['axis_skewness']}`\n"
                )
                f.write(
                    f"- Axis kurtosis: `{values['axis_kurtosis']}`\n"
                )
                f.write(
                    f"- Signal magnitude area: "
                    f"{values['signal_magnitude_area_g']:.4f} g\n"
                )
                f.write(
                    f"- Spectral energy: "
                    f"{values['spectral_energy']:.4f}\n"
                )
                f.write(
                    f"- Dominant frequency: "
                    f"{values['dominant_frequency_hz']:.2f} Hz\n\n"
                )

            f.write("### Metadata\n\n")
            f.write("```json\n")
            f.write(json.dumps(report["metadata"], indent=2))
            f.write("\n```\n\n---\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Audit HAR datasets."
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed NPZ files."
    )
    parser.add_argument(
        "--output",
        default="docs/data_audit.md",
        help="Output Markdown report path."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    reports = [
        audit_dataset(data_dir / "uci.npz", "uci"),
        audit_dataset(data_dir / "wisdm.npz", "wisdm"),
    ]

    write_audit_report(reports, args.output)
    print(f"Audit report saved to: {args.output}")


if __name__ == "__main__":
    main()
