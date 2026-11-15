
import json
import numpy as np
from pathlib import Path


CLASS_NAMES = {
    0: "walking",
    1: "upstairs",
    2: "downstairs",
    3: "sitting",
    4: "standing",
}


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

    report = {
        "domain": domain,
        "windows": int(len(X)),
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "subjects": int(len(np.unique(subject_id))),
        "subject_ids": sorted(
            np.unique(subject_id).astype(int).tolist()
        ),
        "metadata": meta,
        "amplitude": {
            "min": float(X.min()),
            "max": float(X.max()),
            "mean": float(X.mean()),
            "std": float(X.std()),
        },
        "axis_mean": X.mean(axis=(0, 1)).tolist(),
        "axis_std": X.std(axis=(0, 1)).tolist(),
        "class_distribution": {},
        "activity_checks": {},
        "dominant_frequency_hz": {},
    }

    for class_id, class_name in CLASS_NAMES.items():
        mask = y == class_id
        class_X = X[mask]

        report["class_distribution"][class_name] = int(mask.sum())

        if len(class_X) == 0:
            continue

        magnitude = np.linalg.norm(class_X, axis=2)

        report["activity_checks"][class_name] = {
            "windows": int(len(class_X)),
            "mean_magnitude_g": float(magnitude.mean()),
            "mean_axis_values_g": class_X.mean(
                axis=(0, 1)
            ).tolist(),
        }

        centered = class_X - class_X.mean(axis=1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centered, axis=1))

        frequencies = np.fft.rfftfreq(50, d=1 / 20)
        energy = np.mean(spectrum ** 2, axis=(0, 2))

        report["dominant_frequency_hz"][class_name] = float(
            frequencies[np.argmax(energy[1:]) + 1]
        )

    return report


def write_audit_report(reports, output_path="docs/data_audit.md"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# HAR Dataset Audit\n\n")

        for report in reports:
            f.write(f"## {report['domain'].upper()}\n\n")
            f.write(f"- Windows: {report['windows']}\n")
            f.write(f"- Shape: `{report['shape']}`\n")
            f.write(f"- Data type: {report['dtype']}\n")
            f.write(f"- Subjects: {report['subjects']}\n")
            f.write(f"- Amplitude range: {report['amplitude']['min']:.4f} "
                    f"to {report['amplitude']['max']:.4f} g\n")
            f.write(f"- Overall mean: {report['amplitude']['mean']:.4f} g\n")
            f.write(f"- Overall standard deviation: "
                    f"{report['amplitude']['std']:.4f} g\n\n")

            f.write("### Class distribution\n\n")
            for name, count in report["class_distribution"].items():
                f.write(f"- {name}: {count}\n")

            f.write("\n### Activity checks\n\n")
            for name, values in report["activity_checks"].items():
                f.write(
                    f"- {name}: {values['windows']} windows, "
                    f"mean magnitude {values['mean_magnitude_g']:.4f} g\n"
                )

            f.write("\n### Dominant frequency\n\n")
            for name, frequency in report["dominant_frequency_hz"].items():
                f.write(f"- {name}: {frequency:.2f} Hz\n")

            f.write("\n---\n\n")
