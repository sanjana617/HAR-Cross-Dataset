
import csv
from pathlib import Path


RESULT_COLUMNS = [
    "model",
    "source",
    "target",
    "seed",
    "split",
    "n_windows",
    "n_params",
    "acc",
    "macro_f1",
    "weighted_f1",
    "kappa",
]


def write_result(row, output_path="results/raw/results.csv"):
    """
    Append one experiment result to the master CSV.
    Creates the file and header if it does not exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [key for key in RESULT_COLUMNS if key not in row]
    if missing:
        raise ValueError(f"Missing result fields: {missing}")

    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=RESULT_COLUMNS,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            key: row[key] for key in RESULT_COLUMNS
        })


def write_results(rows, output_path="results/raw/results.csv"):
    """Append multiple experiment results."""
    for row in rows:
        write_result(row, output_path)
