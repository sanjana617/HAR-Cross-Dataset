
"""
Exploratory correlation between dataset shift and HAR performance degradation.

Expected:
- results/raw/results.csv
- results/analysis/shift_metrics.csv

The headline performance measure is macro-F1 drop:
within-dataset macro-F1 minus cross-dataset macro-F1.

Each observation represents one model and one directed dataset pair,
averaged across the three final-run seeds.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SHIFT_METRICS = [
    "mean_KL",
    "mean_JSD",
    "MMD",
    "proxy_A_distance",
    "class_distribution_JSD",
]


def load_results(path):
    """Load and validate model performance results."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    required = [
        "model", "source", "target", "seed", "split", "macro_f1"
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Results CSV is missing columns: {missing}")

    return df


def load_shift_metrics(path):
    """Load overall shift measurements for directed dataset pairs."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Shift metrics file not found: {path}")

    df = pd.read_csv(path)

    required = ["source", "target", "activity"] + SHIFT_METRICS

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Shift CSV is missing columns: {missing}")

    # Use overall dataset-level measurements, not individual activities.
    df = df[df["activity"] == "all"].copy()

    # Each directed source-target pair should occur only once.
    if df.duplicated(["source", "target"]).any():
        raise ValueError("Duplicate overall shift rows found for a pair.")

    return df[["source", "target"] + SHIFT_METRICS]


def calculate_performance_drops(results):
    """
    Match within and cross-dataset results using:
    model, source, seed and split.

    Drop = within macro-F1 - cross macro-F1.
    """

    within = results[
        results["source"] == results["target"]
    ].copy()

    cross = results[
        results["source"] != results["target"]
    ].copy()

    keys = ["model", "source", "seed", "split"]

    within = within[keys + ["macro_f1"]].rename(
        columns={"macro_f1": "within_macro_f1"}
    )

    cross = cross[
        keys + ["target", "macro_f1"]
    ].rename(
        columns={"macro_f1": "cross_macro_f1"}
    )

    matched = cross.merge(
        within,
        on=keys,
        how="inner",
        validate="many_to_one",
    )

    if matched.empty:
        raise ValueError(
            "No matching within/cross results found. "
            "Check source, seed and split values."
        )

    matched["macro_f1_drop"] = (
        matched["within_macro_f1"] - matched["cross_macro_f1"]
    )

    # Average the three seed-level drops for each model and directed pair.
    observations = (
        matched.groupby(["model", "source", "target"], as_index=False)
        .agg(
            mean_within_macro_f1=("within_macro_f1", "mean"),
            mean_cross_macro_f1=("cross_macro_f1", "mean"),
            macro_f1_drop=("macro_f1_drop", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )

    return observations


def calculate_spearman(observations):
    """Calculate exploratory Spearman correlations."""
    rows = []

    for metric in SHIFT_METRICS:
        valid = observations[
            ["macro_f1_drop", metric]
        ].replace([np.inf, -np.inf], np.nan).dropna()

        if len(valid) < 3:
            rho = np.nan
            p_value = np.nan
        elif (
            valid["macro_f1_drop"].nunique() < 2
            or valid[metric].nunique() < 2
        ):
            rho = np.nan
            p_value = np.nan
        else:
            result = spearmanr(
                valid["macro_f1_drop"],
                valid[metric],
            )
            rho = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append({
            "shift_metric": metric,
            "performance_metric": "macro_f1_drop",
            "spearman_rho": rho,
            "p_value_exploratory": p_value,
            "n_observations": len(valid),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Correlate HAR dataset shift with performance drop"
    )

    parser.add_argument(
        "--results",
        default="results/raw/results.csv",
        help="Path to final model results CSV",
    )

    parser.add_argument(
        "--shift",
        default="results/analysis/shift_metrics.csv",
        help="Path to shift analysis CSV",
    )

    parser.add_argument(
        "--output-dir",
        default="results/analysis",
        help="Directory for correlation output files",
    )

    args = parser.parse_args()

    print("Loading model results...")
    results = load_results(args.results)

    print("Loading shift metrics...")
    shift = load_shift_metrics(args.shift)

    print("Calculating macro-F1 performance drops...")
    observations = calculate_performance_drops(results)

    print("Joining performance drops with shift metrics...")
    observations = observations.merge(
        shift,
        on=["source", "target"],
        how="inner",
        validate="many_to_one",
    )

    if observations.empty:
        raise ValueError(
            "No matching directed dataset pairs between "
            "performance results and shift metrics."
        )

    print("Calculating Spearman correlations...")
    correlations = calculate_spearman(observations)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations_path = output_dir / "correlation_observations.csv"
    correlations_path = output_dir / "spearman_correlations.csv"

    observations.to_csv(observations_path, index=False)
    correlations.to_csv(correlations_path, index=False)

    print("\nCorrelation analysis complete.")
    print(f"Observations generated: {len(observations)}")
    print("\nSpearman correlation results:")
    print(correlations.to_string(index=False))

    print(f"\nSaved observations to: {observations_path}")
    print(f"Saved correlations to: {correlations_path}")

    print(
        "\nNOTE: These correlations are exploratory. "
        "Do not interpret p-values as confirmatory evidence."
    )


if __name__ == "__main__":
    main()
