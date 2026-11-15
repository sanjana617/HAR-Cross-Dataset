
import csv
from pathlib import Path

PARITY_LIMIT = 1.10


def count_parameters(model):
    """Count trainable parameters in a PyTorch model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def check_parameter_parity(model_counts):
    """
    model_counts: dictionary containing model names and parameter counts.
    Example: {"cnn": 75000, "tcn": 78000}
    """
    if len(model_counts) < 2:
        raise ValueError("At least two models are required.")

    if any(count <= 0 for count in model_counts.values()):
        raise ValueError("Parameter counts must be positive.")

    smallest = min(model_counts.values())
    largest = max(model_counts.values())

    ratio = largest / smallest

    return {
        "model_counts": model_counts,
        "smallest": smallest,
        "largest": largest,
        "ratio": ratio,
        "limit": PARITY_LIMIT,
        "passed": ratio <= PARITY_LIMIT,
    }


def save_parity_report(result, output_path="results/raw/model_parity.csv"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "model",
            "parameters",
            "parity_ratio",
            "limit",
            "passed",
        ])

        for model_name, count in result["model_counts"].items():
            writer.writerow([
                model_name,
                count,
                result["ratio"],
                result["limit"],
                result["passed"],
            ])


def print_parity_report(result):
    print("\nMODEL PARAMETER PARITY")
    print("-" * 35)

    for name, count in result["model_counts"].items():
        print(f"{name}: {count:,} parameters")

    print(f"\nMaximum/minimum ratio: {result['ratio']:.4f}")
    print(f"Allowed ratio: {result['limit']:.2f}")

    if result["passed"]:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL - Model widths need adjustment.")
