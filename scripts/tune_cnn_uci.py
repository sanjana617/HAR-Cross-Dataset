
"""Run the four-trial UCI-only CNN hyperparameter tuning experiment."""

import csv
from pathlib import Path

from src.data.dataloader import create_dataloaders
from src.models.cnn import build_cnn
from src.train.tuner import tune_model


SEED = 0
SPLIT_INDEX = 0

DATA_PATH = Path("data/processed/uci.npz")
SPLIT_PATH = Path("splits/uci.json")
CHECKPOINT_DIR = Path("checkpoints/tuning/cnn_uci")
OUTPUT_PATH = Path("results/raw/tuning_cnn.csv")


def cnn_factory(cfg):
    """Adapt tuner configuration to Shreyaa's CNN interface."""
    return build_cnn(
        kernel=cfg["kernel_size"],
        dropout=cfg["dropout"],
    )


def main():
    # Check required inputs before starting training.
    for path in [DATA_PATH, SPLIT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading UCI data and split 0...")

    data = create_dataloaders(
        data_path=DATA_PATH,
        split_path=SPLIT_PATH,
        split_index=SPLIT_INDEX,
        batch_size=128,
    )

    print("Starting four CNN tuning trials...")

    result = tune_model(
        model_type="cnn",
        model_factory=cnn_factory,
        train_loader=data["train"],
        val_loader=data["val"],
        class_weights=data["class_weights"],
        normalization=data["normalization"],
        checkpoint_dir=CHECKPOINT_DIR,
        seed=SEED,
    )

    best_config = result["best_config"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "model",
        "dataset",
        "split",
        "seed",
        "trial",
        "learning_rate",
        "dropout",
        "weight_decay",
        "kernel_size",
        "val_macro_f1",
        "best_epoch",
        "is_best",
    ]

    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for trial in result["trials"]:
            config = trial["config"]

            writer.writerow({
                "model": "cnn",
                "dataset": "uci",
                "split": SPLIT_INDEX,
                "seed": SEED,
                "trial": trial["trial"],
                "learning_rate": config["learning_rate"],
                "dropout": config["dropout"],
                "weight_decay": config["weight_decay"],
                "kernel_size": config["kernel_size"],
                "val_macro_f1": trial["val_macro_f1"],
                "best_epoch": trial["best_epoch"],
                "is_best": config == best_config,
            })

    print("\nUCI CNN tuning completed.")
    print("Best configuration:", best_config)
    print("Best validation macro-F1:", result["best_val_macro_f1"])
    print("Tuning CSV saved to:", OUTPUT_PATH)
    print("Checkpoints saved to:", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
