
import itertools
import numpy as np
import torch
from pathlib import Path

from src.train.trainer import train_model


def generate_configs(model_type, seed=0, n_configs=4):
    """Generate reproducible random hyperparameter configurations."""

    learning_rates = [3e-4, 1e-3, 3e-3]
    dropouts = [0.1, 0.3, 0.5]
    weight_decays = [0.0, 1e-4, 1e-3]

    if model_type.lower() in ["cnn", "tcn"]:
        model_options = [3, 5, 7]
        option_name = "kernel_size"
    elif model_type.lower() in ["transformer", "tiny_transformer"]:
        model_options = [2, 4]
        option_name = "num_heads"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    combinations = list(itertools.product(
        learning_rates,
        dropouts,
        weight_decays,
        model_options,
    ))

    rng = np.random.default_rng(seed)

    if n_configs > len(combinations):
        raise ValueError("Requested more configurations than available.")

    selected = rng.choice(
        len(combinations),
        size=n_configs,
        replace=False,
    )

    configs = []

    for index in selected:
        lr, dropout, weight_decay, option = combinations[index]

        configs.append({
            "learning_rate": float(lr),
            "dropout": float(dropout),
            "weight_decay": float(weight_decay),
            option_name: int(option),
        })

    return configs


def tune_model(
    model_type,
    model_factory,
    train_loader,
    val_loader,
    class_weights,
    normalization,
    checkpoint_dir="checkpoints/tuning",
    seed=0,
):
    """
    Tune one model using source validation macro-F1.

    model_factory must accept a configuration dictionary
    and return a new model instance.
    """

    configs = generate_configs(
        model_type=model_type,
        seed=seed,
        n_configs=4,
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trials = []

    for trial_number, config in enumerate(configs, start=1):
        print(f"\nTrial {trial_number}/4: {config}")

        # Same seed for reproducible trial initialization
        torch.manual_seed(seed)
        np.random.seed(seed)

        model_config = {
            key: value
            for key, value in config.items()
            if key in ["dropout", "kernel_size", "num_heads"]
        }

        model = model_factory(model_config)

        checkpoint_path = checkpoint_dir / (
            f"{model_type}_trial_{trial_number}.pt"
        )

        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            normalization=normalization,
            checkpoint_path=checkpoint_path,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            max_epochs=10,
            patience=5,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        trial = {
            "trial": trial_number,
            "config": config,
            "val_macro_f1": result["best_val_macro_f1"],
            "best_epoch": result["best_epoch"],
        }

        trials.append(trial)

        print(
            f"Trial {trial_number} validation macro-F1: "
            f"{trial['val_macro_f1']:.4f}"
        )

    best_trial = max(trials, key=lambda item: item["val_macro_f1"])

    print("\nBest configuration:", best_trial["config"])
    print("Best validation macro-F1:", best_trial["val_macro_f1"])

    return {
        "best_config": best_trial["config"],
        "best_val_macro_f1": best_trial["val_macro_f1"],
        "trials": trials,
    }
