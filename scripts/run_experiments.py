"""Run the fixed cross-dataset HAR experiments using saved best configs."""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloader import create_dataloaders, load_split
from src.data.dataset import HARWindowDataset, load_har_data
from src.data.normalization import apply_normalization
from src.train.trainer import train_model, run_epoch
from src.train.results import write_result


# --------------------------------------------------
# Fixed project configuration
# --------------------------------------------------

ALIASES = {
    "uci_har": "uci",
}

EXPECTED = {
    "uci": (30, (21, 4, 5)),
    "wisdm": (36, (25, 5, 6)),
    "pamap2_hand": (8, None),
}

PAMAP2_SPLIT_SIZES = [
    (4, 1, 3),
    (4, 1, 3),
    (5, 1, 2),
]

BEST_CONFIG_PATHS = {
    "cnn": "configs/cnn_best.json",
    "tcn": "configs/tcn.yaml",
    "tiny_transformer": "configs/best/tiny_transformer.yaml",
}


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------
# Configuration loading
# --------------------------------------------------

def load_configs(config_path):
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with open(config_path) as f:
        experiment_config = yaml.safe_load(f)

    base_path = config_path.parent / experiment_config["_base_"]

    with open(base_path) as f:
        base = yaml.safe_load(f)

    return base, experiment_config["experiment"]


def get_domains(experiment):
    return [
        ALIASES.get(domain, domain)
        for domain in experiment["core_domains"]
    ]


def load_best_configs():
    """Load the team's already selected best configurations."""

    configs = {}

    for model_name, relative_path in BEST_CONFIG_PATHS.items():
        path = ROOT / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Missing best configuration: {path}"
            )

        with open(path) as f:
            config = yaml.safe_load(f)

        required = ["learning_rate", "weight_decay"]

        for key in required:
            if key not in config:
                raise ValueError(
                    f"{model_name}: missing '{key}' in {relative_path}"
                )

        configs[model_name] = config

    return configs


# --------------------------------------------------
# Dataset and split validation
# --------------------------------------------------

def validate_datasets(domains, base):
    print("\nChecking datasets and subject splits...")

    for domain in domains:
        data_path = ROOT / base["data"]["processed_file"].format(
            domain=domain
        )
        split_path = ROOT / base["data"]["splits_file"].format(
            domain=domain
        )

        if not data_path.exists():
            raise FileNotFoundError(data_path)

        if not split_path.exists():
            raise FileNotFoundError(split_path)

        _, _, subject_ids = load_har_data(data_path)
        ids = set(int(x) for x in np.unique(subject_ids))

        expected_count, expected_sizes = EXPECTED[domain]

        if len(ids) != expected_count:
            raise ValueError(
                f"{domain}: expected {expected_count} subjects, "
                f"found {len(ids)}."
            )

        test_subjects = []

        for index in range(3):
            split = load_split(split_path, index)

            train = set(int(x) for x in split["train"])
            val = set(int(x) for x in split["val"])
            test = set(int(x) for x in split["test"])

            actual_sizes = (len(train), len(val), len(test))

            if domain == "pamap2_hand":
                expected_fold_sizes = PAMAP2_SPLIT_SIZES[index]
            else:
                expected_fold_sizes = expected_sizes

            if actual_sizes != expected_fold_sizes:
                raise ValueError(
                    f"{domain}, split {index}: expected "
                    f"{expected_fold_sizes}, got {actual_sizes}."
                )

            if train & val or train & test or val & test:
                raise ValueError(
                    f"{domain}: overlapping subjects in split {index}."
                )

            if train | val | test != ids:
                raise ValueError(
                    f"{domain}: split {index} does not cover all subjects."
                )

            test_subjects.extend(test)

        if domain == "pamap2_hand":
            if len(set(test_subjects)) != 8:
                raise ValueError(
                    "PAMAP2 test folds must cover all 8 subjects "
                    "exactly once."
                )

        print(f"{domain}: OK ({len(ids)} subjects)")


# --------------------------------------------------
# Model creation
# --------------------------------------------------

def build_model(model_name, config):
    config = dict(config)

    if model_name == "cnn":
        from src.models.cnn import build_cnn

        return build_cnn(
            kernel=config["kernel_size"],
            dropout=config["dropout"],
            n_blocks=config["n_blocks"],
            width=config["width"],
        )

    if model_name == "tcn":
        from src.models.tcn import build_model as build_tcn

        return build_tcn(config)

    if model_name == "tiny_transformer":
        from src.models.tiny_transformer import build_model

        # The factory solves the width using the 75k parameter budget.
        config["target_params"] = 75000
        return build_model(config)

    raise ValueError(f"Unknown model: {model_name}")


def count_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def validate_best_configs(models, configs):
    print("\nChecking saved model configurations...")

    parameter_counts = {}

    for model_name in models:
        config = configs[model_name]
        model = build_model(model_name, config)

        actual = count_params(model)
        expected = int(config["n_params"])

        if actual != expected:
            raise ValueError(
                f"{model_name}: saved parameter count is {expected}, "
                f"but built model has {actual}."
            )

        parameter_counts[model_name] = actual

        print(
            f"{model_name}: {actual:,} parameters | "
            f"configuration verified"
        )

    ratio = (
        max(parameter_counts.values())
        / min(parameter_counts.values())
    )

    print(f"\nParameter parity ratio: {ratio:.4f}")

    if ratio > 1.10:
        raise RuntimeError(
            "Parameter parity failed: max/min must be <= 1.10."
        )

    print("Parameter parity passed.")

    return parameter_counts


# --------------------------------------------------
# Target dataset loader
# --------------------------------------------------

def make_target_loader(
    domain, normalization, batch_size, base, split_index=0
):
    """Load only target test subjects using source-training normalization."""

    from src.data.dataloader import load_split

    path = ROOT / base["data"]["processed_file"].format(
        domain=domain
    )

    split_path = ROOT / base["data"]["splits_file"].format(
        domain=domain
    )

    X, y, subject_id = load_har_data(path)
    split = load_split(split_path, split_index)

    test_ids = np.asarray(split["test"])
    test_mask = np.isin(subject_id, test_ids)

    if not test_mask.any():
        raise ValueError(
            f"No target test windows found for {domain}, "
            f"split {split_index}."
        )

    X = X[test_mask]
    y = y[test_mask]

    X = apply_normalization(
        X,
        normalization["mean"],
        normalization["std"],
    )

    dataset = HARWindowDataset(X, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


# --------------------------------------------------
# Result handling
# --------------------------------------------------

def existing_result_keys(output_path):
    if not output_path.exists():
        return set()

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)

        return {
            (
                row["source"],
                row["target"],
                row["seed"],
                row["split"],
            )
            for row in reader
        }


def save_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


# --------------------------------------------------
# Final experiments
# --------------------------------------------------

def run_final_experiments(
    models,
    domains,
    seeds,
    best_configs,
    parameter_counts,
    base,
    device,
):
    batch_size = base["train"]["batch_size"]
    max_epochs = base["train"]["max_epochs"]
    patience = base["train"]["early_stopping_patience"]

    for model_name in models:
        print("\n" + "=" * 60)
        print(f"FINAL EXPERIMENTS: {model_name}")
        print("=" * 60)

        config = best_configs[model_name]
        n_params = parameter_counts[model_name]

        # Protocol requires one results file per model.
        output_path = (
            ROOT / "results/raw" / f"{model_name}_results.csv"
        )

        completed = existing_result_keys(output_path)

        for source in domains:
            for seed in seeds:
                key_prefix = str(seed)

                print(
                    f"\nModel={model_name} | "
                    f"Source={source} | Seed={seed}"
                )

                all_results_exist = all(
                    (source, target, str(seed), str(seed)) in completed
                    for target in domains
                )
                if all_results_exist:
                    print("All target results already exist. Skipping.")
                    continue

                set_seed(seed)

                source_data = ROOT / base["data"][
                    "processed_file"
                ].format(domain=source)

                source_split = ROOT / base["data"][
                    "splits_file"
                ].format(domain=source)

                loaders = create_dataloaders(
                    data_path=source_data,
                    split_path=source_split,
                    split_index=seed,
                    batch_size=batch_size,
                )

                model = build_model(model_name, config)

                checkpoint = (
                    ROOT
                    / "checkpoints/final"
                    / model_name
                    / source
                    / f"seed_{seed}.pt"
                )

                trained = train_model(
                    model=model,
                    train_loader=loaders["train"],
                    val_loader=loaders["val"],
                    class_weights=loaders["class_weights"],
                    normalization=loaders["normalization"],
                    checkpoint_path=checkpoint,
                    learning_rate=float(config["learning_rate"]),
                    weight_decay=float(config["weight_decay"]),
                    max_epochs=max_epochs,
                    patience=patience,
                    device=device,
                )

                model = trained["model"]
                model.eval()

                metadata = {
                    "model": model_name,
                    "source": source,
                    "seed": seed,
                    "split": seed,
                    "best_config": config,
                    "n_params": n_params,
                    "best_epoch": trained["best_epoch"],
                    "best_val_macro_f1": trained[
                        "best_val_macro_f1"
                    ],
                    "git_commit": get_git_commit(),
                }

                save_metadata(
                    checkpoint.parent
                    / f"seed_{seed}_metadata.json",
                    metadata,
                )

                criterion = nn.CrossEntropyLoss()

                for target in domains:
                    key = (
                        source,
                        target,
                        str(seed),
                        str(seed),
                    )

                    if key in completed:
                        print(
                            f"{source} -> {target}: "
                            "already recorded, skipping."
                        )
                        continue

                    if target == source:
                        target_loader = loaders["test"]
                    else:
                        target_loader = make_target_loader(
                            target,
                            loaders["normalization"],
                            batch_size,
                            base,
                            split_index=seed,
                        )

                    metrics = run_epoch(
                        model=model,
                        loader=target_loader,
                        criterion=criterion,
                        device=device,
                    )

                    row = {
                        "model": model_name,
                        "source": source,
                        "target": target,
                        "seed": seed,
                        "split": seed,
                        "n_windows": len(target_loader.dataset),
                        "n_params": n_params,
                        "acc": metrics["acc"],
                        "macro_f1": metrics["macro_f1"],
                        "weighted_f1": metrics["weighted_f1"],
                        "kappa": metrics["kappa"],
                    }

                    write_result(row, output_path)
                    completed.add(key)

                    print(
                        f"{source} -> {target} | "
                        f"Macro-F1: {metrics['macro_f1']:.4f}"
                    )

        print(f"\nResults saved: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run fixed cross-dataset HAR experiments."
    )

    parser.add_argument(
        "--config",
        default="configs/exp_cross_dataset.yaml",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate datasets, splits and model configs without training.",
    )

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no GPU is available."
        )

    base, experiment = load_configs(args.config)

    domains = get_domains(experiment)
    models = experiment["models"]
    seeds = base["final"]["seeds"]

    print("Domains:", domains)
    print("Models:", models)
    print("Seeds:", seeds)
    print("Device:", args.device)

    validate_datasets(domains, base)

    best_configs = load_best_configs()
    parameter_counts = validate_best_configs(
        models,
        best_configs,
    )

    if args.preflight_only:
        print("\nPreflight checks passed.")
        return

    print("\nUsing saved best configurations.")
    print("Hyperparameter tuning will NOT be repeated.")

    run_final_experiments(
        models=models,
        domains=domains,
        seeds=seeds,
        best_configs=best_configs,
        parameter_counts=parameter_counts,
        base=base,
        device=args.device,
    )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
