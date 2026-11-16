
"""Run the fixed cross-dataset HAR experiments."""

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
from src.train.tuner import tune_model
from src.train.results import write_result


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def validate_datasets(domains, base):
    """Validate processed data and all three fixed subject splits."""

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


def build_model(model_name, config):
    """Build a model using its existing model factory."""

    config = dict(config)

    if model_name == "cnn":
        from src.models.cnn import build_cnn

        return build_cnn(
            kernel=config.get("kernel_size", 5),
            dropout=config.get("dropout", 0.3),
            n_blocks=3,
        )

    if model_name == "tcn":
        from src.models.tcn import build_model as build_tcn

        return build_tcn(config)

    if model_name == "tiny_transformer":
        from src.models.tiny_transformer import build_model

        return build_model(config)

    raise ValueError(f"Unknown model: {model_name}")


def count_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def save_tuning_results(model_name, tuning):
    output = ROOT / "results/raw" / f"tuning_{model_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for trial in tuning["trials"]:
        rows.append({
            "trial": trial["trial"],
            **trial["config"],
            "val_macro_f1": trial["val_macro_f1"],
            "best_epoch": trial["best_epoch"],
        })

    columns = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tune_all_models(models, base):
    """Tune each model on UCI split 0 only."""

    print("\nStarting UCI-only hyperparameter tuning...")

    loaders = create_dataloaders(
        data_path=ROOT / "data/processed/uci.npz",
        split_path=ROOT / "splits/uci.json",
        split_index=0,
        batch_size=base["train"]["batch_size"],
    )

    best_configs = {}
    parameter_counts = {}

    for model_name in models:
        print(f"\nTuning {model_name}...")
        set_seed(0)

        tuning = tune_model(
            model_type=model_name,
            model_factory=lambda cfg, name=model_name: (
                build_model(name, cfg)
            ),
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            class_weights=loaders["class_weights"],
            normalization=loaders["normalization"],
            checkpoint_dir=(
                ROOT / "checkpoints/tuning" / model_name
            ),
            seed=0,
        )

        best_configs[model_name] = tuning["best_config"]

        save_tuning_results(model_name, tuning)

        model = build_model(model_name, tuning["best_config"])
        parameter_counts[model_name] = count_params(model)

        print("Best configuration:", tuning["best_config"])
        print("Parameter count:", parameter_counts[model_name])

    ratio = (
        max(parameter_counts.values())
        / min(parameter_counts.values())
    )

    print(f"\nParameter parity ratio: {ratio:.4f}")

    if ratio > 1.10:
        raise RuntimeError(
            "Parameter parity failed. Maximum/minimum ratio "
            "must be <= 1.10."
        )

    output = ROOT / "results/raw/best_configs.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(best_configs, f, indent=2)

    return best_configs, parameter_counts


def make_target_loader(domain, normalization, batch_size, base):
    """Load all target windows using source-training normalization."""

    path = ROOT / base["data"]["processed_file"].format(
        domain=domain
    )

    X, y, _ = load_har_data(path)

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


def existing_result_keys(output_path):
    if not output_path.exists():
        return set()

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)

        return {
            (
                row["model"],
                row["source"],
                row["target"],
                row["seed"],
                row["split"],
            )
            for row in reader
        }


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_final_experiments(
    models,
    domains,
    seeds,
    best_configs,
    parameter_counts,
    base,
    device,
):
    """Train each source model once and evaluate on all targets."""

    batch_size = base["train"]["batch_size"]
    max_epochs = base["train"]["max_epochs"]
    patience = base["train"]["early_stopping_patience"]

    for model_name in models:
        print(f"\n{'=' * 55}")
        print(f"FINAL EXPERIMENTS: {model_name}")
        print(f"{'=' * 55}")

        config = best_configs[model_name]
        n_params = parameter_counts[model_name]

        output_path = ROOT / "results/raw" / "results.csv"
        completed = existing_result_keys(output_path)

        for source in domains:
            for seed in seeds:
                print(
                    f"\nModel={model_name}, "
                    f"Source={source}, Seed={seed}"
                )

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
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
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
                    "git_commit": get_git_commit(),
                }

                with open(
                    checkpoint.parent
                    / f"seed_{seed}_metadata.json",
                    "w",
                ) as f:
                    json.dump(metadata, f, indent=2)

                criterion = nn.CrossEntropyLoss()

                for target in domains:
                    if target == source:
                        target_loader = loaders["test"]
                    else:
                        target_loader = make_target_loader(
                            target,
                            loaders["normalization"],
                            batch_size,
                            base,
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

                    key = (
                        model_name,
                        source,
                        target,
                        str(seed),
                        str(seed),
                    )

                    if key not in completed:
                        write_result(row, output_path)
                        completed.add(key)

                    print(
                        f"{source} -> {target} | "
                        f"Macro-F1: {metrics['macro_f1']:.4f}"
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-dataset HAR experiments."
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
        help="Validate datasets and splits without training.",
    )

    args = parser.parse_args()

    base, experiment = load_configs(args.config)

    domains = get_domains(experiment)
    models = experiment["models"]
    seeds = base["final"]["seeds"]

    print("Domains:", domains)
    print("Models:", models)
    print("Seeds:", seeds)
    print("Device:", args.device)

    validate_datasets(domains, base)

    if args.preflight_only:
        print("\nPreflight checks passed.")
        return

    best_configs, parameter_counts = tune_all_models(
        models,
        base,
    )

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
