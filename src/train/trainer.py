
import copy
import torch
import torch.nn as nn
from pathlib import Path

from src.train.metrics import calculate_metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(X)
            loss = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(y)
        all_true.extend(y.detach().cpu().numpy())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())

    metrics = calculate_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    normalization,
    checkpoint_path,
    learning_rate=1e-3,
    weight_decay=0.0,
    max_epochs=30,
    patience=5,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(max_epochs):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": best_state,
        "normalization_mean": normalization["mean"],
        "normalization_std": normalization["std"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }, checkpoint_path)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
    }
