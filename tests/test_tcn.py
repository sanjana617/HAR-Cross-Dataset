"""Run from the folder that holds tcn.py and data/processed/uci.npz:  python3 -m pytest -s test_tcn.py"""
import os
import numpy as np
import torch
try:
    from src.models.tcn import build_model, count_params      # inside the team repo
except ImportError:
    from tcn import build_model, count_params                 # next to the file on your laptop


def test_output_shape():
    m = build_model({"kernel_size": 5, "dropout": 0.1}).eval()
    out = m(torch.randn(8, 50, 3))
    assert out.shape == (8, 5)


def test_params_in_range_for_all_kernels():
    for k in (3, 5, 7):
        n = count_params(build_model({"kernel_size": k, "dropout": 0.1}))
        print(f"kernel {k}: {n} params")
        assert 60_000 <= n <= 90_000
        assert abs(n - 75_000) / 75_000 <= 0.05


def test_overfit_64_windows():
    path = "data/processed/uci.npz"
    if not os.path.exists(path):
        print("uci.npz not found, skipping overfit test")
        return
    d = np.load(path)
    idx = np.random.default_rng(0).choice(len(d["X"]), 64, replace=False)
    X = d["X"][idx]
    y = torch.tensor(d["y"][idx])
    X = torch.tensor((X - X.reshape(-1, 3).mean(0)) / X.reshape(-1, 3).std(0), dtype=torch.float32)
    torch.manual_seed(0)
    m = build_model({"kernel_size": 5, "dropout": 0.0})
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(X), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (m(X).argmax(1) == y).float().mean().item()
    print("overfit accuracy:", acc)
    assert acc > 0.95