import pytest
import torch
import torch.nn.functional as F

from src.models.cnn import CNN, build_cnn, count_params, solve_width


def test_output_shape():
    m = build_cnn()
    assert m(torch.randn(8, 50, 3)).shape == (8, 5)


@pytest.mark.parametrize("n_blocks", [3, 4])
@pytest.mark.parametrize("kernel", [3, 5, 7])
def test_param_budget(kernel, n_blocks):
    m = build_cnn(kernel=kernel, n_blocks=n_blocks)
    assert 60_000 <= m.n_params <= 90_000
    assert abs(m.n_params - 75_000) / 75_000 < 0.05      # solver gets close to target


def test_solver_is_deterministic_and_counts_match():
    w = solve_width(5, 3)
    assert w == solve_width(5, 3)
    assert count_params(CNN(w, 5, 3, 0.3)) == build_cnn(5, 0.3, 3).n_params


def test_overfits_64_windows():
    """Training loop for this test only. Never used for experiment results."""
    torch.manual_seed(0)
    X = torch.randn(64, 50, 3)
    y = torch.randint(0, 5, (64,))
    m = build_cnn(kernel=5, dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(m(X), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    m.eval()
    with torch.no_grad():
        acc = (m(X).argmax(1) == y).float().mean().item()
    assert loss.item() < 0.2 * first
    assert acc >= 0.95