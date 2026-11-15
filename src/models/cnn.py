"""Lightweight 1D CNN for HAR. Input (B,50,3) -> logits (B,5).

Blocks: Conv1d -> BatchNorm -> ReLU (-> MaxPool except after the last block),
then global average pooling, dropout and a linear head.

Width is solved, not searched (PROTOCOL.md section 2): solve_width picks the base
channel width whose parameter count is closest to the target (75k). Re-solve it for
every tuned kernel size and depth, then run the parity check across models (<= 10%).
"""
import torch
import torch.nn as nn

N_CLASSES = 5
IN_CHANNELS = 3
MULTS = (1, 1, 2, 2)          # per-block channel multipliers; deeper blocks are wider
TARGET_PARAMS = 75_000


class CNN(nn.Module):
    def __init__(self, width=64, kernel=5, n_blocks=3, dropout=0.3):
        super().__init__()
        assert 3 <= n_blocks <= 4 and kernel % 2 == 1
        chans = [width * m for m in MULTS[:n_blocks]]
        layers, c_in = [], IN_CHANNELS
        for i, c_out in enumerate(chans):
            layers += [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False),
                       nn.BatchNorm1d(c_out), nn.ReLU()]
            if i < n_blocks - 1:
                layers.append(nn.MaxPool1d(2))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, N_CLASSES)

    def forward(self, x):                      # x: (B, 50, 3)
        x = x.permute(0, 2, 1)                 # -> (B, 3, 50)
        x = self.features(x).mean(dim=-1)      # global average pooling -> (B, C)
        return self.head(self.drop(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def solve_width(kernel=5, n_blocks=3, target=TARGET_PARAMS, max_width=512):
    """Base width whose parameter count is closest to target."""
    best_w, best_gap = 1, float("inf")
    for w in range(2, max_width + 1):
        n = count_params(CNN(w, kernel, n_blocks, 0.0))
        gap = abs(n - target)
        if gap < best_gap:
            best_w, best_gap = w, gap
        if n >= target:        # parameter count only grows with width, so stop here
            break
    return best_w


def build_cnn(kernel=5, dropout=0.3, n_blocks=3, target=TARGET_PARAMS, width=None):
    """Build a CNN. Width is solved from the budget unless given explicitly."""
    w = width if width is not None else solve_width(kernel, n_blocks, target)
    model = CNN(w, kernel, n_blocks, dropout)
    model.width, model.n_params = w, count_params(model)
    return model