"""TCN (temporal convolutional network) for 5-class HAR.
Input (B, 50, 3) float32 -> logits (B, 5).  Only the model lives here: no training code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DILATIONS = (1, 2, 4, 8)      # protocol: dilated residual blocks 1, 2, 4, 8
TARGET_PARAMS = 75_000        # protocol: about 75k parameters
N_CLASSES = 5


class Block(nn.Module):
    """Residual block: two dilated causal convs (conv -> batchnorm -> ReLU -> dropout) + skip."""
    def __init__(self, c_in, c_out, k, d, dropout):
        super().__init__()
        self.pad = (k - 1) * d                       # pad the left side only = causal
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=d)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        # 1x1 conv so the skip connection has the same channels as the output
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(F.relu(self.bn1(self.conv1(F.pad(x, (self.pad, 0))))))
        h = self.drop(F.relu(self.bn2(self.conv2(F.pad(h, (self.pad, 0))))))
        return F.relu(h + self.skip(x))


class TCN(nn.Module):
    def __init__(self, width, kernel_size, dropout):
        super().__init__()
        chans = [3] + [width] * len(DILATIONS)
        self.blocks = nn.Sequential(*[
            Block(chans[i], chans[i + 1], kernel_size, d, dropout)
            for i, d in enumerate(DILATIONS)])
        self.head = nn.Linear(width, N_CLASSES)

    def forward(self, x):                  # x: (B, 50, 3)
        x = x.permute(0, 2, 1)             # Conv1d wants (B, channels, time)
        x = self.blocks(x)                 # (B, width, 50)
        x = x.mean(dim=2)                  # global average pooling over time
        return self.head(x)                # (B, 5)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def solve_width(kernel_size, dropout, target=TARGET_PARAMS, tol=0.05):
    """Try widths 8..256 and keep the one whose parameter count is closest to the target."""
    best = min(range(8, 257), key=lambda w: abs(count_params(TCN(w, kernel_size, dropout)) - target))
    n = count_params(TCN(best, kernel_size, dropout))
    assert abs(n - target) / target <= tol, f"width {best} gives {n} params, not within {tol:.0%} of {target}"
    return best


def build_model(cfg):
    """cfg keys: kernel_size (3/5/7), dropout, optional width (solved from the budget if missing)."""
    k = cfg.get("kernel_size", 5)
    p = cfg.get("dropout", 0.1)
    width = cfg.get("width") or solve_width(k, p)
    return TCN(width, k, p)


if __name__ == "__main__":
    for k in (3, 5, 7):
        m = build_model({"kernel_size": k, "dropout": 0.1})
        print(f"kernel {k}: width {m.head.in_features}, params {count_params(m)}")