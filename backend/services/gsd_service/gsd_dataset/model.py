import torch.nn as nn


class GSDNet(nn.Module):
    """65 -> 128 -> 64 -> 1, predicts log(GSD)."""

    def __init__(self, n_in=65, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
