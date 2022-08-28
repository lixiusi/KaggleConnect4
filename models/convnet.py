import torch
from torch import nn


class ConvNet(nn.Module):

    def __init__(self, n_row: int, n_col: int, conv_out_channels: int = 16, hidden_size: int = 128):
        super().__init__()
        self.n_row = n_row
        self.n_col = n_col
        self.conv = nn.Conv2d(in_channels=3, out_channels=conv_out_channels, kernel_size=4)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(12 * conv_out_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_col),
            nn.Softsign()
        )

    def forward(self, x):
        x = x.reshape(-1, self.n_row, self.n_col)
        x = torch.stack([torch.where(x == i, 1., 0.) for i in range(3)], dim=1)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x
