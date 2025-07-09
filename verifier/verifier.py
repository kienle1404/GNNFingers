import torch
import torch.nn as nn
import torch.nn.functional as F

class Univerifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, z):
        return F.log_softmax(self.net(z), dim=-1)
