
import torch
import torch.nn as nn

# MLP
class MNISTMLP(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_teacher() -> MNISTMLP:
    return MNISTMLP(hidden_dim=512)


def create_student() -> MNISTMLP:
    return MNISTMLP(hidden_dim=64)
