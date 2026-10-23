import torch
import torch.nn as nn


class MNISTMLP(nn.Module):
    """可变深度的 MLP。

    hidden_dims: 每个隐层的宽度,如 [64] 或 [512, 256, 128]。
    dropout:     每个隐层 ReLU 后的 Dropout 概率(0 = 不使用)。
    """

    def __init__(self, hidden_dims, dropout: float = 0.0):
        super().__init__()

        dims = [28 * 28, *hidden_dims, 10]

        layers = [nn.Flatten()]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.network = nn.Sequential(*layers)
        self.hidden_dims = list(hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_teacher(deep: bool = False) -> MNISTMLP:
    """构造 teacher。

    deep=True:  深网络(512-256-128)+ dropout,软目标更丰富;
    deep=False: 单隐层 512,与历史结果同构。
    """
    if deep:
        return MNISTMLP(hidden_dims=[512, 256, 128], dropout=0.3)
    return MNISTMLP(hidden_dims=[512])


def create_student() -> MNISTMLP:
    """构造 student,容量固定为 16,与 deep 模式无关。"""
    return MNISTMLP(hidden_dims=[16])
