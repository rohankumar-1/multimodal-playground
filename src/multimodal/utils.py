from __future__ import annotations

from torch import nn

__all__ = [
    "get_activation",
    "get_norm",
]


def get_norm(norm: str | None = None, num_channels: int = 64, dim: int = 2) -> nn.Module | None:
    """
    norm: string or None
    dim: 1, 2, or 3  (spatial dimensionality)
    """
    if norm is None:
        return None

    norm = norm.lower()

    if norm == "bn":
        return {
            1: nn.BatchNorm1d(num_channels),
            2: nn.BatchNorm2d(num_channels),
            3: nn.BatchNorm3d(num_channels),
        }[dim]

    if norm == "in":
        return {
            1: nn.InstanceNorm1d(num_channels),
            2: nn.InstanceNorm2d(num_channels),
            3: nn.InstanceNorm3d(num_channels),
        }[dim]

    if norm == "gn":
        return nn.GroupNorm(8, num_channels)

    if norm == "ln":
        return nn.LayerNorm(num_channels)

    raise ValueError(f"Unknown normalization: {norm}")


def get_activation(act):
    if act is None:
        return None

    act = act.lower()

    return {
        "relu": nn.ReLU(inplace=True),
        "relu6": nn.ReLU6(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "mish": nn.Mish(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leakyrelu": nn.LeakyReLU(0.1, inplace=True),
    }[act]
