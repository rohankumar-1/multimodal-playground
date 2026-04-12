from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import MLP

from multimodal.utils import get_activation, get_norm


class MultiTaskHead(nn.Module):
    """ Maps a single fused representation to task outputs. Outputs can be logits, reconstructions, etc. """
    def __init__(self, input_dim: int, decoders: dict[str, nn.Module], **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.decoders = decoders

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: decoder(x) for task, decoder in self.decoders.items()}


class MultiTaskLinearHead(MultiTaskHead):
    """ Projects a single fused representation to task-specific outputs via linear layers."""

    def __init__(self, input_dim: int, out_dims: dict[str, int], **kwargs) -> None:
        super().__init__(input_dim,{task: nn.Linear(input_dim, out_dims[task]) for task in out_dims.keys()})


class MultiTaskMLPHead(MultiTaskHead):
    """ Projects a single fused representation to task-specific outputs via MLPs."""

    def __init__(
        self,
        input_dim: int,
        hidden_channels:
        dict[str, list[int]],
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str = "bn",
        ) -> None:
        super().__init__(input_dim,
        decoders={
            task: MLP(
                in_channels=input_dim,
                hidden_channels=hidden_channels[task],
                activation_layer=get_activation(activation),
                dropout=dropout,
                norm_layer=get_norm(norm)
                )
                for task in hidden_channels.keys()
            },
        )


class MultiTaskSliceHead(nn.Module):
    """Maps a per-task fused tensor ``(B, T, D)`` to task outputs.

    Use with fusion modules that emit one feature vector per task (e.g.
    :class:`~multimodal.heads.m3h.M3HHead`), where ``T == len(tasks)`` and
    task ``tasks[t]`` is decoded from ``x[:, t, :]`` of shape ``(B, D)``.

    This differs from :class:`MultiTaskHead`, which applies every decoder to the
    same ``(B, D)`` vector.
    """

    def __init__(self, decoders: dict[str, nn.Module]) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got shape {tuple(x.shape)}")
        tdim = x.size(1)
        if tdim != len(self.decoders):
            raise ValueError(
                f"task dimension T={tdim} must equal len(decoders)={len(self.decoders)}"
            )
        return {task: decoder(x[:, i, :]) for i, (task, decoder) in enumerate(self.decoders.items())}


class MultiTaskLinearSliceHead(MultiTaskSliceHead):
    """Linear decoders on each ``(B, D)`` slice of a ``(B, T, D)`` tensor."""

    def __init__(self, feat_dim: int, out_dims: dict[str, int], **kwargs) -> None:
        super().__init__({task: nn.Linear(feat_dim, out_dims[task]) for task in out_dims.keys()})


class MultiTaskMLPSliceHead(MultiTaskSliceHead):
    """MLP decoders on each ``(B, D)`` slice of a ``(B, T, D)`` tensor."""

    def __init__(
        self,
        feat_dim: int,
        out_dims: dict[str, int],
        hidden_channels: dict[str, list[int]],
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str = "bn",
        ) -> None:
        super().__init__(
            decoders={
                task: MLP(
                    in_channels=feat_dim,
                    hidden_channels=hidden_channels[task],
                    activation_layer=get_activation(activation),
                    dropout=dropout, norm_layer=get_norm(norm)
                )
                for task in hidden_channels.keys()
            },
        )
