from __future__ import annotations

import torch
from torch import nn

from multimodal.layers.basic import MLP
from typing import Dict, List



class MultiTaskHead(nn.Module):
    """ Maps a single fused representation to task outputs. Outputs can be logits, reconstructions, etc. """
    def __init__(self, input_dim: int, tasks: List[str], decoders: Dict[str, nn.Module], **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.tasks = tasks
        self.decoders = decoders

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {task: self.decoders[task](x) for task in self.tasks}


class MultiTaskLinearHead(MultiTaskHead):
    """ Projects a single fused representation to task-specific outputs via linear layers."""

    def __init__(self, input_dim: int, out_dims: Dict[str, int], tasks: List[str], **kwargs) -> None:
        super().__init__(input_dim, tasks, {task: nn.Linear(input_dim, out_dims[task]) for task in self.tasks})


class MultiTaskMLPHead(MultiTaskHead):
    """ Projects a single fused representation to task-specific outputs via MLPs."""

    def __init__(self, input_dim: int, out_dims: Dict[str, int], tasks: List[str], hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.0, **kwargs) -> None:
        super().__init__(input_dim, tasks, {task: MLP(input_dim, out_dims[task], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout) for task in self.tasks})


class MultiTaskSliceHead(nn.Module):
    """Maps a per-task fused tensor ``(B, T, D)`` to task outputs.

    Use with fusion modules that emit one feature vector per task (e.g.
    :class:`~multimodal.fusion.m3h.M3HFusion`), where ``T == len(tasks)`` and
    task ``tasks[t]`` is decoded from ``x[:, t, :]`` of shape ``(B, D)``.

    This differs from :class:`MultiTaskHead`, which applies every decoder to the
    same ``(B, D)`` vector.
    """

    def __init__(self, tasks: List[str], decoders: Dict[str, nn.Module]) -> None:
        super().__init__()
        self.tasks = tasks
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got shape {tuple(x.shape)}")
        tdim = x.size(1)
        if tdim != len(self.tasks):
            raise ValueError(
                f"task dimension T={tdim} must equal len(tasks)={len(self.tasks)}"
            )
        return {task: self.decoders[task](x[:, i, :]) for i, task in enumerate(self.tasks)}


class MultiTaskLinearSliceHead(MultiTaskSliceHead):
    """Linear decoders on each ``(B, D)`` slice of a ``(B, T, D)`` tensor."""

    def __init__(self, feat_dim: int, out_dims: Dict[str, int], tasks: List[str], **kwargs) -> None:
        super().__init__(tasks, {task: nn.Linear(feat_dim, out_dims[task]) for task in tasks})


class MultiTaskMLPSliceHead(MultiTaskSliceHead):
    """MLP decoders on each ``(B, D)`` slice of a ``(B, T, D)`` tensor."""

    def __init__(self, feat_dim: int, out_dims: Dict[str, int], tasks: List[str], hidden_dim: int = 128,
        num_layers: int = 2, dropout: float = 0.0, **kwargs,) -> None:
        super().__init__(tasks, {task: MLP(feat_dim, out_dims[task], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout) for task in tasks})
