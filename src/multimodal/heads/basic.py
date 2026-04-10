from __future__ import annotations

import torch
from torch import nn

from multimodal.layers.basic import MLP
from typing import Dict, List



class MultiTaskHead(nn.Module):
    """ Maps a fused representation to task outputs. Outputs can be logits, reconstructions, etc. """
    def __init__(self, input_dim: int, tasks: List[str], decoders: Dict[str, nn.Module], **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.tasks = tasks
        self.decoders = decoders

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {task: self.decoders[task](x) for task in self.tasks}


class MultiTaskLinearHead(MultiTaskHead):
    """ Projects a fused representation to task-specific outputs via linear layers."""

    def __init__(self, input_dim: int, out_dims: Dict[str, int], tasks: List[str], **kwargs) -> None:
        super().__init__(input_dim, tasks, {task: nn.Linear(input_dim, out_dims[task]) for task in self.tasks})


class MultiTaskMLPHead(MultiTaskHead):
    """ Projects a fused representation to task-specific outputs via MLPs."""

    def __init__(self, input_dim: int, out_dims: Dict[str, int], tasks: List[str], hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.0, **kwargs) -> None:
        super().__init__(input_dim, tasks, {task: MLP(input_dim, out_dims[task], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout) for task in self.tasks})


