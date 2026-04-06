from __future__ import annotations

import torch
from torch import nn

from multimodal.layers.basic import MLP
from typing import Dict, List


class MultiTaskLinearHead(nn.Module):
    """Maps a fused representation to task logits."""

    def __init__(self, input_dim: int, num_classes: Dict[str, int], tasks: List[str], hidden_dim: int = 128, num_layers: int = 2, **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.tasks = tasks
        self.heads = nn.ModuleDict({
            task: MLP(input_dim, num_classes[task], hidden_dim=hidden_dim, num_layers=num_layers) for task in self.tasks
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {task: self.heads[task](x) for task in self.tasks}


