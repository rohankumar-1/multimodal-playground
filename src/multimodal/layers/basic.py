from __future__ import annotations

import torch
from torch import nn
from typing import Optional
from multimodal.layers.utils import get_activation

class MLP(nn.Module):

    """
    Multi-layer perceptron. Configurable number of layers and hidden dimensions.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2, act: Optional[str] = "relu"):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim), *([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]), nn.Linear(hidden_dim, out_dim)]
        )
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        return x


class CrossTaskAttention(nn.Module):
    """
    Simple cross-task attention: queries=tasks, keys/values=tasks
    """
    def __init__(self, embed_dim, num_tasks, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_tasks = num_tasks

    def forward(self, task_embeddings):
        """
        task_embeddings: [num_tasks, batch, embed_dim]
        returns: [num_tasks, batch, embed_dim]
        """
        attn_out, _ = self.attn(task_embeddings, task_embeddings, task_embeddings)
        return attn_out