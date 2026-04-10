from __future__ import annotations

import torch
from torch import nn
from typing import Optional
from multimodal.layers.utils import get_activation



class MLP(nn.Module):
    """
    Multi-layer perceptron with correct dropout & activations.
    Hidden layers: Linear → Activation → Dropout
    Final layer: Linear only.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2, act: Optional[str] = "relu", dropout: float = 0.0):
        super().__init__()

        assert num_layers >= 1

        activation = get_activation(act)

        layers = []
        dim = in_dim

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            if activation is not None:
                layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



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