from __future__ import annotations

from torch import nn


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