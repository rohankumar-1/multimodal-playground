from __future__ import annotations

import torch
from torch import nn
from multimodal.layers.basic import CrossTaskAttention, MLP
from typing import Dict, List


class M3H(nn.Module):
    """
    Minimal M3H: multiple modalities, multiple tasks
    """
    def __init__(self, modalities, tasks, shared_dim=128):
        """
        modalities: dict of {mod_name: encoder_module}
        tasks: dict of {task_name: (output_dim, task_type)}
        """
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.adapters = nn.ModuleDict({
            k: MLP(getattr(m, "output_dim", 128), shared_dim, hidden_dim=128, num_layers=2)
            for k, m in modalities.items()
        })
        self.tasks = nn.ModuleDict({
            t: MLP(shared_dim, out_dim, hidden_dim=128, num_layers=2)
            for t, (out_dim, task_type) in tasks.items()
        })
        self.shared_dim = shared_dim
        self.num_tasks = len(tasks)
        self.cross_task_attn = CrossTaskAttention(shared_dim, self.num_tasks)

    def forward(self, x_modalities):
        """
        x_modalities: dict of {mod_name: tensor [B, ...]}
        """
        # 1. Encode each modality
        mod_feats = []
        for k, encoder in self.modalities.items():
            feat = encoder(x_modalities[k])  # output [B, feat_dim]
            feat = self.adapters[k](feat)   # project to shared_dim
            mod_feats.append(feat)

        # 2. Fuse modalities (simple sum or mean)
        fused = torch.stack(mod_feats, dim=0).mean(dim=0)  # [B, shared_dim]

        # 3. Prepare for cross-task attention
        # expand for each task: [num_tasks, batch, shared_dim]
        task_inputs = fused.unsqueeze(0).repeat(self.num_tasks, 1, 1)
        task_outputs = self.cross_task_attn(task_inputs)  # [num_tasks, B, shared_dim]

        # 4. Pass through task heads
        outputs = {}
        for i, (t_name, head) in enumerate(self.tasks.items()):
            outputs[t_name] = head(task_outputs[i])

        return outputs