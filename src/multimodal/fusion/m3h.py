""" Implementation of the M3H algorithm from https://arxiv.org/abs/2404.18975 """

from __future__ import annotations

from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from multimodal.fusion.common_fusions import ConcatFusion


class M3HFusion(nn.Module):
    """ Multimodal Multitask Head (M3H) fusion module. """

    def __init__(self, base_fusion: nn.Module = ConcatFusion(dim=-1), in_dim: int = 1, n_tasks: int = 1, attn_dim: int = 1, alpha: float = 1.0):
        """
        Args:
            base_fusion: Base fusion module to use, defaults to ConcatFusion(dim=-1)
            n_tasks: Number of tasks.
            n_features: Number of features.
            alpha: exploration alpha.
        """
        super().__init__()

        self.base_fusion = base_fusion
        self.proj_W = nn.Parameter(torch.randn(n_tasks, attn_dim, in_dim) * (1 / in_dim**0.5)) # project fused representation to task embedding dimension
        self.n_tasks = n_tasks
        self.attn_dim = attn_dim
        self.alpha = alpha

        # WQ, WK, WV ∈ R[nfeature × nfeature]: projection matrices
        self.WQ = nn.Linear(attn_dim, attn_dim, bias=False)
        self.WK = nn.Linear(attn_dim, attn_dim, bias=False)
        self.WV = nn.Linear(attn_dim, attn_dim, bias=False)

        # WT ∈ R[nfeature × ntasks] but implemented as a projection from ntasks → nfeatures
        self.WT = nn.Linear(n_tasks, attn_dim, bias=False)

        # Task token vector Ts = [0, 1, ..., n_tasks-1]
        Ts = torch.arange(n_tasks).float()  # shape = [ntasks]
        self.register_buffer("Ts", Ts)

        # Identity tensor Is ∈ R[nbatch × ntasks × ntasks]
        # (we generate per batch inside forward, but store a base identity)
        self.register_buffer("I_base", torch.eye(n_tasks))


    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        x = self.base_fusion(modalities) # fuse representation into a single tensor
        x = torch.einsum("b i, t o i -> b t o", x, self.proj_W)
        B, T, D = x.shape

        # Step 1: Create task embeddings from Task Tokens Ts: Qs = WT(Ts)
        Ts_onehot = F.one_hot(self.Ts.long(), num_classes=T).float()  # [ntasks, ntasks]
        Qs = self.WT(Ts_onehot)  # [ntasks, nfeatures]
        Qs = self.WQ(Qs)         # [ntasks, nfeatures]
        Qs = Qs.unsqueeze(0).expand(B, T, D) # expand to [batch, ntasks, nfeatures]

        # Step 2: Compute Ks, Vs from x: Ks = WK(x), Vs = WV(x)
        Ks = self.WK(x)   # [batch, ntasks, nfeatures]
        Vs = self.WV(x)   # [batch, ntasks, nfeatures]

        # Step 3: Compute attention matrix: Ms = Qs · Ks^T
        Ms = torch.matmul(Qs, Ks.transpose(1, 2))  # [batch, ntasks, ntasks]

        # Step 4: Normalize and apply self-learning identity bias: Ms_norm = Ms / (Ms_max + 1e-8)
        Ms_max = Ms.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        Ms_norm = Ms / (Ms_max + 1e-8)

        # Identity tensor repeated for batch: Is = I_base.unsqueeze(0).expand(B, T, T)
        Is = self.I_base.unsqueeze(0).expand(B, T, T)

        # Apply attention: logits = Is + α * Ms_norm
        logits = Is + self.alpha * Ms_norm
        Ws = F.softmax(logits, dim=-1)  # [batch, ntasks, ntasks]

        # Step 5: Cross-learned embedding output: Os = Ws · Vs
        Os = torch.matmul(Ws, Vs)  # [batch, ntasks, nfeatures]

        return Os



if __name__ == "__main__":
    import torch

    base_fusion = ConcatFusion(dim=-1)
    m3h = M3HFusion(base_fusion, n_tasks=3, in_dim=75, attn_dim=32, alpha=1.0)
    x = [torch.randn(10, 25), torch.randn(10, 20), torch.randn(10, 30)]
    print(m3h(x).shape)