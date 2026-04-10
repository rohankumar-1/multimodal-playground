from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

def add_bias(x: torch.Tensor) -> torch.Tensor:
    """Append a bias term (1) to the feature dimension."""
    return torch.cat([torch.ones(x.shape[0], 1), x], dim=1)


class ConcatFusion(nn.Module):
    """Concatenate modality features along feature dimension."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(features, dim=self.dim)


class AdditiveFusion(nn.Module):
    """Sum or mean of modality features."""
    def __init__(self, reduction: str = "sum"):
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(features, dim=-1)  # new modality dim
        if self.reduction == "sum":
            return torch.sum(stacked, dim=-1)
        else:
            return torch.mean(stacked, dim=-1)


class MultiplicativeFusion(nn.Module):
    """Fuse per-modality features into a single representation by multiplying them."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        fused = features[0]
        for x in features[1:]:
            fused = fused * x  # elementwise multiplication
        return fused


class TensorFusion(nn.Module):
    """Exact tensor fusion with optional output projection."""
    def __init__(self, d_out: Optional[int] = None):
        super().__init__()
        self.d_out = d_out
        self.proj: Optional[nn.Linear] = None

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        xs = [add_bias(x) for x in features]  # bias augmentation
        B = xs[0].shape[0]

        m = xs[0]
        for x in xs[1:]:
            fused = torch.einsum('bi,bj->bij', m, x)
            m = fused.reshape(B, -1)

        if self.d_out is not None:
            if self.proj is None:
                self.proj = nn.Linear(m.shape[1], self.d_out)
                nn.init.xavier_uniform_(self.proj.weight)
                if self.proj.bias is not None:
                    nn.init.zeros_(self.proj.bias)
                self.proj = self.proj.to(m.device)
            m = self.proj(m)

        return m



class LowRankTensorFusion(nn.Module):
    """Low-Rank Tensor Fusion (LRTF) with bias augmentation."""
    
    def __init__(self, dims_in: List[int], d_out: int, rank: int = 32):
        super().__init__()
        self.dims_in = dims_in
        self.rank = rank
        self.d_out = d_out

        # per-modality projection: [rank, d_i + 1] to include bias
        self.U = nn.ParameterList([
            nn.Parameter(torch.empty(rank, d + 1)) for d in dims_in
        ])
        for u in self.U:
            nn.init.xavier_uniform_(u)

        # final linear projection
        self.W = nn.Linear(rank, d_out)
        nn.init.xavier_uniform_(self.W.weight)
        if self.W.bias is not None:
            nn.init.zeros_(self.W.bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:

        # bias augmentation
        xs_aug = [add_bias(x) for x in features]

        # per-modality projection
        proj = [x_aug @ u.T for x_aug, u in zip(xs_aug, self.U)]

        # elementwise product across modalities
        fused = proj[0]
        for p in proj[1:]:
            fused = fused * p

        return self.W(fused)  # [B, d_out]


# ----------------------------
# Tests
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    modalities_list = [torch.randn(10, 6), torch.randn(10, 6), torch.randn(10, 6)]

    # ConcatFusion
    concat_fusion = ConcatFusion()
    out_concat = concat_fusion(modalities_list)
    print("ConcatFusion:", out_concat.shape)

    # AdditiveFusion
    add_fusion = AdditiveFusion("sum")
    out_add = add_fusion(modalities_list)
    print("AdditiveFusion:", out_add.shape)

    # MultiplicativeFusion
    mult_fusion = MultiplicativeFusion(dim=1)
    out_mult = mult_fusion(modalities_list)
    print("MultiplicativeFusion:", out_mult.shape)

    # TensorFusion
    tf_fusion = TensorFusion(d_out=8)
    out_tf = tf_fusion(modalities_list)
    print("TensorFusion:", out_tf.shape)

    # LowRankTensorFusion
    dims_in = [x.shape[1] for x in modalities_list]
    lrtf_fusion = LowRankTensorFusion(dims_in=dims_in, d_out=8, rank=16)
    out_lrtf = lrtf_fusion(modalities_list)
    print("LowRankTensorFusion:", out_lrtf.shape)