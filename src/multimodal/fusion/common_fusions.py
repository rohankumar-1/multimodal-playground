from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


def add_bias(x: torch.Tensor) -> torch.Tensor:
    """Append a bias term (1) along the feature dimension.

    Args:
        x: Tensor of shape ``(B, D)``.

    Returns:
        Tensor of shape ``(B, D + 1)`` (a column of ones prepended to ``x``).
    """
    return torch.cat([torch.ones(x.shape[0], 1), x], dim=1)


class BaseFusion(nn.Module):
    """Base class for all fusion modules.

    Subclasses implement ``forward(features)`` where ``features`` is a list of
    per-modality tensors sharing the same leading batch dimension(s).
    """
    def __init__(self) -> None:
        super().__init__()
        self.modality_order: list[str] | None = None

    @abstractmethod
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")


class ConcatFusion(BaseFusion):
    """Concatenate modality features along one dimension.

    Args:
        dim: Dimension along which to concatenate (default ``1``, the usual
            feature axis for inputs shaped ``(B, D)``).

    Input / output:
        Given ``M`` tensors each ``(*, D_m, *)`` with identical shape except
        along ``dim``, returns a single tensor ``(*, sum_m D_m, *)`` on that
        axis. Typical case: each ``(B, D_m)`` -> ``(B, sum_m D_m)``.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(features, dim=self.dim)


class AdditiveFusion(BaseFusion):
    """Sum or mean of modality features over a stacked modality axis.

    Args:
        reduction: ``"sum"`` or ``"mean"`` over the modality stack.

    Input / output:
        All tensors must have the same shape ``S``. Internally they are stacked
        to ``(*S, M)`` where ``M`` is the number of modalities, then reduced along
        the last dimension. Returns shape ``S`` (same as each input).
    """
    def __init__(self, reduction: str = "sum"):
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(features, dim=-1)  # new modality dim
        if self.reduction == "sum":
            return torch.sum(stacked, dim=-1)
        else:
            return torch.mean(stacked, dim=-1)


class MultiplicativeFusion(BaseFusion):
    """Fuse modalities by elementwise multiplication (Hadamard product).

    Args:
        dim: Unused; kept for API compatibility with older call sites.

    Input / output:
        All tensors must have the same shape ``S``. Returns ``x_1 * x_2 * ...``
        with shape ``S``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = features[0]
        for x in features[1:]:
            fused = fused * x  # elementwise multiplication
        return fused


class BilinearFusion(BaseFusion):
    """Bilinear map from two modality vectors (``nn.Bilinear``).

    Args:
        x1_dim: Feature size of the first modality.
        x2_dim: Feature size of the second modality.
        out_dim: Output feature size.

    Input / output:
        ``features`` must be exactly two tensors ``x1``, ``x2`` of shapes
        ``(B, x1_dim)`` and ``(B, x2_dim)``. Returns ``(B, out_dim)``.
    """

    def __init__(self, x1_dim: int, x2_dim: int, out_dim: int) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(x1_dim, x2_dim, out_dim)
        nn.init.xavier_uniform_(self.bilinear.weight)
        if self.bilinear.bias is not None:
            nn.init.zeros_(self.bilinear.bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 2:
            raise ValueError(f"BilinearFusion expects 2 features, got {len(features)}")
        x1 = features[0]
        x2 = features[1]
        return self.bilinear(x1, x2)

class TensorFusion(BaseFusion):
    """Full outer-product (tensor) fusion with bias augmentation per modality.

    Each modality is augmented with a leading ``1`` (see :func:`add_bias`), then
    fused by repeated outer product and flattening.

    Args:
        d_out: If set, a ``nn.Linear`` is lazily created from the fused size to
            ``d_out`` on first forward. If ``None``, no projection is applied.

    Input / output:
        Let modalities have sizes ``d_1, ..., d_M`` and batch size ``B``.
        Each input tensor is ``(B, d_m)``. After bias, each is ``(B, d_m + 1)``.
        The fused vector (before projection) has length ``prod_m (d_m + 1)``.
        If ``d_out`` is ``None``, returns ``(B, prod_m (d_m + 1))``; otherwise
        ``(B, d_out)``.
    """
    def __init__(self, d_out: int | None = None):
        super().__init__()
        self.d_out = d_out
        self.proj: nn.Linear | None = None

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
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



class LowRankTensorFusion(BaseFusion):
    """Low-rank tensor fusion (LRTF) with bias augmentation per modality.

    Args:
        dims_in: ``[d_1, ..., d_M]`` feature sizes (before bias) for each modality.
        d_out: Output dimension.
        rank: Rank of the factorized tensor interaction.

    Input / output:
        ``features`` must have length ``M`` with tensors ``(B, d_m)`` matching
        ``dims_in``. Returns ``(B, d_out)``.
    """

    def __init__(self, dims_in: list[int], d_out: int, rank: int = 32):
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

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:

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


