"""Tests for M3H head."""

import torch

from multimodal.fusion import ConcatFusion
from multimodal.heads import M3HHead


def test_m3h_head_output_shape() -> None:
    in_dim = 25 + 20 + 30
    attn_dim = 32
    fused = ConcatFusion(dim=-1)(
        [torch.randn(10, 25), torch.randn(10, 20), torch.randn(10, 30)]
    )
    head = M3HHead(in_dim=in_dim, attn_dim=attn_dim, out_dims={"a": 10, "b": 20, "c": 30}, alpha=1.0)
    out = head(fused)
    assert out["a"].shape == (10, 10)
    assert out["b"].shape == (10, 20)
    assert out["c"].shape == (10, 30)
