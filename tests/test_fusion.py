"""Tests for fusion modules."""

import torch

from multimodal.fusion import ConcatFusion, M3HFusion


def test_concat_fusion_matches_manual_cat() -> None:
    a = torch.randn(4, 3)
    b = torch.randn(4, 5)
    fused = ConcatFusion(dim=-1)([a, b])
    assert fused.shape == (4, 8)
    assert torch.equal(fused, torch.cat([a, b], dim=-1))


def test_m3h_output_shape() -> None:
    in_dim = 25 + 20 + 30
    n_tasks = 3
    attn_dim = 32
    m3h = M3HFusion(
        ConcatFusion(dim=-1),
        in_dim=in_dim,
        n_tasks=n_tasks,
        attn_dim=attn_dim,
        alpha=1.0,
    )
    x = [torch.randn(10, 25), torch.randn(10, 20), torch.randn(10, 30)]
    out = m3h(x)
    assert out.shape == (10, n_tasks, attn_dim)
