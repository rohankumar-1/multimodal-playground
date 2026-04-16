"""Tests for fusion modules (each fusion class in ``common_fusions``)."""

import pytest
import torch

from multimodal.fusion import (
    AdditiveFusion,
    BilinearFusion,
    ConcatFusion,
    LowRankTensorFusion,
    MultiplicativeFusion,
    TensorFusion,
)


def test_concat_fusion_matches_manual_cat() -> None:
    a = torch.randn(4, 3)
    b = torch.randn(4, 5)
    fused = ConcatFusion(dim=-1)([a, b])
    assert fused.shape == (4, 8)
    assert torch.equal(fused, torch.cat([a, b], dim=-1))


def test_concat_fusion_dim1() -> None:
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    fused = ConcatFusion(dim=1)([a, b])
    assert fused.shape == (2, 6)


def test_additive_fusion_sum() -> None:
    x = [torch.ones(5, 4), torch.ones(5, 4) * 2, torch.ones(5, 4) * 3]
    out = AdditiveFusion("sum")(x)
    assert out.shape == (5, 4)
    assert torch.allclose(out, torch.full((5, 4), 6.0))


def test_additive_fusion_mean() -> None:
    x = [torch.ones(5, 4), torch.ones(5, 4) * 3]
    out = AdditiveFusion("mean")(x)
    assert torch.allclose(out, torch.full((5, 4), 2.0))


def test_additive_fusion_invalid_reduction() -> None:
    with pytest.raises(ValueError, match="Invalid reduction"):
        AdditiveFusion("max")


def test_multiplicative_fusion_elementwise_product() -> None:
    a = torch.tensor([[2.0, 3.0]])
    b = torch.tensor([[4.0, 5.0]])
    out = MultiplicativeFusion(dim=1)([a, b])
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[8.0, 15.0]]))


def test_multiplicative_fusion_three_modalities() -> None:
    xs = [torch.randn(4, 6) for _ in range(3)]
    out = MultiplicativeFusion(dim=1)(xs)
    manual = xs[0] * xs[1] * xs[2]
    assert torch.equal(out, manual)


def test_bilinear_fusion_shape() -> None:
    b, d1, d2, out_dim = 8, 12, 16, 7
    x1 = torch.randn(b, d1)
    x2 = torch.randn(b, d2)
    fusion = BilinearFusion(x1_dim=d1, x2_dim=d2, out_dim=out_dim)
    y = fusion([x1, x2])
    assert y.shape == (b, out_dim)


def test_tensor_fusion_two_modalities_no_projection_shape() -> None:
    d0, d1 = 3, 4
    b = 6
    x0 = torch.randn(b, d0)
    x1 = torch.randn(b, d1)
    fused = TensorFusion(d_out=None)([x0, x1])
    expected_dim = (d0 + 1) * (d1 + 1)
    assert fused.shape == (b, expected_dim)


def test_tensor_fusion_with_d_out() -> None:
    modalities = [torch.randn(10, 6), torch.randn(10, 6), torch.randn(10, 6)]
    d_out = 8
    fused = TensorFusion(d_out=d_out)(modalities)
    assert fused.shape == (10, d_out)


def test_tensor_fusion_second_call_reuses_projection() -> None:
    modalities = [torch.randn(4, 2), torch.randn(4, 2)]
    tf = TensorFusion(d_out=3)
    y1 = tf(modalities)
    y2 = tf(modalities)
    assert y1.shape == y2.shape == (4, 3)
    assert tf.proj is not None


def test_low_rank_tensor_fusion_shape() -> None:
    b = 10
    d0, d1, d2 = 6, 6, 6
    modalities = [torch.randn(b, d0), torch.randn(b, d1), torch.randn(b, d2)]
    d_out = 8
    rank = 16
    lrtf = LowRankTensorFusion(dims_in=[d0, d1, d2], d_out=d_out, rank=rank)
    out = lrtf(modalities)
    assert out.shape == (b, d_out)
