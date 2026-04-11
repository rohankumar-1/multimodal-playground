#!/usr/bin/env python3
"""Example: two modalities, M3H fusion producing (B, T, F), slice linear heads per task.

Run from the repo root with the package on the path::

    PYTHONPATH=src python examples/m3h_multitask.py

Or after ``pip install -e ".[dev]"``::

    python examples/m3h_multitask.py
"""

from __future__ import annotations

import torch
from torch import nn

from multimodal.fusion import ConcatFusion, M3HFusion
from multimodal.heads import MultiTaskLinearSliceHead
from multimodal.model import MultimodalModel


def main() -> None:
    torch.manual_seed(0)
    d_v, d_t = 16, 24
    embed_dim = 32
    in_dim = embed_dim * 2
    n_tasks = 2
    attn_dim = 64
    batch_size = 8

    encoders = {
        "vision": nn.Sequential(nn.Linear(d_v, embed_dim), nn.ReLU()),
        "text": nn.Sequential(nn.Linear(d_t, embed_dim), nn.ReLU()),
    }
    m3h = M3HFusion(
        ConcatFusion(dim=-1),
        in_dim=in_dim,
        n_tasks=n_tasks,
        attn_dim=attn_dim,
        alpha=1.0,
    )
    head = MultiTaskLinearSliceHead(
        feat_dim=attn_dim,
        out_dims={"sentiment": 3, "topic": 10},
        tasks=["sentiment", "topic"],
    )
    model = MultimodalModel(
        encoders,
        m3h,
        head,
        fusion_modality_order=("vision", "text"),
    )

    batch = {
        "vision": torch.randn(batch_size, d_v),
        "text": torch.randn(batch_size, d_t),
    }
    out = model(batch)
    print("sentiment logits:", out["sentiment"].shape)
    print("topic logits:", out["topic"].shape)


if __name__ == "__main__":
    main()
