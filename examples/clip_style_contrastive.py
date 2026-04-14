#!/usr/bin/env python3
"""Classic **image–text CLIP-style** contrastive training on Fashion-MNIST + label text.

Uses :class:`~multimodal.model.MultimodalModel` with two modality encoders (image, text).
Loss is only :class:`~multimodal.tasks.ContrastiveTask` on ``embs["image"]`` and
``embs["text"]`` (see ``clip_loss`` in ``multimodal.utils``). The fusion + head path still
runs for API compatibility; :class:`~multimodal.heads.NoOpHead` returns no supervised logits.

For **multi-view / unimodal** contrastive (same encoder, two views), see
``factorcl_multiview.py`` and :class:`~multimodal.model.MultiviewContrastiveModel`.

Run::

    PYTHONPATH=src python examples/clip_style_contrastive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from fashion_mnist_shared import (
    CharTokenizer,
    FashionMNISTWithText,
    SmallImageEncoder,
    TextEncoder,
    data_root,
)

from multimodal.fusion import ConcatFusion
from multimodal.heads import NoOpHead
from multimodal.model import MultimodalModel
from multimodal.tasks import ContrastiveTask
from multimodal.train import Trainer, TrainerConfig


def main() -> None:
    torch.manual_seed(0)
    proj_dim = 128
    tokenizer = CharTokenizer(max_len=32)
    vocab_size = 1 + len(tokenizer.ch2i)

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)

    train_ds = FashionMNISTWithText(root, train=True, tokenizer=tokenizer, download=True)
    val_ds = FashionMNISTWithText(root, train=False, tokenizer=tokenizer, download=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = MultimodalModel(
        encoders={
            "image": SmallImageEncoder(proj_dim),
            "text": TextEncoder(vocab_size, hidden_dim=64, out_dim=proj_dim),
        },
        fusion=ConcatFusion(dim=-1),
        head=NoOpHead(),
        fusion_modality_order=["image", "text"],
    )

    tasks = [ContrastiveTask("clip", "image", "text", temperature=0.07)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = TrainerConfig(
        max_epochs=3,
        device=device,
        mixed_precision=use_amp,
        metric_precision=4,
        progress_bar=True,
    )
    trainer = Trainer(model, tasks, optimizer, config)
    print(f"Data root: {root} | device={device} | contrastive on embs['image'] × embs['text']")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
