#!/usr/bin/env python3
"""FactorCL-style **multiview** contrastive learning: two modalities, each with (x, x_aug).

Uses :class:`~multimodal.model.ContrastiveModel` with **routing** so ``image`` and
``image_aug`` share the vision tower, and ``text`` / ``text_aug`` share the text tower.
Typical losses (via multiple :class:`~multimodal.tasks.ContrastiveTask`):

- Unimodal: ``(image, image_aug)``, ``(text, text_aug)`` — symmetric
  :class:`~multimodal.contrastive_losses.InfoNCE` (cosine logits in embedding space).
- Cross-modal: ``(image, text)`` — :class:`~multimodal.contrastive_losses.CriticInfoNCE` with
  :class:`~multimodal.contrastive_losses.SeparableCritic` (learned bilinear similarity; still
  InfoNCE over the batch).

Data: :class:`FashionMNISTImageTextMultiview` in ``fashion_mnist_shared`` (image aug +
random token masking for text).

For **image–text CLIP-style** training without per-modality aug views, see
``clip_style_contrastive.py``.

Run::

    PYTHONPATH=src python examples/multiview_clip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from multimodal.losses import CriticInfoNCE, SeparableCritic
from multimodal.model import ContrastiveModel
from multimodal.tasks import ContrastiveTask
from multimodal.train import Trainer, TrainerConfig, iter_training_parameters

from fashion_mnist_shared import (
    CharTokenizer,
    FashionMNISTImageTextMultiview,
    SmallImageEncoder,
    TextEncoder,
    data_root,
)


def main() -> None:
    torch.manual_seed(0)
    proj_dim = 128
    tokenizer = CharTokenizer(max_len=32)
    vocab_size = 1 + len(tokenizer.ch2i)

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)

    train_ds = FashionMNISTImageTextMultiview(
        root, train=True, tokenizer=tokenizer, download=True
    )
    val_ds = FashionMNISTImageTextMultiview(
        root, train=False, tokenizer=tokenizer, download=True
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = ContrastiveModel(
        encoders={
            "vision": SmallImageEncoder(proj_dim),
            "text": TextEncoder(vocab_size, hidden_dim=64, out_dim=proj_dim),
        },
        route={
            "image": "vision",
            "image_aug": "vision",
            "text": "text",
            "text_aug": "text",
        },
    )

    # Cross-modal alignment uses a separable bilinear critic (still InfoNCE / CLIP-style batch contrastive).
    critic_dim = 256
    cross_loss = CriticInfoNCE(
        SeparableCritic(proj_dim, proj_dim, proj_dim=critic_dim),
        temperature=0.07,
        symmetric=True,
    )
    tasks = [
        ContrastiveTask("uni_image", "image", "image_aug", temperature=0.07, weight=1.0),
        ContrastiveTask("uni_text", "text", "text_aug", temperature=0.07, weight=1.0),
        ContrastiveTask("cross", "image", "text", loss_fn=cross_loss, weight=1.0),
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    optimizer = torch.optim.Adam(iter_training_parameters(model, tasks), lr=3e-4)
    config = TrainerConfig(
        max_epochs=3,
        device=device,
        mixed_precision=use_amp,
        metric_precision=4,
        progress_bar=True,
    )
    trainer = Trainer(model, tasks, optimizer, config)  # ty:ignore[invalid-argument-type]
    print(
        f"Data root: {root} | device={device} | "
        "ContrastiveModel: uni InfoNCE; cross CriticInfoNCE(SeparableCritic)"
    )
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
