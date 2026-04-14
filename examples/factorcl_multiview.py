#!/usr/bin/env python3
"""FactorCL-style **multiview** contrastive learning (single encoder, two views).

Uses :class:`~multimodal.model.MultiviewContrastiveModel`: one :class:`SmallImageEncoder`
maps both ``image`` and ``image_aug`` into a shared embedding space. Loss is
:class:`~multimodal.tasks.ContrastiveTask` between the two view keys (InfoNCE / ``clip_loss``).

Augmentations are light (flip + affine) on the PIL sample; the clean view is
``transforms.ToTensor()`` only.

For **image–text** CLIP-style training, see ``clip_style_contrastive.py``. For the full
**factorized** setup (uni + cross terms), combine :class:`~multimodal.model.UnifiedContrastiveModel`
with multiple ``ContrastiveTask`` instances.

Run::

    PYTHONPATH=src python examples/factorcl_multiview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from multimodal.model import MultiviewContrastiveModel
from multimodal.tasks import ContrastiveTask
from multimodal.train import Trainer, TrainerConfig

from fashion_mnist_shared import FashionMNISTTwoView, SmallImageEncoder, data_root


def main() -> None:
    torch.manual_seed(0)
    proj_dim = 128

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)

    train_ds = FashionMNISTTwoView(root, train=True, download=True)
    val_ds = FashionMNISTTwoView(root, train=False, download=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = MultiviewContrastiveModel(
        SmallImageEncoder(proj_dim),
        view_keys=("image", "image_aug"),
    )

    tasks = [ContrastiveTask("factorcl", "image", "image_aug", temperature=0.07)]
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
    print(
        f"Data root: {root} | device={device} | "
        "MultiviewContrastiveModel: embs['image'] × embs['image_aug']"
    )
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
