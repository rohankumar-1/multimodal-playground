# Multimodal Playground

This package attempts to standardize multimodal learning. It provides a modular and extensible interface between encoders, fusion gates, and task heads, with a consistent API.

## Installation

```bash
pip install -e .
```

Install with dev tools (pytest, ruff):

```bash
pip install -e ".[dev]"
```

Run tests from the repository root:

```bash
pytest
```

If imports fail, ensure the package is installed as above or run `PYTHONPATH=src pytest`.

## Example usage

- **`model(batch)`** / **`model.forward(batch)`** returns **per-modality embeddings** (`dict[str, Tensor]`).
- **`model.predict(batch)`** returns **`(predictions, embeddings)`** (fusion + head, then encoders).
- **`Trainer`** calls **`model.predict(batch)`** internally, so you can pass a **`MultimodalModel`** with no wrapper.

```python
import torch
from torch import nn

from multimodal.fusion import ConcatFusion
from multimodal.heads import MultiTaskLinearHead
from multimodal.model import MultimodalModel
from multimodal.tasks import ClassificationTask
from multimodal.train import Trainer, TrainerConfig


embed_dim = 32
n_classes = 3
fused_dim = embed_dim * 2

model = MultimodalModel(
    encoders={
        "vision": nn.Linear(10, embed_dim),
        "text": nn.Linear(8, embed_dim),
    },
    fusion=ConcatFusion(dim=-1),
    head=MultiTaskLinearHead(fused_dim, {"cls": n_classes}),
    fusion_modality_order=["vision", "text"],
)

batch = {
    "vision": torch.randn(16, 10),
    "text": torch.randn(16, 8),
    "labels": torch.randint(0, n_classes, (16,)),
}

embs = model(batch)  # encoder outputs only
preds, embs = model.predict(batch)  # logits + embeddings

tasks = [ClassificationTask("cls", "labels")]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
config = TrainerConfig(
    max_epochs=2,
    grad_accum_steps=1,
    mixed_precision=False,
    device="cpu",
)
trainer = Trainer(model, tasks, optimizer, config)

train_loader = [batch]
val_loader = [
    {
        "vision": torch.randn(8, 10),
        "text": torch.randn(8, 8),
        "labels": torch.randint(0, n_classes, (8,)),
    },
]
trainer.train(train_loader, val_loader=val_loader)
```

For GPU training, set `device="cuda"` and `mixed_precision=True` in `TrainerConfig` (requires a CUDA device).

## Overview

We can abstract any multimodal model into the following components:

1. **Encoders**: each modality is encoded into a feature vector (embedding).
2. **Fusion** (optional): a method to fuse the feature vectors into a single (or multiple) representations.
3. **Heads / decoders**: map fused representation(s) to task-specific outputs.

In this package, each encoder maps a modality tensor to an embedding. **`MultimodalModel.forward`** returns those embeddings as a dict. **`MultimodalModel.predict`** runs fusion and the head and returns `(predictions, embeddings)`. List-input fusions use `fusion_modality_order` so modalities are concatenated (or fused) in a fixed order.

Encoders output `(B, latent_dim)` per modality. Fusion yields `(B, fusion_dim)`; the head maps that to task outputs.
