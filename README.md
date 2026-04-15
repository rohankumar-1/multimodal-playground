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

- **`model(batch)`** / **`forward(batch)`** returns **`(predictions, modality_embeddings)`**.
- **`model.predict(batch)`** returns **predictions only** (one full forward pass; embeddings are dropped).
- **`Trainer`** calls **`model(batch)`** and uses both outputs for task losses.
- For optimizers, use **`iter_training_parameters(model, tasks)`** so stateful per-task
  losses (e.g. a critic) are included; **`model.parameters()`** alone can miss those
  weights.

```python
import torch
from torch import nn

from multimodal.fusion import ConcatFusion
from multimodal.heads import MultiTaskLinearHead
from multimodal.model import MultimodalModel
from multimodal.tasks import ClassificationTask
from multimodal.train import Trainer, TrainerConfig, iter_training_parameters


embed_dim = 32
n_sentiment, n_topic = 3, 10  # two classification heads
fused_dim = embed_dim * 2

model = MultimodalModel(
    encoders={
        "vision": nn.Linear(10, embed_dim),
        "text": nn.Linear(8, embed_dim),
    },
    fusion=ConcatFusion(dim=-1),
    head=MultiTaskLinearHead(
        fused_dim,
        {"sentiment": n_sentiment, "topic": n_topic},
    ),
    fusion_modality_order=["vision", "text"],
)

batch = {
    "vision": torch.randn(16, 10),
    "text": torch.randn(16, 8),
    "sentiment_y": torch.randint(0, n_sentiment, (16,)),
    "topic_y": torch.randint(0, n_topic, (16,)),
}

preds, embs = model(batch)
assert preds["sentiment"].shape == (16, n_sentiment)
assert preds["topic"].shape == (16, n_topic)

logits_only = model.predict(batch)  # dict with the same two keys, no embeddings

tasks = [
    ClassificationTask("sentiment", "sentiment_y"),
    ClassificationTask("topic", "topic_y"),
]

optimizer = torch.optim.Adam(iter_training_parameters(model, tasks), lr=1e-3)
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
        "sentiment_y": torch.randint(0, n_sentiment, (8,)),
        "topic_y": torch.randint(0, n_topic, (8,)),
    },
]
trainer.train(train_loader, val_loader=val_loader)
```

For GPU training, set `device="cuda"` and `mixed_precision=True` in `TrainerConfig` (requires a CUDA device).

### Freezing encoders (`TrainerConfig`)

The trainer can freeze encoder weights when it is constructed (after `model.to(device)`):

- **`freeze_all_encoders=True`** — sets `requires_grad=False` on every submodule in `model.encoders`.
- **`freeze_encoder_ids=("vision",)`** — freeze only the listed **encoder tower** keys (must match keys in `model.encoders`). Ignored if `freeze_all_encoders` is True.

```python
from multimodal.train import DDPConfig, TrainerConfig

config = TrainerConfig(
    max_epochs=2,
    grad_accum_steps=1,
    mixed_precision=False,
    device="cpu",
    freeze_encoder_ids=("vision",),  # train `text` encoder + fusion + head
    # freeze_all_encoders=True,  # alternative: freeze every encoder
    # ddp=DDPConfig(backend="nccl", sync_bn=True),  # when using DDP
)
trainer = Trainer(model, tasks, optimizer, config)
```

Optimizers created with `model.parameters()` still work: frozen parameters get no gradient and are not updated. To **exclude** frozen tensors from the optimizer entirely, use `filter(lambda p: p.requires_grad, model.parameters())`.

You can still freeze manually before building the trainer if you prefer not to use these flags.

## Overview

We can abstract any multimodal model into the following components:

1. **Encoders**: each modality is encoded into a feature vector (embedding).
2. **Fusion** (optional): a method to fuse the feature vectors into a single (or multiple) representations.
3. **Heads / decoders**: map fused representation(s) to task-specific outputs.

In this package, each encoder maps a modality tensor to an embedding. **`MultimodalModel.forward`** runs encode → fuse → head and returns **`(predictions, embeddings)`**. **`MultimodalModel.predict`** returns only predictions. List-input fusions use `fusion_modality_order` so modalities are concatenated (or fused) in a fixed order.

Encoders output `(B, latent_dim)` per modality. Fusion yields `(B, fusion_dim)`; the head maps that to task outputs.
