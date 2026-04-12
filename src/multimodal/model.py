from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import nn


class MultimodalModel(nn.Module):
    """Encode each modality, fuse, then apply a task head.

    **Data flow**

    - **Batch** ``Dict[str, Any]``: one entry per modality name (e.g. ``image``,
      ``text``). Values are tensors (or nested structures) consumed by the
      matching encoder.
    - **Encoded** ``Dict[str, Tensor]``: one vector per modality; keys match
      ``encoders``.
    - **Fused** ``Tensor``: a single representation passed to the head.

    Fusions that take a ``List[Tensor]`` (e.g. :class:`~multimodal.fusion.common_fusions.ConcatFusion`)
    should be used with
    ``fusion_modality_order`` so the model stacks encoder outputs in a fixed order.
    Fusions that take a dict should leave ``fusion_modality_order`` unset and read
    ``encoded`` directly.

    **Heads**

    Typical task heads take the fused tensor and return logits or a
    ``Dict[str, Tensor]`` for multi-task heads. Contrastive losses that need
    per-modality vectors (e.g. :class:`~multimodal.heads.contrastive.ModalityContrastiveHead`)
    are not applied to ``fused``; compute them on ``encoded`` in your training
    step alongside ``forward``, or compose modules explicitly.
    """

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        fusion: nn.Module,
        head: nn.Module,
        fusion_modality_order: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a multimodal model.

        Args:
            encoders: Encoder modules keyed by modality name. Each ``forward``
                maps one modality batch item to a tensor embedding.
            fusion: If ``fusion_modality_order`` is set, called as
                ``fusion([encoded[k] for k in fusion_modality_order])``. Otherwise
                ``fusion(encoded)`` with the full dict.
            head: Maps fused features to predictions; often a tensor or a dict of
                tensors (e.g. :class:`~multimodal.heads.basic.MultiTaskHead`).
            fusion_modality_order: Modality names in the order list-based fusion
                modules expect (must match tensor layout assumptions such as
                ``in_dim`` for :class:`~multimodal.heads.m3h.M3HHead`).
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.head = head
        self.fusion_modality_order = (
            tuple(fusion_modality_order) if fusion_modality_order is not None else None
        )

    def forward(self, batch: Dict[str, Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        predictions, _ = self.forward_with_encoded(batch)
        return predictions

    def forward_with_encoded(
        self, batch: Dict[str, Any]
    ) -> tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Run encoders and head; also return per-modality embeddings (for contrastive loss, etc.)."""
        encoded: Dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            if name not in batch:
                raise KeyError(f"batch missing modality {name!r} required by encoders")
            encoded[name] = encoder(batch[name])
        if self.fusion_modality_order is not None:
            missing = [k for k in self.fusion_modality_order if k not in encoded]
            if missing:
                raise KeyError(f"encoded missing modalities required for fusion: {missing}")
            fused = self.fusion([encoded[k] for k in self.fusion_modality_order])
        else:  
            fused = self.fusion(encoded)
        predictions = self.head(fused)
        return predictions, encoded
