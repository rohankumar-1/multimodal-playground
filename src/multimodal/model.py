from __future__ import annotations

from typing import Any

import torch
from torch import nn

from multimodal.heads.basic import NoOpHead


class MultimodalModel(nn.Module):
    """Encode modalities, fuse, and apply a task head.

    **Data flow**

    - **Batch** ``Dict[str, Any]``: one entry per modality name. Values are tensors
      consumed by the matching encoder.
    - **Forward** (``model(batch)``) returns ``(predictions, modality_embeddings)``:
      full encode → fuse → head, plus the encoder outputs as a dict.
    - **Predict** returns **only** predictions (same forward pass, embeddings discarded).

    List-based fusions (e.g. :class:`~multimodal.fusion.common_fusions.ConcatFusion`)
    use ``fusion_modality_order`` so modality tensors are stacked in a fixed order.
    Dict-based fusions leave ``fusion_modality_order`` unset.
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module | nn.Linear],
        fusion: nn.Module,
        head: nn.Module | nn.Linear,
        fusion_modality_order: list[str] | None = None,
    ) -> None:
        """Initialize a multimodal model.

        Args:
            encoders: Encoder modules keyed by modality name.
            fusion: If ``fusion_modality_order`` is set, called as
                ``fusion([encoded[k] for k in fusion_modality_order])``. Otherwise
                ``fusion(encoded)`` with the full dict.
            head: Maps fused features to predictions.
            fusion_modality_order: Modality order for list-input fusions.
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.head = head
        self.fusion_modality_order = (
            tuple(fusion_modality_order) if fusion_modality_order is not None else None
        )

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Encode, fuse, apply head. Returns ``(predictions, per_modality_embeddings)``."""
        encoded: dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            if name not in batch:
                raise KeyError(f"batch missing modality {name!r} required by encoders")
            encoded[name] = encoder(batch[name])
        fused = self._fuse(encoded)
        predictions = self.head(fused)
        return predictions, encoded

    def _fuse(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fusion_modality_order is not None:
            missing = [k for k in self.fusion_modality_order if k not in encoded]
            if missing:
                raise KeyError(f"encoded missing modalities required for fusion: {missing}")
            return self.fusion([encoded[k] for k in self.fusion_modality_order])
        return self.fusion(encoded)

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return predictions only (inference-style API; runs one full forward pass)."""
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode the batch. Returns ``per_modality_embeddings``."""
        encoded: dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            if name not in batch:
                raise KeyError(f"batch missing modality {name!r} required by encoders")
            encoded[name] = encoder(batch[name])
        return encoded


class UnimodalModel(nn.Module):
    """Single-modality encoder followed by a head on the embedding.

    **Batch** maps ``input_key`` to a tensor. **Forward** returns
    ``(predictions, {input_key: embedding})`` for compatibility with :class:`Trainer`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module | nn.Linear,
        *,
        input_key: str = "x",
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.encoders = nn.ModuleDict({input_key: encoder})
        self.head = head

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.input_key not in batch:
            raise KeyError(f"batch missing key {self.input_key!r}")
        h = self.encoders[self.input_key](batch[self.input_key])
        predictions = self.head(h)
        return predictions, {self.input_key: h}

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.input_key not in batch:
            raise KeyError(f"batch missing key {self.input_key!r}")
        return {self.input_key: self.encoders[self.input_key](batch[self.input_key])}


def _module_dict_projs(
    modalities: tuple[str, ...],
    projs: dict[str, nn.Module | None] | None,
) -> nn.ModuleDict:
    out: dict[str, nn.Module] = {}
    for m in modalities:
        p = None if projs is None else projs.get(m)
        out[m] = p if p is not None else nn.Identity()
    return nn.ModuleDict(out)


class ContrastiveModel(nn.Module):
    """Routed encoders for contrastive **training** (losses live in tasks, not here).

    **Encoders** are logical towers keyed by ids (e.g. ``"vision"``, ``"text"``). **Route**
    maps each **dataloader / batch key** to an encoder id so multiple batch keys can share
    one tower (e.g. ``image`` and ``image_aug`` both → ``"vision"``).

    Forward returns ``preds={}`` and ``embs`` keyed by **batch keys** from ``route``, each
    ``proj(encoder(batch[key]))``. Use :class:`~multimodal.tasks.ContrastiveTask` (or other
    tasks) on those ``embs`` keys.

    For :class:`~multimodal.train.Trainer` freezing, ``freeze_encoder_modalities`` refers to
    **encoder ids** in ``self.encoders``, not batch keys—freezing ``"vision"`` freezes the
    tower used for every batch key routed to it.

    Also known conceptually as a *routed embedding* stack (no contrastive loss inside this
    module).
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        route: dict[str, str],
        *,
        projs: dict[str, nn.Module | None] | None = None,
    ) -> None:
        super().__init__()
        if not route:
            raise ValueError("ContrastiveModel requires a non-empty route")
        self.encoders = nn.ModuleDict(dict(sorted(encoders.items())))
        self.route = dict(sorted(route.items()))
        enc_ids = tuple(sorted(self.encoders.keys()))
        for batch_key, enc_id in self.route.items():
            if enc_id not in self.encoders:
                raise ValueError(
                    f"route[{batch_key!r}] = {enc_id!r} is not a key in encoders "
                    f"{sorted(self.encoders.keys())}"
                )
        self.proj = _module_dict_projs(enc_ids, projs)

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        embs: dict[str, torch.Tensor] = {}
        for batch_key in self.route:
            enc_id = self.route[batch_key]
            if batch_key not in batch:
                raise KeyError(
                    f"batch missing key {batch_key!r} required by route "
                    f"(encoder {enc_id!r})"
                )
            h = self.encoders[enc_id](batch[batch_key])
            embs[batch_key] = self.proj[enc_id](h)
        predictions: dict[str, torch.Tensor] = {}
        return predictions, embs

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        _, embs = self.forward(batch)
        return embs


class MultiviewContrastiveModel(nn.Module):
    """Multi-encoder contrastive setup: **per modality** an encoder and an (x, x_aug) pair.

    ``encoders`` maps a modality id (e.g. ``"image"``, ``"text"``) to its tower.
    ``view_pairs`` maps the same modality id to ``(batch_key_x, batch_key_x_aug)``.
    The forward pass fills ``embs`` using those **batch key names**, so you can attach
    multiple :class:`~multimodal.tasks.ContrastiveTask` instances, e.g. unimodal
    ``(image, image_aug)``, ``(text, text_aug)``, and cross-modal ``(image, text)``.

    Requires **at least two modalities** (for cross-modal alignment in typical FactorCL-style
    training). ``head`` defaults to :class:`~multimodal.heads.basic.NoOpHead`.

    ``fuse_keys`` selects which ``embs`` keys are concatenated for ``head`` (default: the
    first view of each modality, in sorted modality-id order).
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        view_pairs: dict[str, tuple[str, str]],
        *,
        projs: dict[str, nn.Module | None] | None = None,
        head: nn.Module | None = None,
        fuse_keys: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if len(encoders) < 2:
            raise ValueError(
                "MultiviewContrastiveModel expects at least two modalities "
                "(use UnimodalModel or MultimodalModel for a single tower)."
            )
        enc_keys = set(encoders.keys())
        vp_keys = set(view_pairs.keys())
        if enc_keys != vp_keys:
            raise ValueError(
                f"encoders keys {sorted(enc_keys)} must match view_pairs keys {sorted(vp_keys)}"
            )
        self.view_pairs = {k: (view_pairs[k][0], view_pairs[k][1]) for k in sorted(view_pairs)}
        self.encoders = nn.ModuleDict(dict(sorted(encoders.items())))
        modalities = tuple(sorted(self.view_pairs.keys()))
        self.proj = _module_dict_projs(modalities, projs)
        self.head = head if head is not None else NoOpHead()
        if fuse_keys is None:
            self._fuse_keys = tuple(self.view_pairs[m][0] for m in modalities)
        else:
            self._fuse_keys = tuple(fuse_keys)

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        embs: dict[str, torch.Tensor] = {}
        for m, (ka, kb) in self.view_pairs.items():
            enc = self.encoders[m]
            proj = self.proj[m]
            for key in (ka, kb):
                if key not in batch:
                    raise KeyError(f"batch missing key {key!r} for modality {m!r}")
            embs[ka] = proj(enc(batch[ka]))
            embs[kb] = proj(enc(batch[kb]))

        missing = [k for k in self._fuse_keys if k not in embs]
        if missing:
            raise KeyError(
                f"fuse_keys {self._fuse_keys} not in embs; missing: {missing}. "
                f"Available keys: {sorted(embs.keys())}"
            )
        fused = torch.cat([embs[k] for k in self._fuse_keys], dim=-1)
        predictions = self.head(fused)
        return predictions, embs

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        _, embs = self.forward(batch)
        return embs


class UnifiedContrastiveModel(MultiviewContrastiveModel):
    """Convenience wrapper: two modalities named ``m1`` and ``m2`` (see :class:`MultiviewContrastiveModel`)."""

    def __init__(
        self,
        encoder_m1: nn.Module,
        encoder_m2: nn.Module,
        *,
        proj_m1: nn.Module | None = None,
        proj_m2: nn.Module | None = None,
        head: nn.Module | None = None,
        keys_m1: tuple[str, str] = ("m1", "m1_aug"),
        keys_m2: tuple[str, str] = ("m2", "m2_aug"),
    ) -> None:
        super().__init__(
            encoders={"m1": encoder_m1, "m2": encoder_m2},
            view_pairs={"m1": keys_m1, "m2": keys_m2},
            projs={"m1": proj_m1, "m2": proj_m2},
            head=head,
        )

