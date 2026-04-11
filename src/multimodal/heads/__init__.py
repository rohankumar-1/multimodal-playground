from multimodal.heads.basic import MultiTaskHead, MultiTaskLinearHead, MultiTaskMLPHead
from multimodal.heads.contrastive import (
    ModalityContrastiveHead,
    SupervisedContrastiveHead,
)

__all__ = [
    "MultiTaskHead",
    "MultiTaskLinearHead",
    "MultiTaskMLPHead",
    "ModalityContrastiveHead",
    "SupervisedContrastiveHead",
]
