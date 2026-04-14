from multimodal.heads.basic import (
    MultiTaskHead,
    MultiTaskLinearHead,
    MultiTaskLinearSliceHead,
    MultiTaskMLPHead,
    MultiTaskMLPSliceHead,
    MultiTaskSliceHead,
    NoOpHead,
)
from multimodal.heads.m3h import M3HHead

__all__ = [
    "NoOpHead",
    "MultiTaskHead",
    "MultiTaskLinearHead",
    "MultiTaskLinearSliceHead",
    "MultiTaskMLPHead",
    "MultiTaskMLPSliceHead",
    "MultiTaskSliceHead",
    "M3HHead",
]
