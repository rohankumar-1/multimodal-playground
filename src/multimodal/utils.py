from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def clip_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_j) * 0.5


def dice_loss(logits, targets, eps=1e-6, apply_sigmoid=True):
    if apply_sigmoid:
        probs = torch.sigmoid(logits)
    else:
        probs = logits
    probs = probs.flatten(start_dim=2)
    targets = targets.flatten(start_dim=2).float()
    intersection = (probs * targets).sum(dim=2)
    denominator = probs.sum(dim=2) + targets.sum(dim=2)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()

def dice_bce_loss(logits, targets, bce_weight=0.5, eps=1e-6):
    """ Dice loss with binary cross entropy loss """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dsc = dice_loss(logits, targets, eps=eps)
    return bce_weight * bce + (1 - bce_weight) * dsc

def multiclass_dice_loss(logits, targets, eps=1e-6):
    """
    logits: [B, C, H, W]
    targets: [B, H, W] with integer class IDs
    """
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.shape[1]

    targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

    return dice_loss(
        probs,
        targets_oh,
        eps=eps,
        apply_sigmoid=False,
    )

def get_norm(norm: Optional[str] = None, num_channels: int = 64, dim: int = 2) -> Optional[nn.Module]:
    """
    norm: string or None
    dim: 1, 2, or 3  (spatial dimensionality)
    """
    if norm is None:
        return None

    norm = norm.lower()

    # BatchNorm
    if norm == "bn":
        return {
            1: nn.BatchNorm1d(num_channels),
            2: nn.BatchNorm2d(num_channels),
            3: nn.BatchNorm3d(num_channels),
        }[dim]

    # InstanceNorm
    if norm == "in":
        return {
            1: nn.InstanceNorm1d(num_channels),
            2: nn.InstanceNorm2d(num_channels),
            3: nn.InstanceNorm3d(num_channels),
        }[dim]

    # GroupNorm: dimension-agnostic (always operates over channels)
    if norm == "gn":
        return nn.GroupNorm(8, num_channels)  # default 8 groups

    # LayerNorm: generally used after flattening, but included here
    if norm == "ln":
        return nn.LayerNorm(num_channels)

    raise ValueError(f"Unknown normalization: {norm}")

def get_activation(act):
    if act is None:
        return None

    act = act.lower()

    return {
        "relu": nn.ReLU(inplace=True),
        "relu6": nn.ReLU6(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),     # swish
        "swish": nn.SiLU(),
        "mish": nn.Mish(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leakyrelu": nn.LeakyReLU(0.1, inplace=True),
    }[act]
