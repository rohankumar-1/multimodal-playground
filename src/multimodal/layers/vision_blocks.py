from __future__ import annotations

import torch
from torch import nn

from multimodal.utils import get_activation, get_norm


class ConvNormAct(nn.Module):
    """
    Unified block for 1D, 2D, and 3D convs.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        dim: int = 2,
        norm: str | None = "bn",
        act: str | None = "relu",
        groups: int = 1,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        # Choose dimensionality of convolution
        conv_layer = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }[dim]

        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=(norm is None)
        )

        self.norm = get_norm(norm, out_channels, dim)
        self.act = get_activation(act)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x



class ResidualBlock(nn.Module):
    """
    Standard 2-layer residual block, works for 1D/2D/3D inputs.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 2,
        stride: int = 1,
        norm: str | None = "bn",
        act: str | None = "relu",
    ) -> None:
        super().__init__()
        self.dim = dim

        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=stride, dim=dim, norm=norm, act=act),
            ConvNormAct(out_channels, out_channels, kernel_size=3, stride=1, dim=dim, norm=norm, act=None),
        )

        # Shortcut if channels or stride don't match
        self.shortcut = (
            ConvNormAct(in_channels, out_channels, kernel_size=1, stride=stride, dim=dim, norm=norm, act=None)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.block(x) + self.shortcut(x))


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block: 1x1 -> 3x3 -> 1x1, dimension-agnostic.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, dim=2, stride=1, norm="bn", act="relu"):
        super().__init__()
        mid_channels = out_channels

        self.block = nn.Sequential(
            ConvNormAct(in_channels, mid_channels, kernel_size=1, stride=1, dim=dim, norm=norm, act=act),
            ConvNormAct(mid_channels, mid_channels, kernel_size=3, stride=stride, dim=dim, norm=norm, act=act),
            ConvNormAct(mid_channels, out_channels * self.expansion, kernel_size=1, stride=1, dim=dim, norm=norm, act=None),
        )

        self.shortcut = (
            ConvNormAct(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, dim=dim, norm=norm, act=None)
            if in_channels != out_channels * self.expansion or stride != 1
            else nn.Identity()
        )

        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))



class SqueezeExciteND(nn.Module):
    def __init__(self, channels, dim=2, reduction=4, act="relu"):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.dim = dim

        if dim == 1:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.conv1 = nn.Conv1d(channels, hidden, 1)
            self.conv2 = nn.Conv1d(hidden, channels, 1)
        elif dim == 2:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.conv1 = nn.Conv2d(channels, hidden, 1)
            self.conv2 = nn.Conv2d(hidden, channels, 1)
        elif dim == 3:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.conv1 = nn.Conv3d(channels, hidden, 1)
            self.conv2 = nn.Conv3d(hidden, channels, 1)
        else:
            raise ValueError(f"Unsupported dim={dim}")

        self.act = get_activation(act)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.conv2(self.act(self.conv1(self.pool(x)))))


class MBConvND(nn.Module):
    """
    MobileNetV2-style inverted residual block, dimension-agnostic.
    """
    def __init__(self, in_channels, out_channels, dim=2, expansion=4, stride=1, kernel_size=3, se=True, norm="bn", act="relu6"):
        super().__init__()
        mid_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.dim = dim

        layers = []

        # expand
        if expansion != 1:
            layers.append(ConvNormAct(in_channels, mid_channels, kernel_size=1, dim=dim, norm=norm, act=act))

        # depthwise
        layers.append(ConvNormAct(
            mid_channels, mid_channels,
            kernel_size=kernel_size, stride=stride,
            dim=dim, groups=mid_channels, norm=norm, act=act
        ))

        # SE
        if se:
            layers.append(SqueezeExciteND(mid_channels, dim=dim, act=act))

        # project
        layers.append(ConvNormAct(mid_channels, out_channels, kernel_size=1, dim=dim, norm=norm, act=None))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


