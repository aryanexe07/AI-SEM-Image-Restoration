"""Foundational computational primitives for NAFNet architecture.

This module implements the core building blocks used in the NAFNet (Nonlinear Activation
Free Network) model: LayerNorm2d, SimpleGate, and SimplifiedChannelAttention.
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Custom 2D Layer Normalization operating on BCHW image tensors.

    Normalizes features across the channel dimension (dim=1) for each spatial
    position (h, w) independently, followed by a learnable element-wise affine
    transformation.

    Mathematical Definition:
        u = mean(x, dim=1, keepdim=True)
        s = mean((x - u)^2, dim=1, keepdim=True)
        x_hat = (x - u) / sqrt(s + eps)
        output = weight * x_hat + bias

    Args:
        channels (int): Number of input feature channels (C).
        eps (float, optional): Small constant added to variance for numerical stability.
            Defaults to 1e-6.

    Raises:
        ValueError: If `channels` is not positive or `eps` is not positive.

    Tensor Shapes:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)

    Attributes:
        weight (nn.Parameter): Learnable scaling parameter of shape (1, C, 1, 1), initialized to 1.
        bias (nn.Parameter): Learnable shift parameter of shape (1, C, 1, 1), initialized to 0.
        eps (float): Epsilon value for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be a positive integer, got {channels}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.channels = channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LayerNorm2d.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Normalized and affine-transformed tensor of shape (B, C, H, W).

        Raises:
            ValueError: If input tensor is not 4-dimensional or channel count mismatches.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D tensor")
        if x.size(1) != self.channels:
            raise ValueError(
                f"Expected tensor with {self.channels} channels at dim=1, "
                f"got tensor with {x.size(1)} channels"
            )

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x_norm + self.bias


class SimpleGate(nn.Module):
    """Activation-free non-linear interaction via channel splitting and multiplication.

    SimpleGate acts as a parameter-free gating operation that replaces standard non-linear
    activation functions (ReLU, GELU, SiLU). It splits the input tensor along the channel
    dimension (dim=1) into two equal halves X1 and X2, and returns their element-wise product.

    Mathematical Definition:
        X1, X2 = split(X, dim=1)
        Output = X1 * X2

    Tensor Shapes:
        - Input: (B, 2C, H, W)
        - Output: (B, C, H, W)

    Notes:
        Contains ZERO learnable parameters.
        Fully differentiable and TorchScript compatible.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SimpleGate.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 2C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).

        Raises:
            ValueError: If input tensor is not 4D or channel dimension (dim=1) is not even.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, 2C, H, W), got {x.dim()}D tensor")

        channels = x.size(1)
        if channels % 2 != 0:
            raise ValueError(
                f"SimpleGate requires input channel dimension to be even, got {channels} channels"
            )

        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simplified Channel Attention (SCA) module for NAFNet.

    SCA simplifies conventional Squeeze-and-Excitation (SE) channel attention by removing
    channel reduction, non-linear activations (ReLU, Sigmoid), and secondary projection layers.
    It pools global spatial context via Global Average Pooling, passes the pooled vector through
    a single 1x1 convolution with bias, and scales the input feature map element-wise.

    Mathematical Definition:
        v = GlobalAvgPool(X)
        w = Conv1x1(v)
        Output = X * w

    Args:
        channels (int): Number of feature channels (C).

    Raises:
        ValueError: If `channels` is not a positive integer.

    Tensor Shapes:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)

    Attributes:
        channels (int): Number of feature channels.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer outputting (B, C, 1, 1).
        conv (nn.Conv2d): 1x1 convolution layer with bias mapping (B, C, 1, 1) -> (B, C, 1, 1).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be a positive integer, got {channels}")

        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SimplifiedChannelAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Channel-scaled output tensor of shape (B, C, H, W).

        Raises:
            ValueError: If input tensor is not 4D or channel dimension mismatches `channels`.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D tensor")
        if x.size(1) != self.channels:
            raise ValueError(
                f"Expected tensor with {self.channels} channels at dim=1, "
                f"got tensor with {x.size(1)} channels"
            )

        attn = self.conv(self.pool(x))
        return x * attn
