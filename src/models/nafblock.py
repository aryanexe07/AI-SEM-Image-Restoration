"""NAFBlock computational block for NAFNet architecture.

This module implements the atomic residual unit of NAFNet, combining LayerNorm2d,
SimpleGate, SimplifiedChannelAttention, and depthwise convolutions with learnable
residual scaling parameters (beta and gamma).
"""

import torch
import torch.nn as nn

from src.models.blocks import LayerNorm2d, SimpleGate, SimplifiedChannelAttention


class NAFBlock(nn.Module):
    """Core computational NAFBlock unit for NAFNet image restoration.

    NAFBlock consists of two sequential residual sub-blocks:
    1. Spatial & Channel Mixer: LayerNorm2d -> 1x1 Conv (Expansion) -> 3x3 DWConv ->
       SimpleGate -> SimplifiedChannelAttention -> 1x1 Conv (Projection) -> Dropout ->
       Beta Scaling -> Add Skip
    2. Feed-Forward Network (FFN): LayerNorm2d -> 1x1 Conv (Expansion) -> SimpleGate ->
       1x1 Conv (Projection) -> Dropout -> Gamma Scaling -> Add Skip

    Mathematical Formulation:
        Sub-Block A:
            x_norm = LayerNorm2d(x)
            branch_a = conv3(SCA(SimpleGate(DWConv(conv1(x_norm)))))
            y = x + beta * dropout(branch_a)

        Sub-Block B:
            y_norm = LayerNorm2d(y)
            branch_b = conv5(SimpleGate(conv4(y_norm)))
            out = y + gamma * dropout(branch_b)

    Args:
        c (int): Number of feature channels (C).
        DW_Expand (int, optional): Expansion factor for spatial depthwise branch. Defaults to 2.
        FFN_Expand (int, optional): Expansion factor for Feed-Forward branch. Defaults to 2.
        drop_out_rate (float, optional): Dropout probability. Defaults to 0.0.

    Raises:
        ValueError: If channel count `c` <= 0, expansion factors <= 0, or `drop_out_rate` < 0.0 or > 1.0.

    Tensor Shapes:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)

    Attributes:
        c (int): Number of channels.
        beta (nn.Parameter): Learnable residual scale parameter for branch A of shape (1, C, 1, 1), initialized to 0.
        gamma (nn.Parameter): Learnable residual scale parameter for branch B of shape (1, C, 1, 1), initialized to 0.
    """

    def __init__(
        self,
        c: int,
        DW_Expand: int = 2,
        FFN_Expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if c <= 0:
            raise ValueError(f"c (channels) must be a positive integer, got {c}")
        if DW_Expand <= 0:
            raise ValueError(f"DW_Expand must be a positive integer, got {DW_Expand}")
        if FFN_Expand <= 0:
            raise ValueError(f"FFN_Expand must be a positive integer, got {FFN_Expand}")
        if not (0.0 <= drop_out_rate <= 1.0):
            raise ValueError(
                f"drop_out_rate must be between 0.0 and 1.0, got {drop_out_rate}"
            )

        self.c = c
        self.DW_Expand = DW_Expand
        self.FFN_Expand = FFN_Expand
        self.drop_out_rate = drop_out_rate

        dw_channels = c * DW_Expand
        dw_gate_channels = dw_channels // 2

        ffn_channels = c * FFN_Expand
        ffn_gate_channels = ffn_channels // 2

        # Sub-Block A: Spatial & Channel Mixer
        self.spatial_norm = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(
            c, dw_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_gate_channels)
        self.conv3 = nn.Conv2d(
            dw_gate_channels, c, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

        # Sub-Block B: Feed-Forward Network
        self.ffn_norm = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(
            c, ffn_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(
            ffn_gate_channels, c, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for NAFBlock.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output feature tensor of shape (B, C, H, W).

        Raises:
            ValueError: If input tensor is not 4D or channel dimension mismatches `c`.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D tensor")
        if x.size(1) != self.c:
            raise ValueError(
                f"Expected input tensor with {self.c} channels at dim=1, "
                f"got tensor with {x.size(1)} channels"
            )

        # Sub-Block A
        x_norm = self.spatial_norm(x)
        x_branch = self.conv1(x_norm)
        x_branch = self.conv2(x_branch)
        x_branch = self.sg1(x_branch)
        x_branch = self.sca(x_branch)
        x_branch = self.conv3(x_branch)
        x_branch = self.dropout1(x_branch)
        y = x + self.beta * x_branch

        # Sub-Block B
        y_norm = self.ffn_norm(y)
        y_branch = self.conv4(y_norm)
        y_branch = self.sg2(y_branch)
        y_branch = self.conv5(y_branch)
        y_branch = self.dropout2(y_branch)
        out = y + self.gamma * y_branch

        return out
