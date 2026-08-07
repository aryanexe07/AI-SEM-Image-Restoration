"""Complete NAFNet architecture for 2× SEM image super-resolution.

This module implements the full NAFNet (Nonlinear Activation Free Network)
encoder-decoder architecture with a 2× PixelShuffle super-resolution tail,
assembling the verified foundational primitives (LayerNorm2d, SimpleGate,
SimplifiedChannelAttention, NAFBlock) into a production-ready model.

Architecture Evidence (Three-Layer Verification):
    Layer 1 (Docs): docs/NAFNet_Architecture_Reverse_Engineering.md Sec 3, 4, 9, 10
    Layer 2 (Repo): megvii-research/NAFNet basicsr/models/archs/NAFNet_arch.py
    Layer 3 (Paper): arXiv:2204.04676 (NAFNet), CVPRW 2022 (NAFSSR)

Architecture Overview:
    Input (B, img_channel, H, W)
        ↓ Head Conv 3×3
    (B, width, H, W)
        ↓ Encoder Stage 0 [NAFBlock × enc_blk_nums[0]] → Skip 0
        ↓ Downsample 0 [Conv 2×2, stride 2]
    (B, width*2, H/2, W/2)
        ↓ Encoder Stage 1 [NAFBlock × enc_blk_nums[1]] → Skip 1
        ↓ Downsample 1
        ...
    (B, width*2^K, H/2^K, W/2^K)
        ↓ Middle Bottleneck [NAFBlock × middle_blk_num]
        ↓ Upsample K-1 [Conv 1×1 (bias=False) + PixelShuffle(2)]
        ↓ Skip Add K-1
        ↓ Decoder Stage K-1 [NAFBlock × dec_blk_nums[0]]
        ...
    (B, width, H, W)
        ↓ PixelShuffle ×2 Tail [Conv 3×3 + PixelShuffle(upscale)]
    (B, img_channel, H*upscale, W*upscale)
        ↓ Global Residual Add (+ bilinear-interpolated input)
    Output (B, img_channel, H*upscale, W*upscale)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nafblock import NAFBlock


def _validate_params(
    img_channel: int,
    width: int,
    middle_blk_num: int,
    enc_blk_nums: list[int],
    dec_blk_nums: list[int],
    upscale: int,
    drop_out_rate: float,
) -> None:
    if img_channel <= 0 or width <= 0 or middle_blk_num <= 0 or upscale <= 0:
        raise ValueError(
            "img_channel, width, middle_blk_num, and upscale must be positive integers"
        )
    if not (0.0 <= drop_out_rate <= 1.0):
        raise ValueError(
            f"drop_out_rate must be between 0.0 and 1.0, got {drop_out_rate}"
        )
    _validate_stage_lists(enc_blk_nums, dec_blk_nums)


def _validate_stage_lists(enc_blk_nums: list[int], dec_blk_nums: list[int]) -> None:
    if not enc_blk_nums or not dec_blk_nums:
        raise ValueError("enc_blk_nums and dec_blk_nums must be non-empty lists")
    if len(enc_blk_nums) != len(dec_blk_nums):
        raise ValueError("enc_blk_nums and dec_blk_nums must have the same length")
    if any(num <= 0 for num in enc_blk_nums):
        raise ValueError("enc_blk_nums block counts must be positive integers")
    if any(num <= 0 for num in dec_blk_nums):
        raise ValueError("dec_blk_nums block counts must be positive integers")


class NAFNet(nn.Module):
    """Complete NAFNet encoder-decoder architecture with 2× PixelShuffle SR tail.

    Implements a symmetric UNet structure with configurable encoder/decoder
    stages, additive skip connections, and a PixelShuffle super-resolution
    tail for upscaling output spatial dimensions.

    Architecture Evidence:
        Head Conv:
            Layer 1: NAFNet_Architecture_Reverse_Engineering.md Sec 3.1 & 4
            Layer 2: NAFNet_arch.py: self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
            Layer 3: Paper Sec 3.1: 3×3 Conv projection

        Encoder Stages:
            Layer 1: NAFNet_Architecture_Reverse_Engineering.md Sec 9
            Layer 2: NAFNet_arch.py: nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            Layer 3: Paper Sec 3.1: NAFBlock stages in encoder

        Downsample:
            Layer 1: NAFNet_Implementation_Specification.md Sec 6: Conv 2×2, s2, p0, bias=True
            Layer 2: NAFNet_arch.py: nn.Conv2d(chan, 2*chan, 2, 2)
            Layer 3: Paper Sec 3.1: Strided convolution for downsampling

        Middle Bottleneck:
            Layer 1: NAFNet_Architecture_Reverse_Engineering.md Sec 3.2
            Layer 2: NAFNet_arch.py: nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
            Layer 3: Paper Sec 3.1: Middle bottleneck blocks

        Upsample (Decoder):
            Layer 1: NAFNet_Implementation_Specification.md Sec 6: Conv 1×1, bias=False + PixelShuffle(2)
            Layer 2: NAFNet_arch.py: nn.Sequential(nn.Conv2d(chan, chan*2, 1, bias=False), nn.PixelShuffle(2))
            Layer 3: Paper Sec 3.1: PixelShuffle upsampling

        Skip Fusion:
            Layer 1: NAFNet_Architecture_Reverse_Engineering.md Sec 10: Element-wise addition
            Layer 2: NAFNet_arch.py: x = x + enc_skip
            Layer 3: Paper Sec 3.1: Additive skip connections

        PixelShuffle Tail (SR):
            Layer 1: data_pipeline_design.md, dataset_characterization.md: PixelShuffle(2) tail
            Layer 2: NAFSSR_arch.py: nn.Sequential(nn.Conv2d(width, img_channel*up_scale**2, 3, 1, 1), nn.PixelShuffle(up_scale))
            Layer 3: NAFSSR paper: PixelShuffle-based SR tail

        Global Residual:
            Layer 1: NAFNet_Implementation_Specification.md Sec 8: x = x + inp (same-res)
            Layer 2: NAFSSR_arch.py: inp_hr = F.interpolate(inp, scale_factor=up_scale, mode='bilinear'); out = out + inp_hr
            Layer 3: NAFSSR paper: Bilinear-interpolated input residual for SR

        Padding:
            Layer 1: NAFNet_Implementation_Specification.md Sec 8: F.pad with mode='constant'
            Layer 2: NAFNet_arch.py: check_image_size with F.pad
            Layer 3: Paper: Input must be compatible with 2^L downsampling

    Args:
        img_channel: Number of input/output image channels. Defaults to 1 (grayscale SEM).
            Verified: NAFNet_Implementation_Specification.md Sec 3.3.
        width: Base feature channel count (model width W). Defaults to 32.
            Verified: NAFNet_Architecture_Reverse_Engineering.md Sec 12.1.
        middle_blk_num: Number of NAFBlocks in the bottleneck stage. Defaults to 1.
            Verified: NAFNet_Architecture_Reverse_Engineering.md Sec 12.1.
        enc_blk_nums: List of NAFBlock counts per encoder stage. Defaults to [1, 1, 1].
            Verified: NAFNet_Architecture_Reverse_Engineering.md Sec 12.1.
        dec_blk_nums: List of NAFBlock counts per decoder stage. Defaults to [1, 1, 1].
            Verified: NAFNet_Architecture_Reverse_Engineering.md Sec 12.1.
        upscale: Super-resolution upscaling factor. Defaults to 2.
            Verified: Issue #11 specification; NAFSSR_arch.py up_scale parameter.
        drop_out_rate: Dropout probability for NAFBlocks. Defaults to 0.0.
            Verified: NAFNet_Implementation_Specification.md Sec 3.2.

    Raises:
        ValueError: If any parameter fails validation checks.

    Tensor Contract:
        Input: (B, img_channel, H, W) where H, W are arbitrary (auto-padded internally).
        Output: (B, img_channel, H * upscale, W * upscale).

    Example:
        >>> model = NAFNet(img_channel=1, width=32, middle_blk_num=1,
        ...                enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1], upscale=2)
        >>> x = torch.randn(1, 1, 128, 128)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 1, 256, 256])
    """

    def __init__(
        self,
        img_channel: int = 1,
        width: int = 32,
        middle_blk_num: int = 1,
        enc_blk_nums: list[int] | None = None,
        dec_blk_nums: list[int] | None = None,
        upscale: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # --- Apply Defaults ---
        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1]
        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1]

        # --- Validate & Store Configuration ---
        _validate_params(
            img_channel,
            width,
            middle_blk_num,
            enc_blk_nums,
            dec_blk_nums,
            upscale,
            drop_out_rate,
        )
        self.img_channel = img_channel
        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = list(enc_blk_nums)
        self.dec_blk_nums = list(dec_blk_nums)
        self.upscale = upscale
        self.drop_out_rate = drop_out_rate

        # --- Head Conv ---
        # Evidence: NAFNet_arch.py line 90:
        #   self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        # Tensor: (B, img_channel, H, W) -> (B, width, H, W)
        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        # --- Encoder Stages & Downsample Blocks ---
        # Evidence: NAFNet_arch.py lines 100-109
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(num)]
                )
            )
            # Evidence: NAFNet_arch.py line 107: nn.Conv2d(chan, 2*chan, 2, 2)
            # Tensor: (B, chan, H_k, W_k) -> (B, 2*chan, H_k/2, W_k/2)
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        # --- Middle Bottleneck ---
        # Evidence: NAFNet_arch.py lines 111-114
        # Tensor: (B, chan, H_bot, W_bot) -> (B, chan, H_bot, W_bot)
        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(chan, drop_out_rate=drop_out_rate)
                for _ in range(middle_blk_num)
            ]
        )

        # --- Decoder Stages & Upsample Blocks ---
        # Evidence: NAFNet_arch.py lines 116-128
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for num in dec_blk_nums:
            # Evidence: NAFNet_arch.py lines 117-121:
            #   nn.Sequential(nn.Conv2d(chan, chan*2, 1, bias=False), nn.PixelShuffle(2))
            # Note: bias=False verified in NAFNet_Implementation_Specification.md Sec 6 & 8
            # Tensor: (B, chan, H_k, W_k) -> (B, chan/2, 2*H_k, 2*W_k)
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(num)]
                )
            )

        # --- PixelShuffle ×2 Super-Resolution Tail ---
        # Evidence: NAFSSR_arch.py lines 171-174:
        #   self.up = nn.Sequential(
        #       nn.Conv2d(width, img_channel * up_scale**2, 3, 1, 1, bias=True),
        #       nn.PixelShuffle(up_scale)
        #   )
        # Tensor: (B, width, H, W) -> (B, img_channel, H*upscale, W*upscale)
        self.up_tail = nn.Sequential(
            nn.Conv2d(
                in_channels=width,
                out_channels=img_channel * upscale**2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True,
            ),
            nn.PixelShuffle(upscale),
        )

        # --- Padding Configuration ---
        # Evidence: NAFNet_arch.py line 130: self.padder_size = 2 ** len(self.encoders)
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete NAFNet architecture.

        Executes the full encoder-decoder pipeline with automatic input
        padding, skip connections, PixelShuffle SR tail, and global
        residual addition.

        Evidence: NAFNet_arch.py lines 132-155; NAFSSR_arch.py lines 177-192.

        Args:
            inp: Input image tensor of shape (B, img_channel, H, W).

        Returns:
            Output super-resolved tensor of shape (B, img_channel, H*upscale, W*upscale).

        Raises:
            ValueError: If input tensor is not 4D or channel count mismatches.
        """
        if inp.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B, C, H, W), got {inp.dim()}D tensor"
            )
        if inp.size(1) != self.img_channel:
            raise ValueError(
                f"Expected input tensor with {self.img_channel} channels at dim=1, "
                f"got tensor with {inp.size(1)} channels"
            )

        B, C, H, W = inp.shape

        # --- Padding ---
        # Evidence: NAFNet_arch.py lines 134, 157-162
        inp = self.check_image_size(inp)

        # --- Head Conv ---
        # Evidence: NAFNet_arch.py line 136: x = self.intro(inp)
        x = self.intro(inp)

        # --- Encoder Path ---
        # Evidence: NAFNet_arch.py lines 138-143
        encs: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # --- Middle Bottleneck ---
        # Evidence: NAFNet_arch.py line 145: x = self.middle_blks(x)
        x = self.middle_blks(x)

        # --- Decoder Path ---
        # Evidence: NAFNet_arch.py lines 147-150
        # Skip connections fused via addition: x = x + enc_skip
        for decoder, up, enc_skip in zip(
            self.decoders, self.ups, encs[::-1], strict=True
        ):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        # --- PixelShuffle ×2 SR Tail ---
        # Evidence: NAFSSR_arch.py lines 188, 171-174
        # Tensor: (B, width, H_padded, W_padded) -> (B, img_channel, H_padded*upscale, W_padded*upscale)
        x = self.up_tail(x)

        # --- Global Residual ---
        # Evidence: NAFSSR_arch.py line 178: inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        # Evidence: NAFSSR_arch.py line 190: out = out + inp_hr
        # Evidence: NAFNet_arch.py line 153: x = x + inp (for upscale == 1)
        if self.upscale == 1:
            x = x + inp
        else:
            inp_hr = F.interpolate(
                inp, scale_factor=self.upscale, mode="bilinear", align_corners=False
            )
            x = x + inp_hr

        # --- Unpadding ---
        # Evidence: NAFNet_arch.py line 155: return x[:, :, :H, :W]
        # For SR, output spatial size is H*upscale × W*upscale
        return x[:, :, : H * self.upscale, : W * self.upscale]

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to ensure spatial dimensions are divisible by padder_size.

        Evidence: NAFNet_arch.py lines 157-162.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Padded tensor with spatial dimensions divisible by self.padder_size.
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
