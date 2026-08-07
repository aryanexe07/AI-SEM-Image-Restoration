# NAFNet Implementation Checklist: Verification Protocol

This document provides a structured protocol to verify the faithfulness and correctness of an independent NAFNet implementation. Each phase focuses on a specific architectural tier, ensuring that errors are caught early in the development cycle.

## Phase 1: Foundational Primitives

The base layers must be verified for mathematical correctness and tensor layout compatibility before proceeding to higher-level blocks.

| Component | Verification Task | Expected Outcome |
| :--- | :--- | :--- |
| **LayerNorm2d** | Spatial Invariance | Mean and variance calculated across `dim=1` (channels) for each $(h, w)$ pixel independently. |
| **SimpleGate** | Channel Reduction | Input $(B, 2C, H, W) \to$ Output $(B, C, H, W)$. Verify with `x.chunk(2, 1)`. |
| **SCA** | Attention Logic | Global pooling followed by a 1x1 Conv with **no activation**. Output is element-wise product. |

> **Warning**: Ensure `LayerNorm2d` does not use the standard `nn.LayerNorm` which operates on the last dimension only.

---

## Phase 2: NAFBlock Atomic Unit

The NAFBlock is the core unit of computation. Its correctness is best verified by checking its identity mapping property at initialization.

### Constructor & Logic
- **Depthwise Groups**: Confirm `conv2` has `groups` equal to the number of input channels.
- **Residual Scales**: The parameters `beta` and `gamma` must be `nn.Parameter` and initialized to **zero**.
- **Branch Flow**: Verify the sequence: `Norm \to Expansion \to DWConv \to Gating \to Attention \to Projection`.

### Identity Verification
Initialize a NAFBlock with random weights. Ensure `beta` and `gamma` are exactly 0.0. A forward pass with any input $X$ must return $X$ within a tolerance of $10^{-6}$.

---

## Phase 3: UNet Hierarchical Structure

The global architecture manages resolution changes and long-range skip connections.

### Resolution & Channels
Verify the scaling logic in the Encoder and Decoder stages using the table below.

| Stage | Operation | Channel Change | Resolution Change |
| :--- | :--- | :--- | :--- |
| **Downsampling** | 2x2 Conv, s2 | $C \to 2C$ | $H, W \to H/2, W/2$ |
| **Upsampling** | PixShuf (r=2) | $C \to C/2$ | $H, W \to 2H, 2W$ |
| **Skip Fusion** | Addition | No change | No change |

### Padding & Cropping
- **Multiple Check**: The `padder_size` must be $2^L$ (usually 16 or 32).
- **Invariance**: Input $(H, W) \to$ Padded $(H', W') \to$ Model $\to$ Cropped $(H, W)$.

---

## Phase 4: Optimization & Training

The training objective and optimization strategy are specific to the NAFNet baseline.

- **PSNRLoss**: Confirm the logarithmic formulation: $- \frac{10}{\ln 10} \ln(\text{MSE} + \epsilon)$.
- **Gradient Clipping**: Verify that `torch.nn.utils.clip_grad_norm_` is set to exactly **0.01**.
- **Optimizer**: Ensure **AdamW** is used instead of standard Adam to correctly handle weight decay.

---

## Phase 5: Final Validation Protocol

Complete these final checks before considering the implementation production-ready.

1.  **Parameter Count**: Compare your implementation's `sum(p.numel())` against the official variants.
    - *SIDD Base (Width 64)*: ~67.8M parameters.
    - *GoPro Base (Width 64)*: ~67.8M parameters.
2.  **Gradient Flow**: Perform a backward pass on a random loss. Check that `beta` and `gamma` have non-zero `grad` attributes.
3.  **Inference Speed**: On a standard $256 \times 256$ image, the model should achieve throughput consistent with the official benchmarks (approx. 20-50 FPS depending on hardware).
4.  **Memory Trace**: Monitor VRAM usage during a forward pass of a $256 \times 256$ patch; it should be significantly lower than traditional UNets due to the additive skip connections.
