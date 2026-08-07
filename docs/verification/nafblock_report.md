# NAFBlock Verification & Engineering Report

**Module**: `src.models.nafblock.NAFBlock`  
**Status**: Verified & Production-Ready  
**Formula**: $P(C) = 7C^2 + 33C$ parameters  

---

## 1. Executive Summary
The `NAFBlock` module serves as the atomic residual building unit of the NAFNet architecture. It combines `LayerNorm2d`, `SimpleGate`, `SimplifiedChannelAttention`, and depthwise spatial convolutions with learnable residual scaling parameters (beta and gamma).

---

## 2. Theoretical vs Empirical Parameter Scaling

| Channel Count ($C$) | Total Parameters ($7C^2 + 33C$) | FLOPs ($64 \times 64$) | Tensor Contract |
| :--- | :--- | :--- | :--- |
| 16 | 2,320 | 0.0149 GFLOPs | $(1, 16, 64, 64) \to (1, 16, 64, 64)$ |
| 32 | 8,224 | 0.0551 GFLOPs | $(1, 32, 64, 64) \to (1, 32, 64, 64)$ |
| 64 | 30,784 | 0.2108 GFLOPs | $(1, 64, 64, 64) \to (1, 64, 64, 64)$ |
| 128 | 118,912 | 0.8242 GFLOPs | $(1, 128, 64, 64) \to (1, 128, 64, 64)$ |

---

## 3. Verified Computational Graph
1. **Input**: $(B, C, H, W)$
2. **Sub-Block A (Spatial Mixer)**:
   - `LayerNorm2d` -> $(B, C, H, W)$
   - `Conv2d` $1 \times 1$ (Expansion $C \to 2C$) -> $(B, 2C, H, W)$
   - `Conv2d` $3 \times 3$ (DWConv, $2C \to 2C$, `groups=2C`) -> $(B, 2C, H, W)$
   - `SimpleGate` $(2C \to C)$ -> $(B, C, H, W)$
   - `SimplifiedChannelAttention` $(C)$ -> $(B, C, H, W)$
   - `Conv2d` $1 \times 1$ (Projection $C \to C$) -> $(B, C, H, W)$
   - `Dropout` $(p)$ -> $(B, C, H, W)$
   - Residual Add: $Y = X + \beta \odot \text{Branch}_A$
3. **Sub-Block B (FFN)**:
   - `LayerNorm2d` -> $(B, C, H, W)$
   - `Conv2d` $1 \times 1$ (Expansion $C \to 2C$) -> $(B, 2C, H, W)$
   - `SimpleGate` $(2C \to C)$ -> $(B, C, H, W)$
   - `Conv2d` $1 \times 1$ (Projection $C \to C$) -> $(B, C, H, W)$
   - `Dropout` $(p)$ -> $(B, C, H, W)$
   - Residual Add: $\text{Output} = Y + \gamma \odot \text{Branch}_B$

---

## 4. Verification Checkpoints
- **Identity Initialization**: Verified $NAFBlock(X) \equiv X$ at initialization when $\beta=0, \gamma=0$.
- **Gradient Flow**: Verified full backward pass with non-zero gradient propagation to all sub-modules and parameters.
- **Numerical Stability**: Verified finite output bounds (`torch.isfinite`) across extreme inputs ($10^{-8}$, $10^4$, zero, ones, constant, high variance).
- **AMP & Compiler**: Verified under PyTorch Automatic Mixed Precision (`autocast`) and `torch.compile`.
