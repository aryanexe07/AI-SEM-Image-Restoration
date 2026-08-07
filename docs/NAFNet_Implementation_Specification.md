# NAFNet Implementation Specification

## Section 1: Complete Module Dependency Graph

The NAFNet architecture is built hierarchically, where each component depends on simpler primitives. A faithful reimplementation must follow this dependency structure to ensure modularity and ease of testing.

```text
LayerNorm2d (Custom)
        ↓
SimpleGate (Gating Logic)
        ↓
SCA (Simplified Channel Attention)
        ↓
NAFBlock (Core Atomic Unit)
        ↓
Encoder / Decoder (Hierarchical Blocks)
        ↓
NAFNet (Full UNet Model)
```

### Dependency Rationale
- **LayerNorm2d**: The foundational normalization layer used in every block. It must be implemented first as it handles the 4D tensor layout.
- **SimpleGate**: The core non-linearity. It is a primitive used inside the NAFBlock.
- **SCA**: The attention mechanism. It is a primitive used inside the NAFBlock's spatial mixing branch.
- **NAFBlock**: Combines normalization, gating, and attention. It is the repeated unit in the Encoder, Decoder, and Middle stages.
- **Encoder/Decoder**: Organizational structures that stack NAFBlocks and handle resolution changes (Downsampling/Upsampling).
- **NAFNet**: The top-level class that orchestrates the data flow through the UNet hierarchy.

---

## Section 2: File Mapping

| Official Repository Path | Target Implementation Path | Responsibility |
| :--- | :--- | :--- |
| `basicsr/models/archs/arch_util.py` | `src/models/layers.py` | `LayerNorm2d`, weight initialization. |
| `basicsr/models/archs/NAFNet_arch.py` | `src/models/nafnet.py` | `SimpleGate`, `NAFBlock`, `NAFNet`. |
| `basicsr/models/archs/NAFSSR_arch.py` | `src/models/nafssr.py` | `SCAM`, `DropPath`, `NAFBlockSR`, `NAFNetSR`, `NAFSSR`. |
| `basicsr/models/archs/local_arch.py` | `src/models/local.py` | `AvgPool2d`, `Local_Base` (TLSC support). |

---

## Section 3: Exact Constructor Specifications

### 3.1 LayerNorm2d
```python
LayerNorm2d(channels: int, eps: float = 1e-6)
```
- **channels**: Number of input channels ($C$).
- **eps**: Small constant for numerical stability in division.

### 3.2 NAFBlock
```python
NAFBlock(
    c: int, 
    DW_Expand: int = 2, 
    FFN_Expand: int = 2, 
    drop_out_rate: float = 0.0
)
```
- **c**: Input channel count.
- **DW_Expand**: Expansion factor for the depthwise convolution branch.
- **FFN_Expand**: Expansion factor for the Feed-Forward branch.
- **drop_out_rate**: Probability of dropout applied after projections.

### 3.3 NAFNet
```python
NAFNet(
    img_channel: int = 3, 
    width: int = 16, 
    middle_blk_num: int = 1, 
    enc_blk_nums: list = [], 
    dec_blk_nums: list = []
)
```
- **img_channel**: Number of input image channels (usually 3 for RGB).
- **width**: Initial feature channel count (the base width $W$).
- **middle_blk_num**: Number of NAFBlocks in the bottleneck (middle) stage.
- **enc_blk_nums**: List of integers defining the number of NAFBlocks in each encoder stage.
- **dec_blk_nums**: List of integers defining the number of NAFBlocks in each decoder stage.

### 3.4 NAFNetLocal
```python
NAFNetLocal(
    *args, 
    train_size: tuple = (1, 3, 256, 256), 
    fast_imp: bool = False, 
    **kwargs
)
```
- **train_size**: The $(B, C, H, W)$ shape used during training, required to calibrate TLSC.
- **fast_imp**: If `True`, uses a faster but non-equivalent sliding window implementation for TLSC.

---

## Section 4: Forward Pass Specification

### 4.1 NAFBlock Internal Execution
| Step | Operation | Input Shape | Output Shape | Logic |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `norm1` | $(B, C, H, W)$ | $(B, C, H, W)$ | LayerNorm2d |
| 2 | `conv1` | $(B, C, H, W)$ | $(B, C \times E_{dw}, H, W)$ | 1x1 Conv (Expansion) |
| 3 | `conv2` | $(B, C \times E_{dw}, H, W)$ | $(B, C \times E_{dw}, H, W)$ | 3x3 Depthwise Conv |
| 4 | `sg` | $(B, C \times E_{dw}, H, W)$ | $(B, \frac{C \times E_{dw}}{2}, H, W)$ | SimpleGate (Chunk + Mul) |
| 5 | `sca` | $(B, C, H, W)$ | $(B, C, H, W)$ | Attention Scaling |
| 6 | `conv3` | $(B, C, H, W)$ | $(B, C, H, W)$ | 1x1 Conv (Projection) |
| 7 | `residual_a` | $(B, C, H, W)$ | $(B, C, H, W)$ | `inp + branch * beta` |
| 8 | `norm2` | $(B, C, H, W)$ | $(B, C, H, W)$ | LayerNorm2d |
| 9 | `conv4` | $(B, C, H, W)$ | $(B, C \times E_{ffn}, H, W)$ | 1x1 Conv (Expansion) |
| 10 | `sg` | $(B, C \times E_{ffn}, H, W)$ | $(B, \frac{C \times E_{ffn}}{2}, H, W)$ | SimpleGate |
| 11 | `conv5` | $(B, C, H, W)$ | $(B, C, H, W)$ | 1x1 Conv (Projection) |
| 12 | `residual_b` | $(B, C, H, W)$ | $(B, C, H, W)$ | `y + branch * gamma` |

---

## Section 5: Tensor Shape Evolution

### UNet Path (NAFNet)
Assume `width=64`, `enc_blk_nums=[2, 2]`, `dec_blk_nums=[2, 2]`, Input `(1, 3, 256, 256)`.

| Layer | Type | In Shape | Out Shape | Skip Output |
| :--- | :--- | :--- | :--- | :--- |
| `intro` | Conv 3x3 | `(1, 3, 256, 256)` | `(1, 64, 256, 256)` | - |
| `encoders[0]` | 2 Blocks | `(1, 64, 256, 256)` | `(1, 64, 256, 256)` | `encs[0]` |
| `downs[0]` | Conv 2x2, s2 | `(1, 64, 256, 256)` | `(1, 128, 128, 128)` | - |
| `encoders[1]` | 2 Blocks | `(1, 128, 128, 128)` | `(1, 128, 128, 128)` | `encs[1]` |
| `downs[1]` | Conv 2x2, s2 | `(1, 128, 128, 128)` | `(1, 256, 64, 64)` | - |
| `middle_blks` | 1 Block | `(1, 256, 64, 64)` | `(1, 256, 64, 64)` | - |
| `ups[0]` | PixShuf (r=2) | `(1, 256, 64, 64)` | `(1, 128, 128, 128)` | - |
| `decoders[0]` | 2 Blocks | `(1, 128, 128, 128)` | `(1, 128, 128, 128)` | (In + `encs[1]`) |
| `ups[1]` | PixShuf (r=2) | `(1, 128, 128, 128)` | `(1, 64, 256, 256)` | - |
| `decoders[1]` | 2 Blocks | `(1, 64, 256, 256)` | `(1, 64, 256, 256)` | (In + `encs[0]`) |
| `ending` | Conv 3x3 | `(1, 64, 256, 256)` | `(1, 3, 256, 256)` | - |

---

## Section 6: Layer Specifications

### Convolutions Detail
| Layer | Kernel | Stride | Padding | Groups | Bias |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `intro` | 3x3 | 1 | 1 | 1 | True |
| `NAFBlock.conv1` | 1x1 | 1 | 0 | 1 | True |
| `NAFBlock.conv2` | 3x3 | 1 | 1 | **$C \times E_{dw}$** | True |
| `NAFBlock.conv3` | 1x1 | 1 | 0 | 1 | True |
| `NAFBlock.sca.conv`| 1x1 | 1 | 0 | 1 | True |
| `downs` | 2x2 | 2 | 0 | 1 | True |
| `ups.conv` | 1x1 | 1 | 0 | 1 | **False** |
| `ending` | 3x3 | 1 | 1 | 1 | True |

---

## Section 7: Mathematical Specification

### 7.1 SimpleGate
$$SG(x) = x[:, 0:C, :, :] \odot x[:, C:2C, :, :]$$
Where $\odot$ is element-wise multiplication. The input must have $2C$ channels, and output will have $C$ channels.

### 7.2 Simplified Channel Attention (SCA)
$$SCA(x) = x \cdot \phi(\text{Pool}(x))$$
$$\phi(v) = W_{1 \times 1} v + b$$
Where `Pool` is Global Average Pooling. Note that there is **no Sigmoid** activation.

### 7.3 LayerNorm2d
$$LN(x)_{c,h,w} = \gamma_c \frac{x_{c,h,w} - \mu_{h,w}}{\sqrt{\sigma^2_{h,w} + \epsilon}} + \beta_c$$
Where $\mu$ and $\sigma$ are calculated across the channel dimension for each pixel.

### 7.4 PSNRLoss
$$Loss = - \frac{10}{\ln 10} \ln(\text{MSE}(P, T) + 10^{-8})$$

---

## Section 8: Implementation Notes

- **Residual Scaling**: The parameters `beta` and `gamma` in `NAFBlock` must be initialized to **zero**.
- **Upsampling Conv**: The convolution before `PixelShuffle` in the decoder has `bias=False`.
- **Global Residual**: The final step of `NAFNet` is `x = x + inp`. This must be done *after* the `ending` convolution but *before* the final cropping.
- **Padding**: The `check_image_size` function uses `F.pad` with `mode='constant'` (default 0).

---

## Section 9: Required Unit Tests

### Test 1: SimpleGate Invariance
- **Input**: Tensor of shape $(1, 64, 32, 32)$ filled with 1.0.
- **Expected Output**: Tensor of shape $(1, 32, 32, 32)$ filled with 1.0.
- **Assertion**: `output.sum() == 1 * 32 * 32 * 32`.

### Test 2: NAFBlock Identity at Init
- **Input**: Random tensor $X$.
- **Expected Output**: $X$.
- **Condition**: All weights initialized, but `beta` and `gamma` parameters are 0.
- **Assertion**: `torch.allclose(NAFBlock(X), X, atol=1e-6)`.

---

## Section 10: Integration Order

1.  **LayerNorm2d**: Crucial for every other block.
2.  **SimpleGate**: Core non-linearity.
3.  **SCA**: Channel attention logic.
4.  **NAFBlock**: Combine 1-3. Verify with Test 2.
5.  **Downsample/Upsample**: Test resolution changes.
6.  **NAFNet**: Assemble the UNet. Verify global residual connection.

---

## Section 11: Debugging Guide

- **Runtime Error: `size mismatch` in Skip Add**: Usually caused by incorrect padding in `check_image_size`. Ensure `padder_size` is $2^L$ where $L$ is the number of downsampling stages.
- **NaN Loss**: Check if `eps` is missing in `LayerNorm2d` or `PSNRLoss`.
- **Poor Convergence**: Verify `beta` and `gamma` are initialized to zero. If they are random, the initial loss will be extremely high.

---

## Section 12: Performance Engineering Notes

- **Depthwise Convolutions**: Ensure `groups=channels` is set in `conv2` to leverage optimized CUDA kernels.
- **SimpleGate Implementation**: Use `x.chunk(2, dim=1)` for memory efficiency.
- **TLSC (Local Base)**: For inference on images $> 2000 \times 2000$, the global statistics in SCA become unreliable. Use the `AvgPool2d` from `local_arch.py` to calculate local statistics.

---

## Section 13: Hyperparameter Reference

| Hyperparameter | Denoising (SIDD) | Deblurring (GoPro) |
| :--- | :--- | :--- |
| **Width** | 64 | 64 |
| **Enc Blocks** | `[2, 2, 4, 8]` | `[1, 1, 1, 28]` |
| **Middle Blocks** | 12 | 1 |
| **Dec Blocks** | `[2, 2, 2, 2]` | `[1, 1, 1, 1]` |
| **Expansion ($E$)** | 2 | 2 |
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | $1e-3$ | $1e-3$ |
| **Grad Clip** | 0.01 | 0.01 |
| **Loss** | PSNRLoss | PSNRLoss |

---

## Section 14: Validation Guide

1.  **Parameter Count Verification**: For `width=64` SIDD variant, parameters should be ~67.8M.
2.  **Shape Trace**: Print `x.shape` after every encoder/decoder stage to verify the $2 \times$ scaling.
3.  **Gradient Check**: Ensure `beta` and `gamma` parameters have non-zero gradients after one backward pass.
