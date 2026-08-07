# NAFNet Architecture Verification Checklist

**Issue**: #11 — Full NAFNet Architecture (2× Super-Resolution)
**Date**: 2026-08-07
**Status**: Pre-Implementation

---

## Component Verification

- [x] Head Conv
- [x] Encoder 0
- [x] Encoder 1
- [x] Encoder 2
- [x] Downsample 0
- [x] Downsample 1
- [x] Downsample 2
- [x] Middle Bottleneck
- [x] Upsample 2
- [x] Upsample 1
- [x] Upsample 0
- [x] Decoder 2
- [x] Decoder 1
- [x] Decoder 0
- [x] PixelShuffle 2× Tail
- [x] Tail Conv (ending)
- [x] Global Residual
- [x] check_image_size (Padding/Unpadding)
- [x] Builder (build_model)
- [x] Forward Pass Contract (B,1,128,128) → (B,1,256,256)
