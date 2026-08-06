# SEM Image Restoration Dataset Documentation

This directory contains datasets used for training, validating, and testing NAFNet models for SEM image restoration.

## Expected Directory Layout

```text
datasets/
├── README.md
├── train/
│   ├── degraded/     # Degraded / noisy SEM images (*.npy or *.png)
│   └── clean/        # Ground-truth clean SEM images (*.npy or *.png)
├── val/
│   ├── degraded/
│   └── clean/
└── test/
    ├── degraded/
    └── clean/
```

## Dataset Specifications

- **Format**: NumPy arrays (`.npy`) or high-bit-depth images (PNG/TIFF).
- **Data Type**: `float32` normalized in range `[0.0, 1.0]` or `uint8`/`uint16`.
- **Dimensions**: Single-channel grayscale `(H, W)` or `(1, H, W)`.

## Download & Acquisition Instructions

1. Place dataset `.npy` files or images into their respective `train/`, `val/`, or `test/` split folders.
2. Ensure pairs (degraded vs. clean) share matching file basenames.

> **Important**: Do NOT commit raw dataset files (`.npy`, `.png`, `.h5`) to Git. They are ignored via `.gitignore`.
