# Experiment Tracking & Notes

This directory holds experiment notes, comparison matrices, hyperparameter tuning logs, and ablation study reports.

## Experiment Naming Convention

- `exp001_<description>`: Initial baseline setup.
- `exp002_<description>`: Architecture variations (e.g., block counts, width).
- `exp003_<description>`: Loss function studies (e.g., PSNRLoss vs. Charbonnier).

## Summary Table Placeholder

| Exp ID | Model Config | Loss Function | Patch Size | Batch Size | Best PSNR (dB) | Best SSIM | Notes |
|---|---|---|---|---|---|---|---|
| exp001 | width=32, blks=[2,2,4,8] | L1 + PSNR | 128x128 | 16 | -- | -- | Baseline run |
