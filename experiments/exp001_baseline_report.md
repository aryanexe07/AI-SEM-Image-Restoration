# Experiment exp001_nafnet_baseline Report

## 1. Experiment Overview
- **Experiment ID**: `exp001_nafnet_baseline`
- **Date**: 2026-08-08
- **Objective**: Establish the initial baseline for SEM image noise reduction and 2× super-resolution using NAFNet.
- **Platform**: Kaggle
- **Training Duration**: Not recorded
- **GPU**: Standard Kaggle GPU (exact GPU model: Not recorded)

## 2. Dataset
- **Purpose**: Paired SEM micrographs for noise reduction and 2× super-resolution.
- **Training Sample Count**: 3,200 paired samples (low-resolution noisy inputs paired with high-resolution ground-truth targets).
- **Test Sample Count**: 400 NoisyLR samples.
- **Paired/Unpaired Nature**: Training and validation splits are paired; test split is unpaired.
- **Input/Target Relationship**: Degraded low-resolution SEM images mapped to 256×256 high-resolution ground-truth SEM images (2× upscaling).
- **Test-Set Ground-Truth Limitation**: The 400 test samples do not have corresponding ground-truth images. Therefore, full-reference PSNR/SSIM metrics cannot be calculated or reported on the test split.

## 3. Model Configuration

| Configuration Parameter | Value |
|---|---|
| Architecture | NAFNet |
| Width | 32 |
| Encoder Blocks | [1, 1, 1] |
| Middle Blocks | 1 |
| Decoder Blocks | [1, 1, 1] |
| Upscale Factor | 2 |
| Loss Function | CharbonnierLoss |
| Loss Epsilon ($\epsilon$) | 1e-3 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| $T_{\max}$ | 50 |
| $\eta_{\min}$ | 1e-6 |
| Batch Size | 4 |
| Epochs | 50 |
| Automatic Mixed Precision (AMP) | False |

## 4. Training Procedure
The experiment follows a standard PyTorch supervised training pipeline:

$$\text{Dataset} \rightarrow \text{DataLoader} \rightarrow \text{NAFNet} \rightarrow \text{Charbonnier Loss} \rightarrow \text{AdamW} \rightarrow \text{CosineAnnealingLR} \rightarrow \text{Validation} \rightarrow \text{PSNR/SSIM Evaluation} \rightarrow \text{Checkpointing}$$

1. **Dataset & DataLoader**: Paired SEM samples are loaded in mini-batches of size 4.
2. **Forward Pass**: Degraded input images pass through NAFNet to generate restored 256×256 high-resolution outputs.
3. **Loss Computation**: Charbonnier Loss ($\epsilon = 1\times 10^{-3}$) computes the pixel-level reconstruction error against the ground truth.
4. **Optimization**: AdamW optimizer updates weights using computed gradients ($\text{lr} = 1\times 10^{-3}$, $\text{weight decay} = 1\times 10^{-4}$).
5. **Learning Rate Schedule**: CosineAnnealingLR adjusts the learning rate each epoch down to $\eta_{\min} = 1\times 10^{-6}$ over $T_{\max} = 50$ epochs.
6. **Validation Evaluation**: Validation set performance is measured after each epoch using full-reference PSNR and SSIM metrics against ground truth.
7. **Checkpointing**: Model weights and metric states are tracked across epochs.

## 5. Baseline Definition
The raw noisy baseline is defined as the degraded input image after appropriate 2× spatial alignment (bicubic interpolation to 256×256 resolution) for direct metric evaluation against the 256×256 ground truth.

*Note*: The raw noisy baseline is the unenhanced input baseline, **not** the NAFNet model output. The raw noisy baseline PSNR (22.9069 dB) was calculated using the paired validation set, as the 400 test images lack ground-truth targets.

## 6. Quantitative Results

| Metric | Result |
|---|---:|
| Raw noisy PSNR | 22.9069 dB |
| Best NAFNet PSNR | 29.4118 dB |
| PSNR improvement | +6.5049 dB |
| Required improvement | +3.0000 dB |
| Best PSNR epoch | 50 |
| Best SSIM | 0.7891 |
| Best SSIM epoch | 46 |
| Acceptance | PASS |

## 7. Acceptance Criterion

### Calculation

$$\text{PSNR Improvement} = \text{Best NAFNet Validation PSNR} - \text{Raw Noisy Baseline PSNR}$$

$$\text{PSNR Improvement} = 29.4118\text{ dB} - 22.9069\text{ dB} = +6.5049\text{ dB}$$

### Requirement
$$\text{Required PSNR Improvement} \ge +3.0000\text{ dB}$$

### Margin
$$6.5049\text{ dB} - 3.0000\text{ dB} = 3.5049\text{ dB}$$

The baseline NAFNet model exceeded the required +3.0000 dB threshold by **3.5049 dB**.

**Result**: **PASS**

## 8. Training Observations
- PSNR improved throughout the observed training period and reached its best recorded value of 29.4118 dB at epoch 50. Because the best PSNR occurred at the final epoch, convergence beyond 50 epochs was not established.
- SSIM reached its best recorded value of 0.7891 at epoch 46, while PSNR reached its best value at epoch 50.
- The model achieved a substantial quantitative improvement of +6.5049 dB PSNR over the unenhanced raw noisy baseline.

## 9. Artifacts
- **Best Model Checkpoint**: Not recorded / Not present in workspace `outputs/checkpoints`.
- **TensorBoard Logs**: Recorded event files present in `outputs/tensorboard/`.
- **Prediction Visualizations**: Not recorded / Not present in workspace `outputs/predictions`.
- **Training Logs**: System execution logs recorded in `logs/`.
- **baseline_outputs.zip**: Not recorded / Not present in repository.

## 10. Limitations
- **Training Epochs**: The experiment was conducted for only 50 epochs.
- **Convergence Verification**: Because the best PSNR occurred at epoch 50, potential performance gains beyond 50 epochs remain unverified.
- **Test-Set Evaluation**: The 400 NoisyLR test images do not have ground-truth counterparts; full-reference test-set PSNR and SSIM cannot currently be computed.
- **Hardware Metrics**: VRAM utilization, GPU model details, and wall-clock training time were not recorded.

## 11. Conclusion
The `exp001_nafnet_baseline` experiment **PASSED** its acceptance criterion. NAFNet achieved a 29.4118 dB validation PSNR compared with a 22.9069 dB raw noisy baseline, resulting in a +6.5049 dB improvement against the required +3.0 dB threshold.

This report documents the initial baseline experiment for the SEM image restoration project and does not represent a final or fully-optimized model configuration.

## 12. Next Steps
- Qualitatively evaluate the trained NAFNet baseline model on the 400 NoisyLR test split images.
- Generate and review output restoration visualizations.
- Inspect TensorBoard training curves for detailed metric trajectory analysis.
- Investigate whether training for additional epochs improves PSNR beyond 29.4118 dB.
- Conduct controlled hyperparameter experiments as required by future project phases.
