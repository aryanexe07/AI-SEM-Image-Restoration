# Experiment exp002_nafnet_width48 Report

## 1. Experiment Overview
- **Experiment ID**: `exp002_nafnet_width48`
- **Objective**: Evaluate the impact of increasing NAFNet model capacity to width=48 on SEM image restoration quality.
- **Platform**: Kaggle
- **GPU**: Tesla T4
- **Parameters**: 2,521,444 (2.52M)

## 2. Quantitative Results

| Metric | Result |
|---|---:|
| Raw noisy PSNR | 22.9069 dB |
| Best NAFNet PSNR | 29.9887 dB |
| PSNR improvement | +7.0818 dB |
| Best PSNR epoch | 50 |
| Best SSIM | 0.8004 |
| Best SSIM epoch | 48 |

## 3. Comparison to Baseline (Width 32)
- **Baseline PSNR**: 29.4118 dB
- **exp002 PSNR**: 29.9887 dB
- **Net PSNR Gain**: +0.5769 dB
- **Baseline SSIM**: 0.7891
- **exp002 SSIM**: 0.8004
- **Parameter Increase**: ~1.13M -> 2.52M (+123%)

## 4. Conclusion
Increasing the NAFNet base width from 32 to 48 yielded a significant quantitative improvement (+0.57dB PSNR) at the cost of slightly more than doubling the parameter count. This indicates that the baseline width 32 model was under-parameterized for this complex noise distribution.
