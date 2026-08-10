# Experiment exp003_nafnet_width64 Report

## 1. Experiment Overview
- **Experiment ID**: `exp003_nafnet_width64`
- **Objective**: Evaluate the impact of increasing NAFNet model capacity to width=64 on SEM image restoration quality.
- **Platform**: Kaggle
- **GPU**: Tesla T4
- **Parameters**: 4,465,796 (4.47M)

## 2. Quantitative Results

| Metric | Result |
|---|---:|
| Raw noisy PSNR | 22.9069 dB |
| Best NAFNet PSNR | 30.0312 dB |
| PSNR improvement | +7.1243 dB |
| Best PSNR epoch | 50 |
| Best SSIM | 0.8013 |
| Best SSIM epoch | 50 |

## 3. Comparison to exp002 (Width 48)
- **exp002 PSNR**: 29.9887 dB
- **exp003 PSNR**: 30.0312 dB
- **Net PSNR Gain**: +0.0425 dB
- **exp002 SSIM**: 0.8004
- **exp003 SSIM**: 0.8013
- **Parameter Increase**: 2.52M -> 4.47M (+77%)

## 4. Conclusion
Scaling the NAFNet base width from 48 to 64 provided severely diminishing returns. We saw only a marginal +0.04 dB gain in PSNR despite nearly doubling the parameter count and computational footprint. 

**Recommendation:** For the final Hackathon submission, **exp002 (Width 48)** represents the optimal tradeoff between image restoration quality and computational efficiency. Width 64 pushes the model beyond the point of worthwhile returns for this specific architecture and dataset.
