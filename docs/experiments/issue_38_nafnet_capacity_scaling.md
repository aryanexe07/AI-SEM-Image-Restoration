# Issue #38 — NAFNet Capacity Scaling Experiment Report

## 1. Issue Overview

- **Issue Number**: #38
- **Objective**: Scale NAFNet capacity and benchmark quality-vs-compute tradeoff
- **Milestone**: Milestone 11 — Restoration Quality & Generalization
- **Status**: Completed
- **Purpose of the Experiment**: Determine whether increasing NAFNet architectural capacity (base channel width) produces meaningful SEM image restoration improvements and identify the point of diminishing returns.
- **Baseline Experiment**: `exp001_nafnet_baseline` (Width 32)
- **Scaled Experiments**: `exp002_nafnet_width48` (Width 48), `exp003_nafnet_width64` (Width 64)

This experiment was designed to systematically evaluate whether scaling model capacity (specifically the base width of NAFNet) improves restoration quality (measured by PSNR and SSIM) on degraded Scanning Electron Microscope (SEM) micrographs, and to pinpoint where computational cost outweighs quality gains.

The largest model (`width=64`) was **not** assumed to be optimal prior to experimentation. The goal was to empirical benchmark the quality-vs-capacity curve.

---

## 2. Baseline Reference — exp001

The baseline benchmark is established by `exp001_nafnet_baseline`, documented in [`experiments/exp001_baseline_report.md`](file:///d:/Programming/python/semicon/experiments/exp001_baseline_report.md) and [`configs/experiments/exp001.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp001.yaml).

### Architecture Configuration

- **Model Architecture**: NAFNet
- **Width (`width`)**: 32
- **Encoder Blocks (`enc_blk_nums`)**: `[1, 1, 1]`
- **Middle Blocks (`middle_blk_num`)**: `1`
- **Decoder Blocks (`dec_blk_nums`)**: `[1, 1, 1]`
- **Upscale Factor (`upscale`)**: 2

### Verified Baseline Metrics

- **Parameters**: 1,129,028 (approximately 1.13M)
- **Raw Noisy PSNR**: 22.9069 dB
- **Best NAFNet Validation PSNR**: 29.4118 dB
- **Best SSIM**: 0.7891
- **PSNR Improvement over Raw Noisy**: +6.5049 dB

---

## 3. Controlled Experimental Protocol

To ensure model capacity (base width) was the primary architectural variable under evaluation, `exp002_nafnet_width48` and `exp003_nafnet_width64` were conducted using identical dataset splits, loss functions, optimization schedules, and system configurations matching `exp001`.

As documented in [`configs/experiments/exp002_nafnet_width48.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp002_nafnet_width48.yaml) and [`configs/experiments/exp003_nafnet_width64.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp003_nafnet_width64.yaml):

- **Dataset**: Paired SEM dataset (`./datasets`, train and validation splits; 3,200 paired training samples, 400 validation samples)
- **Random Seed**: 42
- **Device**: `auto` (Executed on Kaggle Tesla T4 GPU)
- **Epochs**: 50
- **Batch Size**: 4 (`train_batch_size: 4`, `val_batch_size: 4`)
- **Learning Rate**: $1.0 \times 10^{-3}$ (`1.0e-3`)
- **Weight Decay**: $1.0 \times 10^{-4}$ (`1.0e-4`)
- **Minimum Learning Rate (`min_lr`)**: $1.0 \times 10^{-6}$ (`1.0e-6`)
- **Loss Function**: Charbonnier Loss (`charbonnier`)
- **Charbonnier Epsilon ($\epsilon$)**: $1.0 \times 10^{-3}$ (`1.0e-3`)
- **Mixed Precision (`mixed_precision`)**: `false`
- **Validation Frequency (`val_freq`)**: 1 epoch
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR ($T_{\max} = 50$, $\eta_{\min} = 1.0 \times 10^{-6}$)
- **Data-Loader Settings**: `num_workers: 4`, `pin_memory: true`
- **NAFNet Block Structure**: `enc_blk_nums: [1, 1, 1]`, `middle_blk_num: 1`, `dec_blk_nums: [1, 1, 1]`, `img_channel: 1`, `drop_out_rate: 0.0`
- **Upscale Factor**: 2

The experimental setup held depth (`enc_blk_nums=[1,1,1]`, `middle_blk_num=1`, `dec_blk_nums=[1,1,1]`), spatial resolution ($128 \times 128 \to 256 \times 256$), and training parameters constant, isolating model width as the sole architectural variable. Controlled experimental protocol is asserted to the extent verified by these repository configuration files.

---

## 4. Experiment Matrix

Below is the verified metric matrix comparing the baseline and scaled capacity experiments:

| Experiment | Width | Parameters | Best PSNR | Best SSIM | PSNR vs Previous | SSIM vs Previous |
|---|---:|---:|---:|---:|---:|---:|
| exp001 | 32 | ~1.13M | 29.4118 dB | 0.7891 | — | — |
| exp002 | 48 | 2,521,444 | 29.9887 dB | 0.8004 | +0.5769 dB | +0.0113 |
| exp003 | 64 | 4,465,796 | 30.0312 dB | 0.8013 | +0.0425 dB | +0.0009 |

---

## 5. Experiment 002 — Width 48

- **Experiment ID**: `exp002_nafnet_width48`
- **Report Reference**: [`experiments/exp002_nafnet_width48_report.md`](file:///d:/Programming/python/semicon/experiments/exp002_nafnet_width48_report.md)
- **Configuration**:
  - `width`: 48
  - `enc_blk_nums`: `[1, 1, 1]`
  - `middle_blk_num`: `1`
  - `dec_blk_nums`: `[1, 1, 1]`
  - `upscale`: 2
  - `epochs`: 50
  - `batch_size`: 4
  - `learning_rate`: $1.0 \times 10^{-3}$
  - `loss`: Charbonnier ($\epsilon = 1.0 \times 10^{-3}$)
  - `seed`: 42

### Verified Results

- **Parameters**: 2,521,444
- **Execution GPU**: Tesla T4
- **Raw Noisy PSNR**: 22.9069 dB
- **Best PSNR**: 29.9887 dB
- **Best PSNR Epoch**: 50
- **Best SSIM**: 0.8004
- **Best SSIM Epoch**: 48
- **PSNR Improvement over Raw Noisy**: +7.0818 dB ($29.9887\text{ dB} - 22.9069\text{ dB}$)
- **Improvement over exp001 PSNR**: +0.5769 dB ($29.9887\text{ dB} - 29.4118\text{ dB}$)
- **Improvement over exp001 SSIM**: +0.0113 ($0.8004 - 0.7891$)

### Empirical Interpretation

Increasing the NAFNet base width from 32 to 48 yielded a substantial improvement in restoration quality (+0.5769 dB PSNR, +0.0113 SSIM). This indicates that the baseline width 32 model was under-parameterized relative to the complexity of the SEM noise distribution.

---

## 6. Experiment 003 — Width 64

- **Experiment ID**: `exp003_nafnet_width64`
- **Report Reference**: [`experiments/exp003_nafnet_width64_report.md`](file:///d:/Programming/python/semicon/experiments/exp003_nafnet_width64_report.md)
- **Configuration**:
  - `width`: 64
  - `enc_blk_nums`: `[1, 1, 1]`
  - `middle_blk_num`: `1`
  - `dec_blk_nums`: `[1, 1, 1]`
  - `upscale`: 2
  - `epochs`: 50
  - `batch_size`: 4
  - `learning_rate`: $1.0 \times 10^{-3}$
  - `loss`: Charbonnier ($\epsilon = 1.0 \times 10^{-3}$)
  - `seed`: 42

### Verified Results

- **Parameters**: 4,465,796
- **Execution GPU**: Tesla T4
- **Raw Noisy PSNR**: 22.9069 dB
- **Best PSNR**: 30.0312 dB
- **Best PSNR Epoch**: 50
- **Best SSIM**: 0.8013
- **Best SSIM Epoch**: 50
- **PSNR Improvement over Raw Noisy**: +7.1243 dB ($30.0312\text{ dB} - 22.9069\text{ dB}$)
- **Improvement over exp002 PSNR**: +0.0425 dB ($30.0312\text{ dB} - 29.9887\text{ dB}$)
- **Improvement over exp002 SSIM**: +0.0009 ($0.8013 - 0.8004$)

### Empirical Interpretation

Scaling the base width further from 48 to 64 provided only a marginal metric improvement (+0.0425 dB PSNR, +0.0009 SSIM) despite increasing the parameter count by approximately +77%.

---

## 7. Capacity Scaling Analysis

### Width 32 $\to$ Width 48

- **PSNR Gain**: +0.5769 dB
- **SSIM Gain**: +0.0113
- **Parameter Increase**: 1,129,028 $\to$ 2,521,444 (~1.13M $\to$ 2.52M)
- **Parameter Increase Percentage**: $+123.33\%$ (approximately $+123\%$)

### Width 48 $\to$ Width 64

- **PSNR Gain**: +0.0425 dB
- **SSIM Gain**: +0.0009
- **Parameter Increase**: 2,521,444 $\to$ 4,465,796 (2.52M $\to$ 4.47M)
- **Parameter Increase Percentage**: $+77.11\%$ (approximately $+77\%$)

### Comparison of Marginal Returns

Scaling from width 32 to 48 yielded a meaningful improvement within the observed experiment (+0.5769 dB PSNR), whereas scaling from width 48 to 64 yielded only a marginal improvement relative to the additional parameter count (+0.0425 dB PSNR). No formal hypothesis testing was conducted; hence terms such as "significant" are avoided in a statistical sense.

---

## 8. Diminishing Returns

The empirical data across the three width settings demonstrates:

- **Width 32 $\to$ Width 48**: Meaningful metric gain (+0.5769 dB PSNR, +0.0113 SSIM) for +1.39M additional parameters (+123%).
- **Width 48 $\to$ Width 64**: Substantially smaller metric gain (+0.0425 dB PSNR, +0.0009 SSIM) for +1.94M additional parameters (+77%).

Therefore, **`width=48` is the observed point of diminishing returns** in NAFNet capacity scaling for this specific setup. Width 64 does not provide proportional restoration quality gains relative to width 48.

This finding does not prove that `width=48` is universally optimal across all possible depth configurations, loss functions, or dataset variations, but it establishes that width 48 represents the knee of the capacity scaling curve within the tested setup.

---

## 9. Quality vs Compute Trade-off

The table below summarizes all recorded hardware and compute measurements across the experiment suite:

| Compute Parameter | exp001 (Width 32) | exp002 (Width 48) | exp003 (Width 64) |
|---|---:|---:|---:|
| Parameter Count | 1,129,028 (~1.13M) | 2,521,444 (2.52M) | 4,465,796 (4.47M) |
| Training Time | Not recorded / not available in current experiment artifacts | Not recorded / not available in current experiment artifacts | Not recorded / not available in current experiment artifacts |
| Peak GPU Memory (VRAM) | Not recorded / not available in current experiment artifacts | Not recorded / not available in current experiment artifacts | Not recorded / not available in current experiment artifacts |
| FLOPs | Not recorded / not available in current experiment artifacts | Not recorded / not available in current experiment artifacts | Not recorded / not available in current experiment artifacts |
| End-to-End Inference Latency | Not recorded / not available in current experiment artifacts | Not recorded / not available in current experiment artifacts | Not recorded / not available in current experiment artifacts |

### Distinction Between Parameter Count and Compute Cost

While parameter count is fully verified across all three models (1.13M $\to$ 2.52M $\to$ 4.47M), actual runtime compute costs (wall-clock training time, peak VRAM, FLOPs, and inference latency) were not recorded in the available experiment logs. Consequently, quality vs parameter count can be characterized, but the full quality vs actual compute cost trade-off cannot yet be quantified.

---

## 10. LPIPS Evaluation

Inspection of the experiment tracking system introduced in Issue #43 ([`docs/issues/43-experiment-tracking.md`](file:///d:/Programming/python/semicon/docs/issues/43-experiment-tracking.md)) and the experiment records ([`outputs/experiments/exp001_nafnet_baseline_record.yaml`](file:///d:/Programming/python/semicon/outputs/experiments/exp001_nafnet_baseline_record.yaml)) reveals:

- LPIPS metric calculation was disabled by default (`metrics.lpips: false` in `configs/train.yaml`) during these capacity runs.
- **LPIPS Status**: LPIPS was not available/enabled for these capacity experiments (`lpips.best: null`).

### Evaluation Consideration

Full-reference metrics PSNR and SSIM assess pixel-level accuracy and structural similarity, but they do not capture perceptual quality or high-frequency texture preservation as effectively as perceptual metrics. In the absence of LPIPS scores, PSNR and SSIM alone do not provide a complete perceptual evaluation.

---

## 11. Qualitative Restoration

As documented in [`experiments/exp001_qualitative_failure_analysis.md`](file:///d:/Programming/python/semicon/experiments/exp001_qualitative_failure_analysis.md), visual comparison grids and failure mode assessments for improved models (`exp002` and `exp003`) are currently **unavailable / pending prediction artifact generation**.

- Visual output grids (`000214_comparison_grid.png`, etc.) for `exp002` and `exp003` have not yet been rendered or stored in `results/images/qualitative_analysis/`.
- Higher PSNR does not automatically guarantee superior visual quality or absence of subtle oversmoothing. Qualitative visual inspection will be performed once prediction artifacts are generated.

---

## 12. Final Model Selection

### Current Preferred Configuration

**`exp002_nafnet_width48`**

### Selection Rationale

1. Substantially improves over the `width=32` baseline (+0.5769 dB PSNR, +0.0113 SSIM).
2. `width=64` produces only a marginal additional metric gain (+0.0425 dB PSNR, +0.0009 SSIM) despite requiring +77% more parameters.
3. `width=48` provides the better observed quality-to-parameter trade-off.

### Status Clarification

`exp002_nafnet_width48` is designated as the **preferred configuration for subsequent experiments**, NOT as the "final optimal architecture". Subsequent evaluations of loss functions (Issue #39), data augmentations (Issue #40), generalization (Issue #41), LPIPS evaluation, and inference profiling (Issue #15) may adjust the final model selection.

---

## 13. Limitations

- **Width-Only Scaling**: Only model width was varied (`width=32, 48, 64`). Deeper NAFNet block configurations (e.g. altering `enc_blk_nums` or `dec_blk_nums` beyond `[1, 1, 1]`) were not evaluated in this experiment series.
- **Unrecorded Training Time**: Wall-clock training times were not recorded in the experiment logs.
- **Unrecorded Peak VRAM**: Peak GPU memory consumption was not recorded.
- **Unrecorded FLOPs**: Floating point operations per inference/training pass were not computed or logged.
- **Unrecorded Inference Latency**: End-to-end inference latency per image was not measured.
- **Unrecorded LPIPS**: Perceptual distance metrics were disabled during these runs (`lpips: null`).
- **No Generalization Guarantee**: Validation performance on the 400 validation samples does not prove generalization to unseen test distributions or hidden benchmark data.
- **No Statistical Significance Analysis**: No formal statistical significance testing was performed across seeds.
- **Dataset-Specific Results**: Findings are specific to the tested dataset split, resolution ($128 \times 128 \to 256 \times 256$), and training protocol.

---

## 14. Implications for Subsequent Issues

### Issue #39 — Loss Benchmarking
Use `width=48` (`exp002_nafnet_width48`) as the primary baseline architecture for controlled loss function benchmarking (comparing L1, Charbonnier, Perceptual, and composite losses) unless issue requirements specify otherwise.

### Issue #40 — Degradation-Aware Augmentation
Use `width=48` as the controlled baseline model while evaluating Gaussian noise, speckle noise, and downsampling data augmentations.

### Issue #41 — Generalization
Evaluate whether capacity scaling improvements observed on the validation set transfer across out-of-distribution SEM micrographs.

### Issue #42 — Qualitative Analysis
Perform multi-panel visual reviews to evaluate whether numerical metric gains (+0.5769 dB PSNR) correspond to visually sharper nanoscale edge definitions and artifact reduction.

### Issue #15 — Profiling
Measure actual end-to-end inference latency, throughput, and VRAM utilization for `width=48` versus other candidate models during formal profiling.

### Issue #17 — Final Release
Incorporate Issue #38 capacity scaling findings into the final model selection evidence matrix.

---

## 15. Final Conclusion

1. Scaling NAFNet base width from 32 (`exp001`) to 48 (`exp002`) improved validation PSNR by +0.5769 dB and SSIM by +0.0113.
2. Scaling base width further from 48 (`exp002`) to 64 (`exp003`) produced only a marginal additional gain of +0.0425 dB PSNR and +0.0009 SSIM.
3. This establishes an empirical point of diminishing returns around `width=48` under the tested setup.
4. `exp002_nafnet_width48` is therefore selected as the current preferred architecture for subsequent experiments.
5. This selection does not constitute proof that `width=48` is universally optimal.
6. Final model selection must incorporate loss function benchmarking, data augmentation, generalization testing, qualitative evaluation, LPIPS metrics, and actual inference profiling.

**Issue #38 — COMPLETED**

---

## 16. Reproducibility References

The findings in this report were compiled directly from the following repository artifacts:

- **Experiment Configurations**:
  - [`configs/experiments/exp001.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp001.yaml)
  - [`configs/experiments/exp002_nafnet_width48.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp002_nafnet_width48.yaml)
  - [`configs/experiments/exp003_nafnet_width64.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp003_nafnet_width64.yaml)
- **Experiment Reports**:
  - [`experiments/exp001_baseline_report.md`](file:///d:/Programming/python/semicon/experiments/exp001_baseline_report.md)
  - [`experiments/exp002_nafnet_width48_report.md`](file:///d:/Programming/python/semicon/experiments/exp002_nafnet_width48_report.md)
  - [`experiments/exp003_nafnet_width64_report.md`](file:///d:/Programming/python/semicon/experiments/exp003_nafnet_width64_report.md)
- **Experiment Tracking Records & Specifications**:
  - [`docs/issues/43-experiment-tracking.md`](file:///d:/Programming/python/semicon/docs/issues/43-experiment-tracking.md)
  - [`outputs/experiments/exp001_nafnet_baseline_record.yaml`](file:///d:/Programming/python/semicon/outputs/experiments/exp001_nafnet_baseline_record.yaml)
- **Qualitative & Analysis Records**:
  - [`experiments/exp001_qualitative_failure_analysis.md`](file:///d:/Programming/python/semicon/experiments/exp001_qualitative_failure_analysis.md)
- **Verification Scripts**:
  - [`scripts/verify_params.py`](file:///d:/Programming/python/semicon/scripts/verify_params.py)
