# Issue #43 — Standardize Reproducible Experiment Tracking

## 1. Objective

The objective of Issue #43 is to establish standardized, reproducible experiment tracking across all training runs for SEM image restoration. Every training execution automatically produces a machine-readable YAML experiment record (`<exp_id>_record.yaml`) capturing runtime metadata, model configuration, training parameters, compute environment context, evaluation metrics (PSNR, SSIM, LPIPS), and artifact paths.

This infrastructure enables scientific comparison of subsequent experiments (such as model scaling in #38, loss benchmarking in #39, and degradation augmentations in #40) without requiring manual record-keeping or reconstructing configuration details from memory.

---

## 2. Motivation

In deep learning research, unrecorded hyperparameter variations, git commits, or environment differences degrade reproducibility. For the SEM Image Restoration project, standardizing experiment tracking provides a single source of truth for every training run.

### Key KLA Webinar Findings

Insights from the KLA webinar defined the evaluation requirements:

- **Independent Metrics**: PSNR, SSIM, and LPIPS are distinct evaluation metrics used to measure restoration quality across pixel-level accuracy, structural similarity, and perceptual quality.
- **Undisclosed Weights**: KLA uses a fixed weighted combination of metrics for final evaluation, but the exact weighting formula is undisclosed.
- **No Invented Score**: The repository MUST NOT fabricate or estimate an unofficial composite "KLA score". All three metrics are tracked independently.
- **Visual Inspection**: KLA explicitly recommended qualitative visual inspection of restored SEM images alongside quantitative metrics.
- **LPIPS Usage**: LPIPS is purely an evaluation metric, not an assumed training loss function.

---

## 3. Implementation Overview

The experiment tracking system integrates cleanly into the existing training architecture without altering model architecture, loss functions, or training semantics:

```text
                      train.py
                         │
                         ▼
                 ExperimentTracker
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    Metadata          Training         Compute
        │              Config             │
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                     Trainer
                         │
                (Validation Epoch)
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
         PSNR          SSIM          LPIPS*
           │             │             │
           └─────────────┼─────────────┘
                         ▼
               tracker.update_validation()
                         │
                         ▼
        outputs/experiments/<exp_id>_record.yaml
```

`* LPIPS is optional and configurable (disabled by default).`

1. **`train.py` Initialization**: Upon execution, `train.py` resolves the dataset path, loads `Config`, builds dependencies, and constructs `ExperimentTracker`.
2. **Execution Context Gathering**: `ExperimentTracker` inspects the environment, PyTorch runtime, model parameter count, optimizer/scheduler settings, and Git revision.
3. **Incremental Logging**: At each validation epoch, `Trainer` computes validation metrics and invokes `tracker.update_validation(epoch, val_metrics)`.
4. **Atomic Record Persistence**: `ExperimentTracker` saves the YAML record after every validation epoch, protecting metadata against sudden run termination.

---

## 4. Experiment Record Schema

The standard machine-readable experiment record is saved to `outputs/experiments/<exp_id>_record.yaml` with the following structure:

### Experiment
- `id`: Experiment identifier string (e.g. `exp001_nafnet_baseline`).
- `git_commit`: 40-character Git revision SHA (or `null` if git is unavailable).
- `dataset.path`: Resolved portable dataset path (e.g. `datasets` or relative POSIX path).
- `dataset.splits`: List of dataset splits evaluated (e.g. `['train', 'val']`).

### Model
- `architecture`: Model class name (e.g. `NAFNet`).
- `parameters.total`: Total parameter count integer.
- `parameters.trainable`: Trainable parameter count integer.
- `config`: Dictionary of model architecture hyperparameters.

### Training
- `optimizer`: Optimizer class name (e.g. `AdamW`).
- `learning_rate`: Base learning rate float.
- `scheduler`: Scheduler class name (e.g. `CosineAnnealingLR`).
- `loss`: Primary loss function class name (e.g. `CharbonnierLoss`).
- `batch_size`: Mini-batch size integer.
- `epochs`: Total target epochs integer.
- `seed`: Random seed integer.

### Compute
- `platform`: Execution environment (`Kaggle`, `Google Colab`, `Windows`, `Linux`, `Darwin`).
- `device`: Resolved PyTorch device string (`cuda` or `cpu`).
- `gpu`: GPU device model name string (or `null` if CUDA is unavailable or CPU is targeted).
- `cuda_version`: CUDA runtime version string (or `null` if CUDA is unavailable).
- `pytorch_version`: PyTorch version string (e.g. `2.13.0+cpu`).
- `python_version`: Python executable version string (e.g. `3.13.14`).

### Metrics
- `psnr.best`: Peak validation PSNR float (Higher is better).
- `psnr.epoch`: Epoch number where best PSNR was achieved.
- `ssim.best`: Peak validation SSIM float (Higher is better).
- `ssim.epoch`: Epoch number where best SSIM was achieved.
- `lpips.best`: Lowest validation LPIPS float (Lower is better), or `null` if disabled/unavailable.
- `lpips.epoch`: Epoch number where best LPIPS was achieved, or `null`.

### Artifacts
- `best_checkpoint`: Relative path to best model checkpoint (`outputs/checkpoints/best_model.pth`).
- `latest_checkpoint`: Relative path to latest model checkpoint (`outputs/checkpoints/checkpoint_latest.pth`).
- `tensorboard`: Path to TensorBoard event log directory (`outputs/tensorboard`).
- `logs`: Path to system execution log directory (`logs`).
- `predictions`: Path to visual prediction output directory (`outputs/predictions`).
- `report`: Relative path to experiment Markdown report (`experiments/<exp_id>_report.md`).

---

## 5. LPIPS Integration

LPIPS (Learned Perceptual Image Patch Similarity) evaluation is implemented in `src/metrics/lpips.py`:

- **Configurable & Optional**: LPIPS evaluation is disabled by default (`metrics.lpips: false` in `configs/train.yaml`) to preserve training speed and eliminate mandatory weight downloads on routine runs.
- **Deterministic Grayscale Conversion**: SEM images are single-channel `(B, 1, H, W)` in `[0, 1]`. `calculate_lpips` expands 1-channel tensors to 3-channel RGB `(B, 3, H, W)` and rescales values to `[-1, 1]` as expected by LPIPS backbone networks:
  ```python
  if pred_tensor.shape[1] == 1:
      pred_tensor = pred_tensor.repeat(1, 3, 1, 1)
      target_tensor = target_tensor.repeat(1, 3, 1, 1)
  pred_tensor = pred_tensor * 2.0 - 1.0
  target_tensor = target_tensor * 2.0 - 1.0
  ```
- **Robust Error Handling**: If `lpips` library or pretrained weights cannot be loaded (e.g. offline environments), `calculate_lpips` returns `None` (`null` in YAML) without failing training.
- **Direction**: LPIPS distance tracks lower-is-better semantics (0.0 represents identical images).
- **Independent Metric**: LPIPS is recorded independently and is not combined into any synthetic KLA score or used as an assumed training loss.

---

## 6. Runtime / Reproducibility Metadata

`ExperimentTracker` automatically collects runtime environment parameters to prevent reliance on developer memory:

- **Git Revision**: Executed via `git rev-parse HEAD`. If git is absent or uninitialized, returns `null` safely.
- **Dataset Path Normalization**: Resolved from `SEM_DATASET_ROOT`, `SEM_DATASET_DIR`, or `Config`. `_normalize_dataset_path` strips personal developer absolute paths (e.g. `D:\Programming\python\semicon\datasets`) into relative POSIX paths (`datasets`), maintaining portability across local machines, Kaggle, and Colab.
- **Model Parameters**: Counts total and trainable parameters using `sum(p.numel() for p in model.parameters())`.
- **Compute Context**: Dynamically detects platform (Kaggle/Colab/OS) and queries PyTorch device properties. When CUDA is unavailable, `gpu` and `cuda_version` record `null` rather than fabricated values.

---

## 7. Incremental Tracking

To safeguard metadata during long training runs on cloud environments (Kaggle / Google Colab):

- The experiment tracker is updated **after every validation epoch** inside `Trainer.fit()`.
- `update_validation(epoch, val_metrics)` compares incoming epoch metrics against historical bests and immediately saves the YAML record file to disk.
- If a session terminates unexpectedly (e.g. Kaggle timeout or GPU preemption), all completed validation epochs, best metrics observed so far, and runtime metadata remain preserved on disk.

---

## 8. Checkpoint & Artifact Tracking

The tracking system maintains associations with repository artifact outputs:

- **Standard Checkpoints**: Preserves existing `CheckpointManager` semantics (`best_model.pth` updated on validation PSNR improvement, periodic `checkpoint_epoch_XXX.pth`).
- **Path Association**: The experiment record explicitly logs relative paths to checkpoints, TensorBoard events, console execution logs, prediction visual grids, and experiment Markdown reports.

---

## 9. Files Changed

| File | Change | Purpose |
|------|--------|---------|
| [`requirements.txt`](file:///d:/Programming/python/semicon/requirements.txt) | Modified | Added `lpips` package dependency. |
| [`pyproject.toml`](file:///d:/Programming/python/semicon/pyproject.toml) | Modified | Added `lpips` to project dependencies array. |
| [`configs/train.yaml`](file:///d:/Programming/python/semicon/configs/train.yaml) | Modified | Added `metrics` configuration section (`psnr: true`, `ssim: true`, `lpips: false`). |
| [`src/metrics/lpips.py`](file:///d:/Programming/python/semicon/src/metrics/lpips.py) | Created | Implemented `calculate_lpips` function with 1-channel to 3-channel expansion, range rescaling, and exception handling. |
| [`src/metrics/__init__.py`](file:///d:/Programming/python/semicon/src/metrics/__init__.py) | Modified | Re-exported `calculate_lpips`. |
| [`src/utils/experiment_tracker.py`](file:///d:/Programming/python/semicon/src/utils/experiment_tracker.py) | Created | Implemented `ExperimentTracker` class, dataset path normalizer, and compute environment inspector. |
| [`src/utils/__init__.py`](file:///d:/Programming/python/semicon/src/utils/__init__.py) | Modified | Re-exported `ExperimentTracker`. |
| [`src/engine/trainer.py`](file:///d:/Programming/python/semicon/src/engine/trainer.py) | Modified | Added `experiment_tracker` & `metrics_config` to `Trainer`. Updated `validate` for LPIPS and `fit` for incremental tracker updates. |
| [`train.py`](file:///d:/Programming/python/semicon/train.py) | Modified | Updated entry point to instantiate `ExperimentTracker` with resolved dataset path and pass to `Trainer`. |
| [`tests/test_experiment_tracker.py`](file:///d:/Programming/python/semicon/tests/test_experiment_tracker.py) | Created | Added unit tests for parameter counting, compute context, git commit, metric best tracking, null LPIPS formatting, and YAML record persistence. |
| [`tests/test_metrics.py`](file:///d:/Programming/python/semicon/tests/test_metrics.py) | Modified | Added test cases for grayscale LPIPS conversion and identical image evaluation. |
| [`tests/test_trainer.py`](file:///d:/Programming/python/semicon/tests/test_trainer.py) | Modified | Added test case verifying `Trainer` updates `ExperimentTracker` during `fit()`. |
| [`tests/test_train.py`](file:///d:/Programming/python/semicon/tests/test_train.py) | Modified | Updated CLI orchestration test assertions for `experiment_tracker`. |

---

## 10. Example Experiment Record

Below is an example of an actual generated experiment record file (`outputs/experiments/exp001_nafnet_baseline_record.yaml`) created during validation:

```yaml
experiment:
  id: exp001_nafnet_baseline
  git_commit: 2a7a6319cd411c5bf8e0aaa0547315afe1878e4e
  dataset:
    path: datasets
    splits:
    - train
    - val
model:
  architecture: NAFNet
  parameters:
    total: 1129028
    trainable: 1129028
  config:
    name: NAFNet
    img_channel: 1
    width: 32
    middle_blk_num: 1
    enc_blk_nums:
    - 1
    - 1
    - 1
    dec_blk_nums:
    - 1
    - 1
    - 1
    upscale: 2
training:
  optimizer: AdamW
  learning_rate: 0.001
  scheduler: CosineAnnealingLR
  loss: CharbonnierLoss
  batch_size: 4
  epochs: 100
  seed: 42
compute:
  platform: Windows
  device: cpu
  gpu: null
  cuda_version: null
  pytorch_version: 2.13.0+cpu
  python_version: 3.13.14
metrics:
  psnr:
    best: 29.4118
    epoch: 50
  ssim:
    best: 0.7891
    epoch: 50
  lpips:
    best: null
    epoch: null
artifacts:
  best_checkpoint: outputs/checkpoints/best_model.pth
  latest_checkpoint: outputs/checkpoints/checkpoint_latest.pth
  tensorboard: outputs/tensorboard
  logs: logs
  predictions: outputs/predictions
  report: experiments/exp001_nafnet_baseline_report.md
```

---

## 11. Testing & Verification

The implementation was verified by running the full repository test suite:

```bash
.venv\Scripts\pytest.exe
```

### Output Result
```text
=========================== short test summary info ===========================
SKIPPED [1] tests\test_model.py:460: torch.compile requires a C++ compiler not available
SKIPPED [1] tests\test_model.py:472: CUDA not available for AMP test
SKIPPED [1] tests\test_nafblock.py:221: torch.compile skipped
SKIPPED [1] tests\test_trainer.py:261: CUDA not available for FP16 AMP test
SKIPPED [1] tests\test_trainer.py:302: CUDA not available for BF16 test
209 passed, 5 skipped, 37 warnings in 33.72s
```
