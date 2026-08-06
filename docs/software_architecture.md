# Master Software Architecture Specification

**Project Title**: AI-Based Restoration of Degraded Scanning Electron Microscope (SEM) Images using NAFNet  
**Document Type**: System Software Architecture Blueprint  
**Document Status**: Official Master Engineering Specification  
**Complementary Specifications**:  
- [Project README](../README.md)  
- [Dataset Characterization Report](dataset_characterization.md)  
- [Data Pipeline Design Specification](data_pipeline_design.md)  

---

## 1. Architecture Overview

This document specifies the software architecture governing the development of the **sem-image-restoration-nafnet** PyTorch codebase. Modern deep learning research frameworks require a strict separation of concerns to avoid technical debt, silent bugs, and non-reproducible empirical results.

### Architectural Philosophy
The software architecture follows a **decoupled, modular, and configuration-driven design**. Rather than embedding hyperparameters or hardcoded logic inside neural network training loops, every major functional concern is isolated within a dedicated package under `src/`.

```text
+-------------------------------------------------------------------------------+
|                        MODULAR SYSTEM ARCHITECTURE                            |
+-------------------------------------------------------------------------------+
|  High Maintainability   | Isolated packages enable independent refactoring.  |
|  High Reusability       | Network models, metrics, & loaders are decoupled.   |
|  High Testability       | Pure functions & classes enable unit testing.       |
|  Research Scalability   | Swap architectures or datasets via YAML configs.   |
|  Reproducibility        | Deterministic seeding & snapshot tracking.          |
+-------------------------------------------------------------------------------+
```

---

## 2. High-Level System Diagram

The diagram below illustrates the end-to-end execution and component interaction flow of the framework:

```text
                               +-----------------------------+
                               | YAML Configurations         |
                               | (configs/train.yaml, etc.)   |
                               +-----------------------------+
                                              |
                                              v
+-----------------------------+     +-------------------------+     +-------------------------+
| Disk Storage (.npy)         | --> | Dataset Module          | --> | Augmentation Transforms |
| (D:/Programming/python/...) |     | (src/datasets)          |     | (albumentations)        |
+-----------------------------+     +-------------------------+     +-------------------------+
                                                                                 |
                                                                                 v
+-----------------------------+     +-------------------------+     +-------------------------+
| Training Engine             | <-- | PyTorch DataLoader      | <-- | Formatted Tensor        |
| (src/engine/trainer.py)     |     | (Batched Mini-Batches)  |     | Mini-Batches            |
+-----------------------------+     +-------------------------+     +-------------------------+
       |               |
       | Forward Pass  | Backward Loss
       v               v
+-----------------------------+     +-------------------------+     +-------------------------+
| NAFNet Model Architecture   | --> | Differentiable Loss     | --> | Optimizer & Scheduler   |
| (src/models/nafnet.py)      |     | (src/losses)            |     | (AdamW / Cosine LR)     |
+-----------------------------+     +-------------------------+     +-------------------------+
       |
       | Predictions
       v
+-----------------------------+     +-------------------------+     +-------------------------+
| Metric Evaluator            | --> | Checkpoint Manager      | --> | Visualization & Results |
| (PSNR & SSIM in src/metrics)|     | (outputs/checkpoints)   |     | (results/ & TensorBoard)|
+-----------------------------+     +-------------------------+     +-------------------------+
```

---

## 3. Layered Architecture

The framework is structured into seven logical architectural layers:

```text
+-------------------------------------------------------------------------------+
| 7. PRESENTATION & RESULTS LAYER   | TensorBoard, Visualization, Markdown Tables|
+-----------------------------------+-------------------------------------------+
| 6. EVALUATION LAYER               | PSNR, SSIM, Latency Metrics               |
+-----------------------------------+-------------------------------------------+
| 5. EXECUTION & ENGINE LAYER       | Trainer, Evaluator, Checkpoint Manager    |
+-----------------------------------+-------------------------------------------+
| 4. MODEL & LOSS LAYER             | NAFNet Architecture, SimpleGate, Loss Py  |
+-----------------------------------+-------------------------------------------+
| 3. DATA PROCESSING LAYER          | SEMDataset, DataLoader, Augmentations     |
+-----------------------------------+-------------------------------------------+
| 2. CONFIGURATION LAYER            | Config Parser, YAML Schemas, Environment  |
+-----------------------------------+-------------------------------------------+
| 1. CORE UTILITIES LAYER           | Logger, Seeding, Device Autodetect        |
+-------------------------------------------------------------------------------+
```

### Layer Responsibilities
1. **Core Utilities Layer**: Provides non-domain utilities (logging, random seed initialization, GPU autodetect) consumed by all higher layers.
2. **Configuration Layer**: Parses, validates, and freezes execution settings from hierarchical YAML files.
3. **Data Processing Layer**: Ingests raw array files, validates pairs, applies spatial transformations, and constructs PyTorch DataLoaders.
4. **Model & Loss Layer**: Implements neural network modules (NAFNet) and loss functions (PSNR Loss, Charbonnier Loss).
5. **Execution & Engine Layer**: Coordinates training loops, mixed-precision scaling (AMP), validation cycles, and checkpoint saving.
6. **Evaluation Layer**: Computes full-reference metric scores (PSNR, SSIM) for quantitative validation.
7. **Presentation & Results Layer**: Generates visual comparison grids, difference maps, TensorBoard scalar logs, and markdown tables.

---

## 4. Module Responsibilities

The codebase inside `src/` is strictly partitioned into functional sub-packages:

```text
src/
├── __init__.py        # Root package version declaration
├── datasets/          # Dataset classes, data loading logic, data splits
├── engine/            # Training execution engine, validation loop, evaluator
├── losses/            # Differentiable loss functions
├── metrics/           # Evaluation metrics (PSNR, SSIM)
├── models/            # NAFNet network architecture, building blocks
└── utils/             # Logger, seed management, YAML configuration parsers
```

### Package Ownership Boundaries

| Package | Allowed Responsibilities | Forbidden Responsibilities |
|---|---|---|
| `src/utils/` | Log formatting, seed setting, YAML loading. | Neural network layers, model forward calls. |
| `src/datasets/` | Array loading, patch extraction, augmentations. | Loss computation, optimizer stepping. |
| `src/models/` | PyTorch `nn.Module` classes, layer definitions. | Disk file reads, hardcoded path logic. |
| `src/losses/` | Differentiable loss functions (`torch.Tensor` input). | Dataset indexing, file saving. |
| `src/metrics/` | Full-reference evaluation metrics (PSNR, SSIM). | Backward pass computation, gradient stepping. |
| `src/engine/` | Orchestrating training loops, evaluation cycles. | Direct neural layer mathematical definitions. |

---

## 5. Dependency Rules

To prevent circular imports and tight coupling, dependencies between packages follow a strict **acyclic downward hierarchy**:

```text
                                 +------------------+
                                 |  src/utils/      |
                                 +------------------+
                                   ^     ^      ^
                                   |     |      |
            +----------------------+     |      +----------------------+
            |                            |                             |
  +------------------+          +------------------+         +------------------+
  |  src/datasets/   |          |  src/models/     |         |  src/metrics/    |
  +------------------+          +------------------+         +------------------+
            ^                            ^                             ^
            |                            |                             |
            +----------------------+     |      +----------------------+
                                   |     |      |
                                 +------------------+
                                 |  src/losses/     |
                                 +------------------+
                                           ^
                                           |
                                 +------------------+
                                 |  src/engine/     |
                                 +------------------+
                                           ^
                                           |
                                 +------------------+
                                 |  train.py (CLI)  |
                                 +------------------+
```

### Mandatory Dependency Constraints
* `src/utils` MUST NOT import from any other `src` package.
* `src/models` MAY import from `src/utils` ONLY.
* `src/datasets` MAY import from `src/utils` ONLY.
* `src/metrics` MAY import from `src/utils` ONLY.
* `src/losses` MAY import from `src/utils` ONLY.
* `src/engine` MAY import from `src/datasets`, `src/models`, `src/losses`, `src/metrics`, and `src/utils`.
* `train.py` acts as the root entry point, importing from `src/engine` and `src/utils`.

---

## 6. Configuration Flow

The configuration system acts as the single source of operational parameters across all execution layers:

```text
YAML Config Files ──> Config Parser ──> Immutable Config Object ──> Sub-System Injection
```

### Parameter Propagation Mapping

| Target Package | Config Injection Source | Injected Parameters |
|---|---|---|
| `src/utils/` | `configs/default.yaml` | Log paths, random seed, target execution device. |
| `src/datasets/` | `configs/train.yaml` | `dataset_root`, `patch_size`, `batch_size`, `num_workers`, `augmentations`. |
| `src/models/` | `configs/model.yaml` | `in_channels`, `width`, `enc_blk_nums`, `dec_blk_nums`, `middle_blk_num`. |
| `src/losses/` | `configs/train.yaml` | `loss_type`, loss weights, smoothing parameters ($\epsilon$). |
| `src/engine/` | `configs/train.yaml` | `epochs`, `learning_rate`, `weight_decay`, `checkpoint_dir`, `val_freq`. |
| `src/metrics/` | `configs/inference.yaml` | Metric flags (`calculate_psnr`, `calculate_ssim`), crop margin sizes. |

---

## 7. Data Flow

Data moves through a deterministic pipeline from raw disk arrays to evaluation visualizations:

```text
Disk Array (.npy) ──> NumPy float32 ──> Padded Patch ──> PyTorch Tensor (1, H, W)
                                                                 |
                                                                 v
TensorBoard Plot <── Metric Score <── Predicted Tensor (1, 2H, 2W) <── NAFNet Model
```

### Sequential Data Transformations
1. **Disk Ingestion**: Raw low-resolution array read from `D:/Programming/python/semicondata/...`.
2. **Preprocessing**: Array normalized, clipped to $[0.0, 1.0]$, and converted to 3D tensor $(1, H_{LR}, W_{LR})$.
3. **Mini-Batch Assembly**: Collated into mini-batch tensor $(B, 1, H_{LR}, W_{LR})$ on pinned host memory.
4. **Device Transfer**: Pushed to CUDA GPU memory using non-blocking stream operations.
5. **Forward Inference**: Processed by NAFNet to produce restored target prediction tensor $(B, 1, H_{HR}, W_{HR})$.
6. **Loss Calculation**: Evaluated against ground-truth tensor $(B, 1, H_{HR}, W_{HR})$ to produce scalar loss tensor.
7. **Metric Assessment**: Converted to NumPy arrays to compute full-reference PSNR and SSIM scores.
8. **Artifact Generation**: Logged to TensorBoard and saved as visual comparison plots in `results/images/`.

---

## 8. Public APIs

Each package in `src/` exposes a clean, documented public API via its `__init__.py` module.

### Interface Definitions

#### `src.datasets`
* `SEMDataset`: PyTorch `Dataset` subclass for loading and augmenting SEM image pairs.
* `build_dataloaders`: Factory function initializing train, validation, and test PyTorch `DataLoader` instances from configuration.

#### `src.models`
* `NAFNet`: Top-level neural network `nn.Module` class.
* `NAFBlock`: Foundational residual building block containing SimpleGate and Simplified Channel Attention (SCA).
* `build_model`: Factory function constructing a NAFNet model instance from a configuration object.

#### `src.losses`
* `PSNRLoss`: Differentiable PSNR loss function module.
* `CharbonnierLoss`: Differentiable Charbonnier loss module ($\sqrt{\|x - y\|^2 + \epsilon^2}$).
* `build_loss_function`: Factory function returning configured loss module.

#### `src.metrics`
* `calculate_psnr`: Function computing Peak Signal-to-Noise Ratio between prediction and target tensors.
* `calculate_ssim`: Function computing Structural Similarity Index Measure between prediction and target tensors.

#### `src.engine`
* `Trainer`: Execution class handling training epoch loops, optimizer steps, AMP scaling, and validation cycles.
* `Evaluator`: Evaluation class computing full-dataset validation metrics and visual residual outputs.

#### `src.utils`
* `setup_logger`: Function initializing dual console and file loggers.
* `set_seed`: Function enforcing global random seed determinism across Python, NumPy, and PyTorch.
* `load_config`: Function parsing YAML configuration files into structured config objects.

---

## 9. Error Handling Philosophy

The framework enforces a **graceful degradation and early reporting** error handling policy:

```text
                  +-----------------------------------+
                  | Exception Occurs in Execution     |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
  [ Fatal Configuration / Data Shape Error ]     [ Recoverable Runtime Error ]
            |                                               |
            v                                               v
  Log ERROR with Stack Trace                     Log WARNING with Details
  Terminate Execution Immediately                Execute Fallback Routine
```

### Domain Error Strategies

| Failure Mode | Classification | Handling Strategy |
|---|---|---|
| **Missing Config File** | Fatal | Raise `FileNotFoundError`; terminate execution with clear error message. |
| **Shape Mismatch in DataLoader** | Fatal | Assert input shape $(128, 128)$ vs target $(256, 256)$; raise `ValueError`. |
| **Corrupted Dataset `.npy` File** | Recoverable | Log `WARNING` detailing file path; load next index ($i + 1$). |
| **CUDA Out of Memory (OOM)** | Fatal | Catch `torch.cuda.OutOfMemoryError`; log error suggesting batch size reduction. |
| **NaN / Inf in Loss Function** | Fatal | Catch non-finite scalar loss; log model state snapshot; terminate training run. |

---

## 10. Logging Strategy

Logging is managed centrally by `src/utils/logger.py` using Python's standard `logging` library and PyTorch's `SummaryWriter`:

### Multi-Channel Logging Architecture
1. **Console Stream**: Standard output formatted for real-time progress monitoring (`INFO` level).
2. **File Stream**: Persistent plain-text execution logs written to `logs/execution_<timestamp>.log` (`DEBUG` level).
3. **TensorBoard Event Stream**: Binary event logs written to `outputs/tensorboard/<experiment_id>/` tracking:
   * Training loss per mini-batch & epoch.
   * Validation PSNR and SSIM per epoch.
   * Learning rate decay curves.
   * Visual comparison grids (Input vs Prediction vs Ground Truth).

---

## 11. Experiment Management

Experiments are tracked under structured, isolated run directories:

```text
outputs/
├── checkpoints/
│   └── exp001_nafnet_baseline/
│       ├── best_model.pth
│       └── checkpoint_epoch_050.pth
├── predictions/
│   └── exp001_nafnet_baseline/
└── tensorboard/
    └── exp001_nafnet_baseline/
```

### Experiment Artifact Principles
* **Snapshot Conservation**: Upon experiment initialization, the exact resolved configuration file is stored inside the experiment directory.
* **Checkpoint State Standards**: Checkpoint `.pth` files store a state dictionary containing:
  * `epoch`: Current epoch integer.
  * `model_state_dict`: Model weight parameters.
  * `optimizer_state_dict`: Optimizer momentum parameters.
  * `scheduler_state_dict`: Learning rate scheduler parameters.
  * `best_metric`: Peak validation PSNR score achieved.

---

## 12. Testing Strategy

The repository includes a comprehensive `pytest` suite under `tests/` to maintain code quality:

```text
tests/
├── test_dataset.py     # Verifies SEMDataset loading, array shapes, & augmentations
├── test_metrics.py     # Verifies PSNR & SSIM metric calculations against SciPy references
└── test_model.py       # Verifies NAFNet model forward pass, parameter shapes, & gradients
```

### Testing Types & Verification Rules
1. **Unit Tests**: Test individual functions and layers in isolation (e.g. SimpleGate activation behavior, PSNR metric calculation correctness).
2. **Integration Tests**: Test interactions between modules (e.g. DataLoader batch output feeding into NAFNet model forward pass).
3. **Shape Validation Tests**: Assert that input tensor $(B, 1, 128, 128)$ produces output tensor $(B, 1, 256, 256)$ across various batch sizes.

---

## 13. Coding Standards

All code contributions must adhere to strict software quality guidelines:

* **PEP 8 Compliance**: Enforced statically via `black --line-length 88` and `ruff check .`.
* **Import Ordering**: Managed automatically via `isort --profile black .`.
* **Static Type Hints**: Full type annotations required on all function arguments and return values (`typing.List`, `typing.Dict`, `typing.Tuple`, `torch.Tensor`).
* **Google-Style Docstrings**: Required for all public modules, classes, and functions.
* **No Global Mutable State**: No module-level mutable variables. State must be encapsulated in objects.
* **Absolute Imports**: All internal package imports must use absolute module paths rooted in `src` (e.g., `from src.models.nafnet import NAFNet`).

---

## 14. Performance Considerations

High-performance execution is achieved through targeted hardware optimization:

1. **Mixed Precision (AMP)**: Enable `torch.cuda.amp.autocast()` during training to double FLOPS throughput and reduce VRAM allocation.
2. **Memory Mapping**: Load NumPy binary arrays using `mmap_mode='r'` to prevent host RAM saturation.
3. **DataLoader Prefetching**: Use `pin_memory=True` and `persistent_workers=True` to hide CPU pre-processing latency.
4. **C-Contiguous Tensors**: Enforce C-contiguous memory layout on array tensors to maximize GPU memory bus bandwidth.

---

## 15. Extensibility

The software architecture is designed to support future research expansion without breaking backward compatibility:

```text
                              +-------------------------+
                              | Base Pipeline           |
                              | Architecture (src/)     |
                              +-------------------------+
                                           |
    +------------------+-------------------+-------------------+-------------------+
    |                  |                   |                   |                   |
    v                  v                   v                   v                   v
[ Alternative     [ Unpaired /        [ Distributed Data  [ TensorRT /        [ Multi-Channel   ]
  Models (SwinIR) ]   Noise2Noise ]     Parallel (DDP) ]    ONNX Export ]       SEM Inputs    ]
```

* **Pluggable Model Backbone**: New restoration architectures (e.g. SwinIR, Restormer) can be implemented in `src/models/` by extending `nn.Module` and registering in `build_model`.
* **Pluggable Loss Functions**: Custom physics-informed loss modules can be added to `src/losses/` without modifying training engine loops.
* **Multi-GPU Distributed Training**: The training engine can be adapted to PyTorch Distributed Data Parallel (`torch.nn.parallel.DistributedDataParallel`) via configuration flags.

---

## 16. Security & Reproducibility

1. **Deterministic Random Seeds**: `src/utils/seed.py` initializes random seeds across Python `random`, NumPy `np.random`, and PyTorch `torch.manual_seed()` / `torch.cuda.manual_seed_all()`.
2. **PyTorch CUDNN Determinism**: Enable `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` for deterministic GPU convolution operations.
3. **Dependency Pinning**: Development environments are mirrored using `requirements.txt` and `requirements-dev.txt`.

---

## 17. Development Roadmap

To minimize technical debt, avoid premature integration bugs, and streamline debugging, module implementation **must** proceed strictly according to the following 14-step sequence:

```text
Step 1: Configuration System (src/utils/config.py)
   │
   ▼
Step 2: Utility & Logger Modules (src/utils/logger.py, seed.py)
   │
   ▼
Step 3: Dataset Class Implementation (src/datasets/sem_dataset.py)
   │
   ▼
Step 4: Data Augmentation Pipeline (src/datasets/transforms.py)
   │
   ▼
Step 5: DataLoader Factory (src/datasets/builder.py)
   │
   ▼
Step 6: Metrics Calculator (src/metrics/psnr_ssim.py)
   │
   ▼
Step 7: Differentiable Loss Modules (src/losses/charbonnier.py, psnr_loss.py)
   │
   ▼
Step 8: NAFNet Architecture Modules (src/models/nafnet.py)
   │
   ▼
Step 9: Unit Test Suite Execution (tests/test_model.py, test_dataset.py)
   │
   ▼
Step 10: Training Engine Implementation (src/engine/trainer.py)
   │
   ▼
Step 11: Validation Engine Implementation (src/engine/evaluator.py)
   │
   ▼
Step 12: Main Entry Script (train.py CLI)
   │
   ▼
Step 13: Model Training & Hyperparameter Tuning
   │
   ▼
Step 14: Benchmarking, Visualization, & Final Reporting
```

### Rationale for Implementation Sequence
* **Steps 1–2**: Establish underlying infrastructure (logging, seeding, configuration parsing) required by all downstream modules.
* **Steps 3–5**: Build and verify the data layer independently using unit tests before introducing neural network models.
* **Steps 6–8**: Implement evaluation metrics, losses, and the NAFNet architecture, ensuring tensor shape contract compatibility ($(B, 1, 128, 128) \to (B, 1, 256, 256)$).
* **Steps 9–12**: Connect tested components into the training engine and CLI entry point.
* **Steps 13–14**: Execute training runs, generate quantitative metrics, and finalize research reporting.
