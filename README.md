# AI-Based Restoration of Degraded SEM Images using NAFNet

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Project Overview

Scanning Electron Microscopy (SEM) is critical in material science, nanotechnology, and semiconductor engineering for nanoscale surface imaging. However, SEM images often suffer from various noise degradation sources, including thermal noise, drift, charging artifacts, and low signal-to-noise ratios (SNR) during high-speed scans. 

This repository provides a modular, research-grade PyTorch implementation of **NAFNet (Nonlinear Activation Free Network)** tailored specifically for restoring noisy and degraded SEM images.

---

## Research Goal

The primary objective of this project is to restore degraded SEM images using the NAFNet baseline architecture while maximizing image quality measured quantitatively via:

* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index Measure)

---

## Features (Planned)

- **Config-Driven Architecture**: Fully modular training and inference pipeline configurable via YAML files.
- **NAFNet Backbone**: Efficient image restoration backbone eliminating conventional nonlinear activation functions (e.g., GELU, ReLU) in favor of SimpleGate and Simplified Channel Attention (SCA).
- **SEM Data Augmentation Pipeline**: Dedicated augmentation primitives tuned for electron microscopy noise characteristics using `albumentations`.
- **Reproducible Experimentation**: Unified experiment tracking with TensorBoard, structured loggers, and checkpoint management.
- **Pre-commit Quality Checks**: Automated linting and formatting via Ruff, Black, and isort.

---

## Repository Structure

```text
sem-image-restoration-nafnet/
│
├── .github/                  # Issue templates, PR templates, and CI workflows
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── pull_request_template.md
│   └── workflows/
├── assets/                   # Architecture diagrams, flowcharts, figures, and logos
│   ├── architecture/
│   ├── diagrams/
│   ├── figures/
│   └── logo/
├── configs/                  # Hierarchical YAML configuration files
│   ├── default.yaml          # Base configuration defaults
│   ├── train.yaml            # Training hyperparameters
│   ├── model.yaml            # NAFNet model architecture definition
│   ├── inference.yaml        # Inference and evaluation settings
│   └── experiments/          # Reproducible experiment configurations
├── datasets/                 # SEM datasets directory (see datasets/README.md)
│   └── README.md
├── docs/                     # Detailed project documentation and research notes
├── experiments/              # Experiment logs, comparison notes, and ablation notes
│   └── README.md
├── logs/                     # Standalone runtime log output directory
├── notebooks/                # Jupyter notebooks for data analysis & visualization
├── outputs/                  # Generated artifacts (checkpoints, predictions, tensorboard)
│   ├── checkpoints/
│   ├── predictions/
│   └── tensorboard/
├── results/                  # Publication-ready figures and evaluation metrics
│   ├── images/
│   └── tables/
├── scripts/                  # Standalone shell and utility scripts
├── src/                      # Core python package source
│   ├── datasets/             # PyTorch Dataset definitions & preprocessing loaders
│   ├── engine/               # Trainer, Evaluator, and execution engine
│   ├── losses/               # Custom and standard loss functions (PSNR, L1, Charbonnier)
│   ├── metrics/              # Quantitative metrics calculator (PSNR, SSIM)
│   ├── models/               # NAFNet architecture modules and blocks
│   └── utils/                # Helper utilities, logger, config parsers, seeding
├── tests/                    # Pytest unit & integration test suite
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_model.py
├── weights/                  # Pretrained and fine-tuned model checkpoint storage
│   ├── pretrained/
│   └── finetuned/
│
├── .env.example              # Template environment configuration file
├── .gitignore                # Comprehensive git ignore file
├── .pre-commit-config.yaml   # Pre-commit hook definitions
├── CHANGELOG.md              # Project history and milestone release log
├── CONTRIBUTING.md           # Development guidelines and standards
├── LICENSE                   # Open-source MIT License
├── README.md                 # Project main documentation
├── VERSION                   # Semantic version indicator (0.1.0)
├── pyproject.toml            # Python tool configurations (Ruff, Black, Pytest)
├── requirements.txt          # Core unpinned runtime dependencies
├── requirements-dev.txt      # Development & testing dependencies
└── train.py                  # Entry-point script for training and evaluation
```

---

## Installation & Setup

### Prerequisites

* Python 3.11+
* CUDA-capable GPU (Recommended for training)
* Git

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harshwardhan1507/AI-SEM-Image-Restoration.git
   cd AI-SEM-Image-Restoration
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Initialize pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```

---

## Dataset Placeholder & Characterization

SEM datasets are stored under the `datasets/` directory (typically formatted as `.npy` array files or high-bit depth images). 

Refer to [`datasets/README.md`](datasets/README.md) for full instructions regarding dataset directory layout, source links, download steps, and preprocessing expectations.

> **Note**: Raw datasets and binary `.npy` files are excluded from Git via `.gitignore`.

---

## Planned Architecture

This project leverages **NAFNet** (*Nonlinear Activation Free Network for Image Restoration*). Key architectural highlights include:

1. **SimpleGate**: Replaces standard activation functions (GELU, ReLU) by splitting channel dimensions into two paths and multiplying element-wise:
   $$\text{SimpleGate}(X_1, X_2) = X_1 \odot X_2$$
2. **Simplified Channel Attention (SCA)**: Aggregates global context via global average pooling followed by channel-wise scaling without multi-layer perceptron (MLP) bottlenecks.
3. **Multi-Scale Encoder-Decoder**: U-Net style hierarchical feature processing with skip connections for high-frequency detail preservation.

---

## Documentation Standards

All Python code contributed to this repository must follow strict standards:

- **Type Hints**: Complete type annotations on all function parameters and return types.
- **Google-Style Docstrings**: Comprehensive module, class, and method docstrings.
- **Absolute Imports**: All package imports must use absolute paths rooted in `src/` (e.g., `from src.models.nafnet import NAFNet`).
- **No Wildcard Imports**: Explicit imports only (`from module import symbol`).
- **Modular & Config-Driven**: Pure functions and decoupled components configured via YAML schemas.
- **No Hardcoded Paths**: Dynamic path resolution using environment variables or configuration files.
- **No Global Mutable State**: Deterministic executions using explicit seed management.

---

## Coding Standards

- **PEP8 Compliance**: Checked via `ruff check .`.
- **Code Formatting**: Formatted via `black --line-length 88 .`.
- **Import Ordering**: Managed via `isort --profile black .`.
- **Testing**: Managed via `pytest`.

---

## Future Modules (Roadmap)

The upcoming implementation phase will introduce:

1. **Dataset Loader**: Custom PyTorch `Dataset` for `.npy` SEM image pairs (degraded vs. ground truth).
2. **Data Augmentation**: `albumentations`-based spatial & intensity transformations.
3. **NAFNet Architecture**: Modular NAFBlock, SimpleGate, SCA, and U-Net encoder-decoder implementation.
4. **Training Engine**: Decoupled trainer class supporting mixed precision (AMP) and distributed training.
5. **Validation Engine**: Automated evaluation loop logging PSNR and SSIM metrics.
6. **Inference Pipeline**: Tiled & sliding-window inference pipeline for high-resolution SEM micrographs.
7. **Metrics Module**: PyTorch & SciPy PSNR and SSIM implementations.
8. **Visualization Tools**: Visual grid generators comparing degraded input, restored output, and ground truth.
9. **Checkpoint Manager**: Model checkpoint save/resume logic with best-metric tracking.
10. **Logger**: Dual console and TensorBoard logging system.
11. **Utilities**: Seed setup, device auto-detection, and memory usage profiling.
12. **Configuration Manager**: YAML parser with schema validation and command-line overrides.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this repository or NAFNet baseline in your research, please cite the original NAFNet paper:

```bibtex
@inproceedings{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={17--33},
  year={2022},
  organization={Springer}
}
```

---

## Acknowledgements

- **NAFNet Authors**: For demonstrating that simple nonlinear-activation-free baselines can achieve state-of-the-art image restoration performance.
- PyTorch and Open-Source Computer Vision Community.