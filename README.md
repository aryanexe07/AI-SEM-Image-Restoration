# AI-Based Restoration of Degraded Scanning Electron Microscope (SEM) Images using NAFNet

An end-to-end research framework and production-grade implementation of the Nonlinear Activation Free Network (NAFNet) architecture for restoring degraded Scanning Electron Microscope (SEM) images in semiconductor defect inspection pipelines.

---

## 1. Project Title

**AI-Based Restoration of Degraded Scanning Electron Microscope (SEM) Images using NAFNet**

This project provides a modular, configuration-driven PyTorch framework for restoring low-dose, noise-degraded Scanning Electron Microscope (SEM) micrographs while preserving critical nanometer-scale structural fidelity.

---

## 2. Introduction

Scanning Electron Microscopy (SEM) is an indispensable imaging modality in modern semiconductor fabrication and material science. Unlike optical microscopes limited by diffraction limits, SEM utilizes focused electron beams to achieve sub-nanometer spatial resolution. In semiconductor manufacturing, SEM imaging is employed for Critical Dimension SEM (CD-SEM) metrology, overlay measurements, defect review, and reverse engineering of integrated circuits.

However, electron beam interaction with sensitive semiconductor substrate samples presents a fundamental physics trade-off:
1. **High Electron Dose**: Yields high signal-to-noise ratio (SNR) images but risks beam damage, hydrocarbon contamination, charging artifacts, and sample modification.
2. **Low Electron Dose**: Protects delicate semiconductor structures and enables fast scanning speeds, but results in severe image degradation dominated by shot noise and thermal detector fluctuations.

Automated image restoration algorithms capable of reconstructing clean, high-SNR micrographs from low-dose SEM scans are essential for maintaining high inspection throughput while preventing sample damage.

---

## 3. Problem Statement

Semiconductor defect inspection relies on precise edge detection and nanometer-scale pattern recognition. Low-dose SEM image acquisition introduces severe noise phenomena that degrade image quality:

* **Poisson Shot Noise**: Originates from the quantum nature of primary electron arrival and secondary electron emission at low beam currents.
* **Gaussian Thermal Noise**: Introduced by analog signal amplification electronics and detector hardware.
* **Charging Artifacts**: Non-conductive semiconductor surfaces accumulate electrical charge, causing local beam deflection, brightness distortion, and line jitter.
* **Scan Drift and Blur**: Mechanical stage instability and thermal drift induce spatial blurring during frame integration.

### Consequences on Semiconductor Metrology
Severe image degradation impairs critical automated tasks:
* CD-SEM line-edge roughness (LER) and line-width roughness (LWR) algorithms fail due to noisy edge boundaries.
* Sub-10nm defect detection algorithms produce high false-positive or false-negative rates.
* Multi-frame averaging reduces fabrication line throughput.

An effective deep-learning restoration model must remove complex mixed noise while strictly preserving structural edges without introducing hallucinated artifacts.

---

## 4. Project Objectives

### Primary Objective
To construct a research-grade, reproducible PyTorch framework implementing the NAFNet (Nonlinear Activation Free Network) architecture to restore degraded SEM images, maximizing quantitative image quality measured by Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

### Secondary Objectives
1. **Nanometer Edge Preservation**: Retain sharp line edges, contact hole boundaries, and fine pattern topographies essential for semiconductor CD metrology.
2. **Computational Efficiency**: Achieve low computational complexity and memory footprint by leveraging activation-free operations.
3. **Modular Codebase**: Provide a clean separation between dataset handling, model architecture, training engine, evaluation metrics, and configuration management.
4. **Reproducible Experimentation**: Ensure deterministic training execution, explicit seed control, and structured metric logging.

### Expected Outcomes
* A trained NAFNet model outperforming traditional spatial filters (Gaussian, Median, Non-Local Means) and conventional CNN baselines on SEM benchmarks.
* A standardized evaluation benchmark comparing restored SEM micrographs against ground-truth high-dose acquisitions.

---

## 5. Why NAFNet?

Image restoration has historically relied on deep Convolutional Neural Networks (CNNs) such as DnCNN or Vision Transformers (ViTs) such as SwinIR and Restormer. While Vision Transformers achieve competitive PSNR metrics, their self-attention mechanisms impose high computational complexity $O(H^2 W^2)$ and heavy memory demands.

### Technical Justification for NAFNet
NAFNet (Nonlinear Activation Free Network) demonstrates that conventional nonlinear activation functions (e.g., GELU, ReLU, Softmax) are not strictly necessary for achieving state-of-the-art restoration performance. Key architectural advantages include:

1. **SimpleGate Mechanism**: Replaces standard activation functions by splitting channel dimensions in half and computing an element-wise multiplication ($X_1 \odot X_2$), providing non-linear modeling capability with lower computational overhead.
2. **Simplified Channel Attention (SCA)**: Replaces complex multi-layer perceptron (MLP) channel attention blocks with a single global average pooling operation followed by channel-wise scaling, capturing global context efficiently.
3. **Residual Learning**: Employs intra-block and cross-stage residual skip connections, allowing the network to focus exclusively on learning the residual noise map ($I_{\text{degraded}} - I_{\text{clean}}$).
4. **Sub-Linear Computational Complexity**: Maintains spatial convolution operations ($O(HW)$), making it computationally suited for high-resolution semiconductor image patches.

### Comparative Architectural Analysis

| Architecture | Paradigm | Activation Function | Attention Mechanism | Computational Complexity | Parameter Efficiency | SEM Suitability |
|---|---|---|---|---|---|---|
| **U-Net** | Encoder-Decoder CNN | ReLU / LeakyReLU | None | Low | Moderate | Moderate (lacks global context) |
| **DnCNN** | Feed-forward CNN | ReLU | None | Low | High | Poor for complex SEM noise |
| **RIDNet** | Residual CNN | ReLU | Feature Attention | Moderate | Moderate | Moderate |
| **Restormer** | Transformer | GELU | Multi-Dhead Transposed Attention | High | Low | High (expensive computation) |
| **SwinIR** | Swin Transformer | GELU | Windowed Self-Attention | High | Low | High (expensive computation) |
| **NAFNet (Ours)** | Activation-Free CNN | SimpleGate (None) | Simplified Channel Attention (SCA) | Low-Moderate | High | Optimal (high fidelity, efficient) |

---

## 6. Dataset

### Overview & Data Storage
The project utilizes paired SEM dataset splits containing low-dose (noisy/degraded) micrographs alongside corresponding high-dose (clean/ground-truth) acquisitions.

### Technical Dataset Specifications
* **File Format**: NumPy binary arrays (`.npy`) allowing 32-bit floating-point precision preservation without lossy compression.
* **Channel Configuration**: Single-channel grayscale micrographs ($H \times W$ or $1 \times H \times W$).
* **Data Organization**:
  ```text
  datasets/
  ├── train/
  │   ├── degraded/     # Low-dose SEM arrays (*.npy)
  │   └── clean/        # High-dose ground-truth arrays (*.npy)
  ├── val/
  │   ├── degraded/
  │   └── clean/
  └── test/
      ├── degraded/
      └── clean/
  ```

### Preprocessing & Normalization Protocol
1. **Characterization Phase**: Statistical profiling of dataset intensity range, mean, variance, and spatial dimension consistency prior to training.
2. **Dynamic Range Normalization**: Scaling raw sensor intensity values to a standardized $[0.0, 1.0]$ floating-point range.
3. **Paired Validation**: Rigorous verification ensuring strict alignment between noisy input arrays and corresponding ground-truth reference targets.

---

## 7. Repository Structure

The codebase is organized as a modular Python package located in `src/`, adhering to modern packaging standards:

```text
sem-image-restoration-nafnet/
├── .github/             # GitHub templates and continuous integration workflows
├── assets/              # Architecture diagrams, flowcharts, and figures
├── configs/             # YAML configurations (default, train, model, inference, experiments)
├── datasets/            # Dataset split storage and dataset documentation
├── docs/                # Research papers, technical notes, and design documents
├── experiments/         # Experiment execution logs and ablation comparative studies
├── logs/                # Standalone application execution logs
├── notebooks/           # Jupyter notebooks for data characterization and visualization
├── outputs/             # Runtime artifacts (checkpoints, predictions, tensorboard logs)
├── results/             # Publication figures and quantitative metric tables
├── scripts/             # Standalone shell utilities and entry helpers
├── src/                 # Main Python package source code
│   ├── datasets/        # Dataset classes, PyTorch DataLoaders, and transforms
│   ├── engine/          # Trainer, Evaluator, and execution loops
│   ├── losses/          # Loss functions (PSNR Loss, Charbonnier, L1)
│   ├── metrics/         # PSNR and SSIM metric calculators
│   ├── models/          # NAFNet model architecture implementation
│   └── utils/           # Configuration parsers, random seeds, and logging helpers
├── tests/               # Pytest unit and integration test modules
├── weights/             # Pretrained and fine-tuned model checkpoint files
├── .env.example         # Template environment variables configuration
├── .gitignore           # Comprehensive Git ignore rules
├── .pre-commit-config.yaml # Pre-commit formatting and linting hook definitions
├── CHANGELOG.md         # Version release history and milestone tracking
├── CONTRIBUTING.md      # Development workflow and coding guidelines
├── LICENSE              # Open-source MIT License
├── README.md            # Master repository specification and technical documentation
├── VERSION              # Semantic version tracking file (0.1.0)
├── pyproject.toml       # Build system and Python tool configurations (Ruff, Black, Pytest)
├── requirements.txt     # Runtime dependencies definition
├── requirements-dev.txt # Development dependencies definition
└── train.py             # Root execution entry-point script
```

### Module Responsibilities
* `src/datasets/`: Custom PyTorch `Dataset` implementations handling `.npy` file loading, patch cropping, and data augmentations.
* `src/models/`: Modular NAFNet neural network blocks, including `SimpleGate`, `SCA`, `NAFBlock`, and top-level encoder-decoder structures.
* `src/engine/`: Training execution engine managing epoch loops, mixed-precision scaling, validation evaluations, and checkpoint persistence.
* `src/losses/`: Differentiable objective functions tailored for image restoration.
* `src/metrics/`: Full-reference image quality metric algorithms (PSNR, SSIM).
* `src/utils/`: Infrastructure utilities for logging, seeding, and configuration parsing.

---

## 8. System Architecture

The high-level restoration pipeline processes degraded SEM micrographs through a deterministic, end-to-end data flow:

```text
+-----------------------+     +------------------------+     +-------------------------+
| Degraded SEM Input    | --> | Preprocessing &        | --> | PyTorch DataLoader      |
| (.npy array)          |     | Normalization          |     | (Batched Patches)       |
+-----------------------+     +------------------------+     +-------------------------+
                                                                          |
                                                                          v
+-----------------------+     +------------------------+     +-------------------------+
| Restored Output       | <-- | Post-Processing &      | <-- | NAFNet Model            |
| Micrograph            |     | Range Denormalization  |     | (Encoder-Decoder)       |
+-----------------------+     +------------------------+     +-------------------------+
```

### Operational Pipeline Flow
1. **Input Stage**: Raw degraded SEM array files (`.npy`) are loaded from disk.
2. **Preprocessing**: Intensity ranges are normalized to $[0.0, 1.0]$ and split into spatial sub-patches during training.
3. **DataLoader Stage**: Parallel workers construct mini-batches with applied spatial augmentations.
4. **NAFNet Processing**: Feature extraction, multi-scale encoding, residual attention processing, and multi-scale decoding reconstruct the clean residual feature map.
5. **Post-Processing Stage**: Output feature tensors are converted back to normalized arrays, denormalized, and saved alongside calculated evaluation metrics.

---

## 9. Data Processing Pipeline

The data processing pipeline prepares low-dose SEM micrographs for model consumption through seven sequential stages:

1. **Array Loading**: Direct memory-mapped loading of `.npy` arrays using `numpy.load`.
2. **Validation**: Assertion checks verifying shape consistency, non-empty content, and matching input/target spatial dimensions.
3. **Intensity Normalization**: Scaling raw pixel intensities to float32 values bounded in $[0.0, 1.0]$.
4. **Patch Extraction**: Random cropping of fixed-size sub-patches (e.g., $128 \times 128$ or $256 \times 256$) to enforce uniform tensor dimensions and increase training sample efficiency.
5. **Data Augmentation**: Applying geometry-preserving spatial transformations (random horizontal flips, vertical flips, $90^\circ$ rotations) using `albumentations`.
6. **Tensor Conversion**: Converting NumPy arrays into PyTorch floating-point tensors (`torch.FloatTensor`) formatted as $(C, H, W)$.
7. **Batch Assembly**: Collating individual patch tensors into mini-batches $(B, C, H, W)$ for GPU transfer.

---

## 10. Model Architecture

NAFNet operates as an encoder-decoder network enhanced with cross-stage skip connections and activation-free residual blocks.

```text
Input (C, H, W)
      |
[ Head Conv3x3 ]
      |
[ Encoder Stage 1 ] ------ (Skip Connection) -------> [ Decoder Stage 1 ] ---> [ Tail Conv3x3 ] ---> Output (C, H, W)
      |                                                     ^
 [ Downsample ]                                        [ Upsample ]
      |                                                     |
[ Encoder Stage 2 ] ------ (Skip Connection) -------> [ Decoder Stage 2 ]
      |                                                     ^
 [ Downsample ]                                        [ Upsample ]
      \------------------ [ Middle Block ] ----------------/
```

### Key Architectural Components

1. **Head Convolution**: Maps input image channels $C$ (1 for grayscale SEM) into an initial feature dimension $N$ (width parameter).
2. **NAFBlock**: The foundational building block containing:
   * **Depthwise Convolution**: $3 \times 3$ depthwise convolution for spatial context extraction.
   * **SimpleGate**: Splits feature channels in half ($2C \to C$) and applies element-wise multiplication, replacing conventional non-linear activations.
   * **Simplified Channel Attention (SCA)**: Computes global channel statistics via mean pooling and scales channel features directly.
   * **Layer Normalization**: Applied before gating and attention blocks to stabilize optimization.
3. **Encoder**: Hierarchical stages of NAFBlocks followed by strided convolution downsampling operations, doubling channel capacity while halving spatial resolution.
4. **Middle Block**: Deep sequence of NAFBlocks operating at the bottleneck resolution to model global spatial context.
5. **Decoder**: Hierarchical stages incorporating transposed convolution upsampling, feature concatenation via skip connections, and NAFBlock refinement.
6. **Tail Reconstruction**: $3 \times 3$ convolution mapping bottleneck features back to $C$ output channels, added back to the original input via long residual skip connection ($I_{\text{restored}} = I_{\text{degraded}} + R_{\text{pred}}$).

---

## 11. Training Pipeline

The training pipeline defines the complete optimization protocol for model convergence:

* **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-3}$).
* **Learning Rate Scheduler**: Cosine Annealing Learning Rate schedule with linear warmup epochs.
* **Loss Functions**: Primary reconstruction loss using **PSNR Loss** or **Charbonnier Loss** ($\sqrt{\|I_{\text{pred}} - I_{\text{gt}}\|^2 + \epsilon^2}$ with $\epsilon=10^{-3}$), which provides smoother gradient optimization around zero error compared to standard $L_1$ loss.
* **Mixed Precision (AMP)**: PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast`) using FP16/BF16 to accelerate training speed and reduce GPU memory consumption.
* **Validation Protocol**: Epoch-wise evaluation computing PSNR and SSIM on validation splits.
* **Checkpointing**: Automatic tracking and saving of `best_model.pth` based on peak validation PSNR, alongside regular periodic epoch state saves.
* **Logging**: Dual-channel output writing formatted text logs to `logs/` and quantitative scalar curves to `outputs/tensorboard/`.
* **Early Stopping**: Monitored validation PSNR plateau detection to terminate non-converging runs.

---

## 12. Evaluation Strategy

Restoration fidelity on SEM micrographs is evaluated using both quantitative full-reference metrics and qualitative visual inspection:

### Quantitative Metrics

1. **Peak Signal-to-Noise Ratio (PSNR)**:
   Measures logarithmic mean squared error (MSE) relative to maximum signal value ($MAX_I = 1.0$):
   $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{\text{MSE}} \right)$$
   *Higher values indicate superior noise reduction.*

2. **Structural Similarity Index Measure (SSIM)**:
   Evaluates structural fidelity across luminance ($\text{l}$), contrast ($\text{c}$), and structure ($\text{s}$) components:
   $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
   *Essential for verifying that nanometer-scale line edges and contact hole structures are preserved without distortion.*

### Qualitative & Runtime Evaluation
* **Difference Maps**: Computing absolute error residual maps ($|I_{\text{restored}} - I_{\text{gt}}|$) to visually inspect structural artifact distribution.
* **Line-Edge Profile Analysis**: Plotting 1D intensity cross-sections across line boundaries to verify edge sharpness preservation.
* **Inference Latency**: Measuring millisecond execution times per patch to evaluate suitability for inline semiconductor inspection tools.

---

## 13. Configuration System

The project utilizes a hierarchical YAML configuration system located in `configs/`:

* **`default.yaml`**: Defines base execution environment parameters (CUDA device, random seeds, worker counts, default logging directory paths).
* **`train.yaml`**: Specifies optimization settings (learning rate, batch size, epoch bounds, loss function parameters, scheduler steps).
* **`model.yaml`**: Defines NAFNet architecture hyperparameters (channel width, block counts per encoder/decoder stage, attention flags).
* **`inference.yaml`**: Specifies evaluation settings (checkpoint path, input directory, patch tiling parameters, output destination).
* **`configs/experiments/`**: Standalone experiment override files (e.g., `exp001.yaml`) enabling full experiment reproducibility without mutating base configuration files.

---

## 14. Development Workflow

The repository enforces modern software engineering practices:

### Coding Standards
* **Python Version**: Python 3.11+.
* **Code Style**: Strictly compliant with PEP8, enforced via `black --line-length 88`.
* **Linting**: Static code analysis via `ruff`.
* **Import Sorting**: Standardized grouping via `isort --profile black`.
* **Type Annotations**: Comprehensive static type hints across all function parameters and return values.
* **Docstrings**: Google-style docstrings for all modules, classes, and methods.

### Quality Assurance & Testing
* Automated pre-commit hooks executing formatters and linters on git commit (`.pre-commit-config.yaml`).
* Modular unit and integration tests executing under `pytest`.

---

## 15. Experiment Tracking

To ensure rigorous scientific reproducibility:

1. **Experiment Isolation**: Every experiment run is assigned a unique identifier (e.g., `exp001_nafnet_baseline`).
2. **Configuration Snapshots**: The exact resolved YAML configuration is copied into the experiment output folder at runtime start.
3. **Checkpoint Retention**: Model weights are stored in `outputs/checkpoints/` alongside optimizer states and current epoch counts.
4. **Metric Logging**: TensorBoard event logs track training loss, validation PSNR, validation SSIM, and learning rate curves over time.
5. **Result Summaries**: Summary evaluation tables are stored in Markdown and CSV formats under `results/tables/`.

---

## 16. Current Project Roadmap

- [x] **Phase 1: Repository Foundation**
  - [x] Establish directory structure (`src/` package layout, `configs/`, `tests/`, `assets/`, `weights/`).
  - [x] Create project setup files (`pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `.pre-commit-config.yaml`).
  - [x] Create master research-grade `README.md` specification.
  - [x] Initialize Git version control and remote synchronization.

- [ ] **Phase 2: Dataset Characterization & Preprocessing**
  - [ ] Implement exploratory data analysis scripts for `.npy` SEM arrays in `notebooks/`.
  - [ ] Compute dataset-wide intensity statistics (min, max, mean, std, variance).
  - [ ] Define dataset split boundaries (train / validation / test).

- [ ] **Phase 3: Dataset Loader & Augmentation Pipeline**
  - [ ] Implement `src/datasets/sem_dataset.py` for `.npy` image pair loading.
  - [ ] Build random sub-patch extraction logic.
  - [ ] Integrate spatial augmentations via `albumentations`.
  - [ ] Write unit tests for dataset loading in `tests/test_dataset.py`.

- [ ] **Phase 4: NAFNet Model Implementation**
  - [ ] Implement `SimpleGate` and `Simplified Channel Attention (SCA)` modules in `src/models/nafnet.py`.
  - [ ] Build `NAFBlock` and hierarchical Encoder-Decoder network.
  - [ ] Write unit tests verifying input/output tensor shape consistency in `tests/test_model.py`.

- [ ] **Phase 5: Loss Functions & Evaluation Metrics**
  - [ ] Implement `PSNR` and `SSIM` calculation modules in `src/metrics/`.
  - [ ] Implement `CharbonnierLoss` and `PSNRLoss` in `src/losses/`.
  - [ ] Write unit tests for metrics in `tests/test_metrics.py`.

- [ ] **Phase 6: Training & Validation Engine**
  - [ ] Implement modular `Trainer` class in `src/engine/trainer.py`.
  - [ ] Integrate Automatic Mixed Precision (AMP) and Cosine Annealing scheduler.
  - [ ] Implement checkpoint saving and TensorBoard logging.

- [ ] **Phase 7: Inference Pipeline & Visualization**
  - [ ] Implement sliding-window / patch-tiling inference pipeline for full-resolution micrographs in `src/engine/evaluator.py`.
  - [ ] Build visual grid generators (Degraded Input vs. Restored Output vs. Ground Truth).

- [ ] **Phase 8: Benchmarking & Reporting**
  - [ ] Execute comparative ablation experiments.
  - [ ] Populate quantitative result tables under `results/tables/`.
  - [ ] Generate final technical research report.

---

## 17. Future Enhancements

* **Self-Supervised & Blind Denoising**: Explore Noise2Noise or Neighbor2Neighbor approaches to remove the requirement for paired high-dose ground-truth images.
* **Mixed Noise Modeling**: Incorporate physics-informed noise modeling specific to electron beam charging and scan line jitter.
* **Model Compression & Quantization**: Apply post-training quantization (INT8/FP16) or structured pruning to reduce memory footprint.
* **TensorRT / ONNX Acceleration**: Export trained NAFNet models to ONNX and TensorRT for real-time inline deployment on semiconductor inspection hardware.
* **Transformer Hybrid Architectures**: Evaluate hybrid activation-free transformer blocks for long-range spatial artifact modeling.

---

## 18. References

1. **NAFNet**: Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). *Simple Baselines for Image Restoration*. European Conference on Computer Vision (ECCV). [arXiv:2204.04676](https://arxiv.org/abs/2204.04676)
2. **DnCNN**: Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*. IEEE Transactions on Image Processing, 26(7), 3142-3155.
3. **Restormer**: Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., & Yang, M. H. (2022). *Restormer: Efficient Transformer for High-Resolution Image Restoration*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
4. **SwinIR**: Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). *SwinIR: Image Restoration Using Swin Transformer*. IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
5. **RIDNet**: Anwar, S., & Barnes, N. (2019). *Real Image Denoising With Feature Attention*. IEEE/CVF International Conference on Computer Vision (ICCV).

---

## 19. License

This repository is distributed under the open-source **MIT License**. See the [`LICENSE`](LICENSE) file for full license terms and conditions.

---

## 20. Acknowledgements

* **NAFNet Authors**: For introducing activation-free baseline architectures for image restoration.
* **PyTorch & Open-Source Vision Community**: For providing core machine learning frameworks, computer vision primitives, and performance optimization tools.