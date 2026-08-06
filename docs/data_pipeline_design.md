# Data Pipeline Design & Architecture Specification

**Project Title**: AI-Based Restoration of Degraded Scanning Electron Microscope (SEM) Images using NAFNet  
**Target Architecture**: PyTorch Data Pipeline Infrastructure  
**Document Status**: Official Engineering Specification  
**Reference Document**: [Dataset Characterization Report](dataset_characterization.md)  

---

## 1. Purpose

The data pipeline serves as the primary abstraction layer connecting raw, disk-bound SEM image data to the PyTorch neural network training environment. In deep learning for computer vision—specifically image restoration and super-resolution—the throughput, numerical correctness, and deterministic behavior of the data pipeline directly dictate model convergence, training stability, and research reproducibility.

### Key Engineering Objectives
* **Data Integrity**: Guarantee that only valid, non-corrupt, and strictly paired low-resolution noisy (`NoisyLR`) and high-resolution ground-truth (`GT`) micrographs enter the training graph.
* **Reproducibility**: Provide deterministic data indexing, paired spatial augmentations, and patch extraction governed by centralized random seed management.
* **Scalability & Throughput**: Maximize GPU utilization by minimizing disk I/O bottlenecks through efficient memory mapping, parallel multi-process loading, and non-blocking host-to-device memory transfers.
* **Flexibility & Configuration-Driven Behavior**: Decouple data loading logic from hyperparameter choices (patch sizes, batch sizes, normalizations) via a centralized YAML configuration system.

---

## 2. Overall Pipeline Architecture

The data pipeline follows a unidirectional processing DAG (Directed Acyclic Graph) structured into distinct stages:

```text
                               +-----------------------------+
                               | External Dataset Storage    |
                               | (D:/Programming/python/...) |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 1. Directory Scanner        |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 2. Pair & Path Validator    |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 3. Dataset Index Builder    |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 4. PyTorch Dataset Class    |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 5. Array Loader & Preproc   |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 6. Spatial Augmentation     |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 7. Patch Extractor          |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 8. Tensor Formatting        |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 9. PyTorch DataLoader       |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               | 10. NAFNet Model Input      |
                               +-----------------------------+
```

### Functional Component Responsibilities

1. **Directory Scanner**: Recursively traverses the raw dataset directory hierarchy to discover available image files while filtering OS artifacts (e.g. macOS `__MACOSX` folders and `._*.npy` resource forks).
2. **Pair & Path Validator**: Verifies that every degraded image has a corresponding ground-truth target file with identical filename stems and matching extensions.
3. **Dataset Index Builder**: Constructs an in-memory index mapping sample indices ($0, 1, \dots, N-1$) to validated tuple pairs `(noisy_path, gt_path)`.
4. **PyTorch Dataset Class**: Extends `torch.utils.data.Dataset` to handle sample indexing, loading execution, and transform invocation.
5. **Array Loader & Preprocessor**: Loads NumPy binary arrays via memory mapping, casts data types to `float32`, enforces 2D/3D channel shapes, and clips intensity outliers.
6. **Spatial Augmentation**: Applies paired geometric transformations (flips, rotations) identically to both input and target arrays.
7. **Patch Extractor**: Extracts spatially aligned sub-patches ($128 \times 128$ for input, $256 \times 256$ for GT) to enable batched training.
8. **Tensor Formatter**: Adds channel dimensions ($1 \times H \times W$), formats arrays to C-contiguous memory layout, and converts to `torch.FloatTensor`.
9. **PyTorch DataLoader**: Assembles individual tensor samples into mini-batches, handles worker process synchronization, and pins host memory.
10. **NAFNet Interface**: Yields formatted mini-batch dictionaries onto GPU hardware for forward pass execution.

---

## 3. Dataset Organization

Based on measurements established in the [Dataset Characterization Report](dataset_characterization.md), the external dataset directory is structured as follows:

```text
semicondata/
├── train/
│   └── train/
│       ├── GT/         # 3,200 GT files (256 x 256 float32, range [0.0, 1.0])
│       └── NoisyLR/    # 3,200 NoisyLR files (128 x 128 float32, range [-0.278, 1.938])
└── Test_NoisyLR/
    └── NoisyLR/        # 400 NoisyLR test files (128 x 128 float32)
```

### Key Dataset Characteristics & Design Constraints
* **Task Type**: Simultaneous **$2\times$ Spatial Super-Resolution** and **Denoising**.
* **Input Resolution**: Low-Resolution Noisy (`NoisyLR`) micrographs at $128 \times 128$ pixels.
* **Target Resolution**: High-Resolution Ground-Truth (`GT`) micrographs at $256 \times 256$ pixels.
* **File Format**: Uncompressed 32-bit floating-point NumPy arrays (`.npy`).
* **Pair Matching Rule**: Files in `GT/` and `NoisyLR/` share identical numeric filenames (`000000.npy` through `003199.npy`).

---

## 4. Dataset Indexing Strategy

To eliminate runtime filesystem scanning delays during training, the pipeline implements an **eager index building strategy** during initialization.

```text
Filesystem Scanning ──> Exclude `__MACOSX` & `._*` ──> Match Filenames ──> Memory Index List [(N_0, G_0), ...]
```

### Indexing Algorithm Protocol
1. **Directory Discovery**: Discover paths for `train/train/GT`, `train/train/NoisyLR`, and `Test_NoisyLR/NoisyLR`.
2. **Resource Fork Exclusion**: Explicitly reject paths matching:
   * Any directory containing `__MACOSX` in its path.
   * Any filename starting with `._` (AppleDouble metadata files).
3. **Filename Pair Alignment**: Build a key-value dictionary keyed by file stem (e.g. `"000000"`). Intersect keys between `GT` and `NoisyLR` folders to create a validated index list.
4. **Validation Check**: If $|Keys_{GT} \cap Keys_{Noisy}| \neq |Keys_{GT}|$, issue a descriptive warning detailing missing files and exclude unpaired items from the active index.

---

## 5. Dataset API Design

The custom `SEMDataset` class implements PyTorch's standard `__len__` and `__getitem__` interface, returning a structured sample dictionary:

```text
+-----------------------------------------------------------------------+
| Sample Dictionary Output (__getitem__)                                |
+-----------------------------------------------------------------------+
| Key         | Type               | Shape         | Data Type | Range  |
+-------------+--------------------+---------------+-----------+--------+
| "input"     | torch.FloatTensor  | (1, 128, 128) | float32   | [0, 1] |
| "target"    | torch.FloatTensor  | (1, 256, 256) | float32   | [0, 1] |
| "filename"  | str                | N/A           | string    | N/A    |
| "metadata"  | dict               | N/A           | dict      | N/A    |
+-----------------------------------------------------------------------+
```

### Output Field Descriptions
* **`input`**: The low-resolution degraded SEM micrograph tensor formatted as $(C, H_{LR}, W_{LR})$. Default shape: $(1, 128, 128)$.
* **`target`**: The high-resolution ground-truth SEM micrograph tensor formatted as $(C, H_{HR}, W_{HR})$. Default shape: $(1, 256, 256)$.
* **`filename`**: The basename of the sample (e.g., `"000123.npy"`), essential for tracking visual outputs, logging error residuals, and saving prediction files.
* **`metadata`**: A dictionary containing sample attributes including original array dynamic range, applied patch coordinates, and split designation.

---

## 6. Validation Pipeline

The pipeline enforces validation at two distinct execution checkpoints:

### Checkpoint A: Offline / Initialization Validation
* **Path Verification**: Confirm dataset root and subfolder paths exist on host filesystem.
* **Pair Consistency**: Verify $100\%$ matching of file basenames across `GT` and `NoisyLR` directories.
* **File Exclusion Verification**: Ensure hidden OS files (`.DS_Store`, `._*.npy`) are excluded from index.

### Checkpoint B: Runtime / Sample Load Validation
* **NaN / Inf Sanity Check**: Inspect loaded array elements for non-finite values (`np.isnan`, `np.isinf`).
* **Dimensionality Check**: Assert loaded array matches expected 2D shape ($(128, 128)$ for input, $(256, 256)$ for target).
* **Data Type Assertion**: Verify array data type is `float32`.

---

## 7. Preprocessing Pipeline

The preprocessing pipeline converts raw array values into normalized, numerically stable model inputs through five sequential operations:

```text
Raw .npy File ──> mmap Read ──> Float32 Cast ──> Range Clipping [0,1] ──> Channel Expansion (1,H,W)
```

### Preprocessing Operations Specification

| Step | Operation | Input Format | Output Format | Justification |
|---|---|---|---|---|
| 1 | **Memory-Mapped Load** | Disk `.npy` file | NumPy `ndarray` | Fast disk read without full RAM duplication (`mmap_mode='r'`). |
| 2 | **Type & Copy Enforce** | Read-only array | Contiguous `float32` | Ensures memory contiguity for PyTorch C++ backend. |
| 3 | **Intensity Clipping** | Out-of-range floats | Bounded $[0.0, 1.0]$ | Removes physical noise outliers ($< -0.27$ and $> 1.93$) preventing gradient explosion. |
| 4 | **Channel Addition** | 2D $(H, W)$ | 3D $(1, H, W)$ | Standard PyTorch 2D image format requiring explicit channel dimension. |
| 5 | **Spatial Alignment Check** | Input & Target | Matched Ratio ($1:2$) | Confirms target spatial dimensions equal exactly $2\times$ input spatial dimensions. |

---

## 8. Data Augmentation Strategy

Data augmentation increases effective dataset size and prevents neural network overfitting. However, SEM micrographs possess unique physical properties compared to natural RGB photographs.

### Approved Augmentations for SEM Micrographs

```text
                  +-----------------------------------+
                  | Input Pair (NoisyLR & GT Target)  |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
  +------------------+    +------------------+    +-------------------+
  | Random H-Flip    |    | Random V-Flip    |    | Random Rotate 90  |
  | (p = 0.5)        |    | (p = 0.5)        |    | (p = 0.5)         |
  +------------------+    +------------------+    +-------------------+
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Synchronized Transformed Pair     |
                  +-----------------------------------+
```

1. **Horizontal Flip (`p=0.5`)**: Flips both input and target horizontally. Valid because semiconductor structures (lines, spaces, contact holes) exhibit spatial mirror symmetry.
2. **Vertical Flip (`p=0.5`)**: Flips both input and target vertically.
3. **Random Rotation by $90^\circ, 180^\circ, 270^\circ$ (`p=0.5`)**: Rotates arrays orthogonally. Preserves exact pixel grid alignments without interpolation artifacts.

### Explicitly Forbidden Augmentations
* **Color Jitter / Hue / Saturation Adjustments**: SEM images are single-channel grayscale arrays representing electron detector counts. Color augmentations are physically meaningless.
* **Arbitrary Angle Rotations (e.g. $15^\circ$)**: Non-orthogonal rotations require bilinear/bicubic sub-pixel interpolation, altering native noise distributions and blurring sharp semiconductor line edges.
* **Gaussian Blur / Random Noise Injection**: Degraded input images already contain real physical low-dose SEM noise. Adding synthetic noise distorts the true degradation manifold.

### Training vs. Evaluation Pipeline Separation
* **Training Pipeline**: Full stochastic augmentation active (`RandomFlip`, `RandomRotate90`, `RandomCrop`).
* **Validation / Testing Pipeline**: Zero spatial augmentations applied. Entire $128 \times 128$ input evaluated against full $256 \times 256$ target.

---

## 9. Patch Extraction Strategy

Because NAFNet is a fully convolutional neural network, training can be performed on cropped spatial sub-patches rather than full micrographs, significantly reducing GPU VRAM consumption while increasing mini-batch diversity.

### Patch Pair Dimensions & Spatial Scale Alignment
Due to the $2\times$ super-resolution task setup:
* **LR Input Patch Size ($P_{LR}$)**: $128 \times 128$ (Full input image) or $64 \times 64$ sub-patch.
* **HR Target Patch Size ($P_{HR}$)**: $256 \times 256$ (Full target image) or $128 \times 128$ sub-patch.
* **Scale Ratio**: $P_{HR} = 2 \times P_{LR}$.

```text
Low-Res Input Patch (64 x 64)  ====== Scale Factor 2x ======> High-Res Target Patch (128 x 128)
[ Top-Left: (x, y) ]                                         [ Top-Left: (2x, 2y) ]
```

### Coordinates Synchronized Crop Rule
When extracting a random sub-patch starting at top-left coordinate $(y_{LR}, x_{LR})$ on the low-resolution input image, the corresponding target patch top-left coordinate on the high-resolution image **must** be set to $(2 \cdot y_{LR}, 2 \cdot x_{LR})$.

---

## 10. Tensor Conversion

Before mini-batch assembly, NumPy arrays are formatted into PyTorch tensors complying with memory contiguity rules:

1. **Memory Contiguity**: Apply `np.ascontiguousarray()` to ensure stride values match standard C-ordering following spatial augmentations.
2. **PyTorch Tensor Casting**: Construct `torch.from_numpy(array).float()` to prevent data copying overhead where possible.
3. **Non-Blocking Device Transfer**: Set `pin_memory=True` in PyTorch DataLoader to allocate host memory in page-locked (pinned) memory, enabling asynchronous CPU-to-GPU memory transfer via CUDA streams (`tensor.to(device, non_blocking=True)`).

---

## 11. DataLoader Design

The DataLoader design balances CPU pre-fetching performance against host RAM utilization.

### DataLoader Strategy Specifications

| Parameter | Recommended Value | Trade-Off & Technical Rationale |
|---|---|---|
| **`batch_size`** | 16 – 32 | Balance between GPU VRAM capacity and gradient noise stability. |
| **`shuffle`** | `True` (Train) / `False` (Val/Test) | Ensures uniform sample distribution across training epochs. |
| **`num_workers`** | 4 – 8 | Matches CPU physical core count to prevent data starvation on GPU. |
| **`pin_memory`** | `True` | Accelerates host-to-device CUDA memory transfer rates. |
| **`persistent_workers`**| `True` | Keeps dataset worker processes alive between epochs, avoiding teardown/spawn overhead. |
| **`prefetch_factor`** | 2 – 4 | Prefetches 2–4 mini-batches per worker to hide disk I/O latency. |
| **`drop_last`** | `True` (Train) / `False` (Val) | Prevents unstable gradient updates from small residual mini-batches. |

---

## 12. Configuration System

All data pipeline parameters must be declared in YAML configuration files (`configs/default.yaml` and `configs/train.yaml`) rather than hardcoded in source files:

```text
+------------------------------------------------------------------------------------+
| Centralized Configuration Parameter Map                                            |
+------------------------------------------------------------------------------------+
| Section   | Parameter           | Description                       | Example Value|
+-----------+---------------------+-----------------------------------+--------------+
| dataset   | `root_dir`          | Path to external dataset root     | "D:/..."     |
| dataset   | `gt_subpath`        | Relative path to train GT folder  | "train/...GT"|
| dataset   | `noisy_subpath`     | Relative path to train Noisy folder| "train/...LR"|
| dataset   | `scale_factor`      | Spatial super-resolution factor   | 2            |
| preproc   | `clip_min`          | Lower intensity clip bound        | 0.0          |
| preproc   | `clip_max`          | Upper intensity clip bound        | 1.0          |
| preproc   | `patch_size_lr`     | LR spatial patch crop size        | 128          |
| preproc   | `patch_size_hr`     | HR spatial patch crop size        | 256          |
| dataloader| `batch_size`        | Mini-batch sample count           | 16           |
| dataloader| `num_workers`       | Worker process count              | 4            |
| dataloader| `pin_memory`        | Page-locked CUDA memory flag      | true         |
| augment   | `use_hflip`         | Enable horizontal flips           | true         |
| augment   | `use_vflip`         | Enable vertical flips             | true         |
| augment   | `use_rot90`         | Enable orthogonal 90deg rotations | true         |
+------------------------------------------------------------------------------------+
```

---

## 13. Error Handling Strategy

The pipeline implements defensive error-handling routines to ensure robust execution during long-running training tasks:

```text
                              +-------------------------+
                              | Dataset Load Attempt    |
                              +-------------------------+
                                           |
                   +-----------------------+-----------------------+
                   | (Success)                                     | (Failure)
                   v                                               v
        +--------------------+                           +-------------------+
        | Return Sample Dict |                           | Catch Exception   |
        +--------------------+                           +-------------------+
                                                                   |
                                                                   v
                                                         +-------------------+
                                                         | Log Detailed Error|
                                                         | Path & Error Type |
                                                         +-------------------+
                                                                   |
                                                                   v
                                                         +-------------------+
                                                         | Fallback: Load    |
                                                         | Index (i + 1)     |
                                                         +-------------------+
```

### Actionable Recovery Protocols
1. **Missing / Unmatched File**: Exclude during dataset index construction; log warning listing unmatched filename stem.
2. **Corrupted `.npy` Array**: Catch `ValueError` or `OSError` during array loading; log error specifying corrupt file path; return fallback sample index ($i + 1 \pmod N$).
3. **Invalid Array Shape**: If input shape $\neq (128, 128)$ or target shape $\neq (256, 256)$, raise `ValueError` detailing expected vs. actual array dimensions.
4. **NaN / Inf Detection**: If non-finite values are detected post-loading, drop sample and log warning specifying file identifier.

---

## 14. Logging Strategy

Data pipeline events are logged using Python's standard `logging` module configured via `src/utils/logger.py`:

### Key Logging Events & Levels
* **`INFO`**: Dataset discovery summary (total paired files found, split breakdown, indexing duration).
* **`INFO`**: DataLoader initialization settings (`batch_size`, `num_workers`, `patch_size`).
* **`WARNING`**: Skipped files (unmatched stems, excluded macOS `._*` resource forks).
* **`ERROR`**: Runtime read failures, shape mismatches, or numerical NaN detections.

---

## 15. Performance Considerations

To maximize GPU compute efficiency, data loading pipeline execution time must remain strictly smaller than model forward/backward pass execution time ($T_{\text{data}} \ll T_{\text{gpu}}$):

1. **Memory Mapping (`np.load(..., mmap_mode='r')`)**: Avoids reading complete file byte arrays into RAM simultaneously.
2. **Pre-Allocated NumPy Buffers**: Avoids repeated array memory allocations during spatial crops.
3. **Batch Pre-Fetching**: Leverages `prefetch_factor` to ensure GPU CUDA streams never stall waiting for CPU mini-batch collation.

---

## 16. Reproducibility

Scientific reproducibility is guaranteed through strict seed initialization across random number generators:

```text
Central Seed (e.g. 42) ──> Python `random.seed` ──> NumPy `np.random.seed` ──> PyTorch `torch.manual_seed`
```

* **Deterministic Augmentations**: Random spatial transformations draw random seeds derived from PyTorch worker seeds (`torch.utils.data.get_worker_info()`).
* **Configuration Snapshot**: The exact resolved YAML configuration is copied into the experiment output directory (`outputs/logs/`) upon pipeline start.

---

## 17. Integration with NAFNet

The data pipeline interfaces directly with the NAFNet model architecture:

```text
DataLoader Output: Batch Dict {"input": (B, 1, 128, 128), "target": (B, 1, 256, 256)}
                                           |
                                           v
                              +-------------------------+
                              | NAFNet Model            |
                              | (With 2x Upsample Tail) |
                              +-------------------------+
                                           |
                                           v
                              Predicted Tensor (B, 1, 256, 256)
                                           |
                                           v
                              Loss Function vs Target (B, 1, 256, 256)
```

### Architectural Handshake Requirements
1. **Channel Input**: NAFNet `in_channels` must be set to `1` (grayscale SEM input).
2. **Spatial Upsampling Capability**: Because input is $128 \times 128$ and target is $256 \times 256$, NAFNet must either:
   * Incorporate an internal $2\times$ upsampling tail (e.g. `PixelShuffle(2)` or Transposed Convolution), OR
   * The pipeline must bicubically upsample input tensors to $256 \times 256$ prior to model entry (configurable via `configs/train.yaml`).

---

## 18. Future Extensions

The pipeline architecture is designed to support future research expansion without requiring structural refactoring:

* **Blind Denoising & Unpaired Training**: Support unpaired image loading strategies (e.g. Noise2Noise, Neighbor2Neighbor).
* **Synthetic Degradation Injection**: Support online synthetic noise addition (Poisson-Gaussian noise, line jitter, charging blur).
* **Multi-Channel SEM Inputs**: Support multi-detector SEM micrographs (e.g., In-Lens + Everhart-Thornley secondary electron detectors) by expanding input channels to $(C, H, W)$.
* **Distributed Data Parallel (DDP)**: Compatible with `torch.utils.data.distributed.DistributedSampler` for multi-GPU training clusters.
