# Issue #47 — SEM and Microscopy Image Restoration Methods

## Objective

Review research literature on SEM image restoration, microscopy restoration, denoising, and super-resolution, with emphasis on approaches relevant to the project's paired degraded low-resolution SEM images and clean 2× higher-resolution targets.

---

## 1. SEM Denoising

### Paper / Source
Park et al.; Shin et al.

### Problem Setting
Restoration of SEM micrographs affected by low electron dose or shortened pixel dwell times.

### Restoration Approach
Supervised deep learning using architectures including NAFNet, Restormer, DnCNN, and autoencoder-based approaches.

### Training Data
Paired SEM observations, such as low-frame-count versus high-frame-count acquisitions.

### Degradation Assumptions
The cited SEM studies consider acquisition noise caused by insufficient electron signal, including additive or structured noise patterns.

### Key Results
The cited SEM study reports that NAFNet achieved a +9.09 dB PSNR improvement and approximately 66× acquisition-speed improvement compared with the corresponding multi-frame acquisition setup.

### Limitations
Performance can decrease when acquisition conditions such as focus, alignment, or accelerating voltage differ between training and evaluation data.

### Relevance
High. These studies provide direct SEM-specific evidence for supervised deep-learning restoration and provide context for the project's NAFNet baseline.

**Evidence Classification:** Direct Evidence

---

## 2. Microscopy / FIB-SEM Restoration

### Paper / Source
Xu et al.

### Problem Setting
Removal of curtain/stripe artifacts in FIB-SEM digital-rock images.

### Restoration Approach
Attention-enhanced U-Net and DnCNN.

### Training Data
Paired noisy and cleaned reference images.

### Degradation Assumptions
Structured vertical stripe noise caused by variations in the FIB-SEM acquisition process.

### Key Results
The cited study reports PSNR of 29.10 dB for the attention U-Net compared with 27.28 dB for DnCNN, with improved structural and pore-texture preservation.

### Limitations
Performance depends on the noise characteristics represented in the training data. The cited study also reports weaker generalization for DnCNN on unseen images.

### Relevance
Demonstrates the importance of preserving fine structural information during microscopy restoration.

**Evidence Classification:** Direct Evidence

---

## 3. Supervised Paired Restoration

### Paper / Source
Chen et al. — NAFNet, "Simple Baselines for Image Restoration"

### Problem Setting
Image denoising and deblurring.

### Restoration Approach
Nonlinear Activation Free Network (NAFNet), using SimpleGate and Simplified Channel Attention.

### Training Data
Paired datasets including SIDD and GoPro.

### Degradation Assumptions
The evaluated tasks involve image noise and motion blur.

### Key Results
NAFNet achieved 40.30 dB PSNR on SIDD denoising while requiring substantially less computation than previous methods.

### Limitations
The paper identifies a train/test inconsistency associated with patch training and full-image testing and introduces test-time local-statistics correction.

### Relevance
Provides the architectural foundation for the project's current NAFNet baseline. However, the paper's natural-image results should not be interpreted as evidence that NAFNet will outperform other methods on the project's SEM dataset.

**Evidence Classification:** Direct Evidence

---

## 4. Unpaired Restoration / CycleGAN

### Paper / Source
Shin et al.

### Problem Setting
SEM denoising when paired clean ground truth is unavailable.

### Restoration Approach
CycleGAN, treating restoration as translation between noisy and clean image domains.

### Training Data
Unpaired noisy and clean images.

### Degradation Assumptions
The method treats the difference between noisy and clean images as a domain/style translation problem.

### Key Results
The cited study reports a PSNR of 22.74 dB and reports satisfactory visual results according to expert evaluation.

### Limitations
Unpaired translation introduces mapping ambiguity and can be sensitive to training and hyperparameter choices.

### Relevance
Potentially useful when clean paired targets are unavailable, but the current project already has paired degraded-to-clean data.

**Evidence Classification:** Direct Evidence

---

## 5. Noise2Noise-style Approaches

### Paper / Source
Treder et al.; project data-pipeline documentation.

### Problem Setting
Denoising when clean ground-truth images are difficult to obtain.

### Restoration Approach
Learning from multiple noisy observations of the same underlying scene.

### Training Data
Noisy-noisy pairs rather than conventional noisy-clean pairs.

### Degradation Assumptions
Noise assumptions such as statistical independence and appropriate expectation properties are important to the approach.

### Limitations
The observations must correspond to the same underlying scene and require appropriate registration.

### Relevance
Potential future approach for datasets where high-quality SEM ground truth cannot be obtained. It is not required for the current paired benchmark.

**Evidence Classification:** Project Inference

---

## 6. CARE-style Restoration

### Paper / Source
Treder et al., as referenced by the reviewed project sources.

### Problem Setting
Restoration of low-SNR fluorescence and electron microscopy images.

### Restoration Approach
Content-Aware Image Restoration (CARE), using supervised deep-learning restoration.

### Training Data
Paired degraded and higher-quality observations.

### Degradation Assumptions
The approach addresses microscopy observations affected by low signal-to-noise conditions and resolution limitations.

### Key Finding
The cited work demonstrates that deep-learning restoration can recover useful image information from low-SNR microscopy observations.

### Limitations
The approach relies on suitable training examples representing the relationship between degraded and high-quality observations.

### Relevance
Conceptually relevant because the current project also learns a mapping from degraded microscopy observations to higher-quality targets.

**Evidence Classification:** Strongly Supported

---

## 7. Diffusion-based Restoration

### Paper / Source
Shin et al.; Benjdira et al.

### Problem Setting
High-fidelity image restoration using iterative probabilistic refinement.

### Restoration Approach
Diffusion-based approaches including DDIM and SR3.

### Training Data
The reviewed sources include paired and clean-data training configurations depending on the method.

### Key Results
The cited SEM study reports DISTS of 0.1068 for the DDIM approach and describes preservation of nuanced tonal variations and fine details.

### Limitations
Diffusion restoration generally requires substantially more iterative computation than a single-pass restoration network and may introduce reconstructed details that are not directly supported by the observation.

### Relevance
Provides an alternative restoration paradigm for comparison, particularly for perceptual fidelity, but the cited evidence does not establish superiority over NAFNet for this project.

**Evidence Classification:** Direct Evidence

---

## 8. Image Super-Resolution

### Paper / Source
Chu et al. (NAFSSR); Benjdira et al.; Zhao et al.; Johnson et al.

### Problem Setting
Reconstruction of high-resolution information from low-resolution observations.

### Restoration Approach
Methods include NAFSSR, SwinIR, and approaches using perceptual or frequency-domain losses.

### Training Data
Typically paired low-resolution and high-resolution images.

### Degradation Assumptions
Many standard super-resolution studies generate low-resolution observations using assumed degradation processes such as bicubic downsampling.

### Key Results
The NAFSSR paper reports state-of-the-art stereo super-resolution results with up to 79% parameter reduction relative to competing models.

### Limitations
The cited super-resolution results are primarily from natural-image datasets. They therefore do not directly establish performance on SEM imagery.

### Relevance
Directly relevant to the project's 2× spatial restoration component, while the reported natural-image results must not be treated as SEM evidence.

**Evidence Classification:** Project Inference

---

## 9. Restoration from Noisy + Low-Resolution Observations

### Paper / Source
KLA Webinar Findings; Dataset Characterization Report.

### Problem Setting
The project requires simultaneous restoration of degraded low-resolution SEM observations into clean higher-resolution targets.

### Training Data
Paired 128×128 degraded inputs and 256×256 clean targets.

### Degradation Assumptions
The benchmark contains additive Gaussian noise, multiplicative speckle noise, and downsampling. KLA indicates that their order should not be assumed to be fixed.

### Key Result
The current project NAFNet baseline achieved 29.4118 dB PSNR, representing a +6.5049 dB improvement over the raw noisy baseline.

### Relevance
This is the exact problem formulation of the current benchmark.

### Important Qualification
The project result is evidence about the current implementation and dataset. It is not evidence that NAFNet is universally superior to alternative restoration methods.

**Evidence Classification:** KLA/Project-Confirmed Fact

---

## 10. Natural Image Restoration vs. SEM / Microscopy

Natural-image restoration and SEM/microscopy restoration differ in their acquisition processes, image statistics, and structural requirements.

The reviewed SEM studies demonstrate sensitivity to acquisition conditions and show that geometric structures can be important when evaluating restoration quality.

Standard pixel-level metrics such as PSNR and SSIM therefore provide useful quantitative information but may not fully describe perceptual or structural fidelity.

Perceptual metrics such as LPIPS and DISTS can provide complementary information when assessing structural and visual differences.

Natural-image pretrained models and results should therefore not automatically be assumed to generalize to SEM imagery.

**Evidence Classification:** Strongly Supported

---

# Restoration Paradigm Comparison

| Paradigm | Target Problem | Training Data | Main Strength | Main Limitation | Project Relevance |
|---|---|---|---|---|---|
| Supervised SEM restoration | SEM acquisition noise | Paired | Directly addresses SEM data | Sensitive to acquisition changes | High |
| FIB-SEM restoration | Structured artifacts | Paired | Structural preservation | Depends on represented noise | High |
| NAFNet | General restoration / SEM | Paired | Efficient restoration | SEM generalization still requires validation | High |
| CycleGAN | Unpaired restoration | Unpaired | Does not require clean pairs | Mapping ambiguity | Medium/Low |
| Noise2Noise | Denoising without clean targets | Noisy-noisy pairs | Avoids clean ground truth | Requires suitable repeated observations | Future |
| CARE | Low-SNR microscopy | Paired | Microscopy-specific restoration | Requires suitable paired examples | High |
| Diffusion | High-fidelity restoration | Method-dependent | Strong perceptual refinement | Computational cost | Medium |
| Super-resolution | Resolution recovery | LR-HR pairs | Reconstructs spatial detail | Many studies use natural images | High |

---

# Key Conclusions for the Project

1. **Paired supervised restoration is directly applicable** because the project has degraded SEM inputs and clean higher-resolution targets.

2. **SEM-specific studies provide stronger evidence for model behavior on this domain** than results obtained exclusively from natural-image datasets.

3. **NAFNet is a justified current baseline**, supported by both general restoration research and SEM-specific evidence. However, the reviewed literature does not establish that it is universally optimal for this dataset.

4. **Super-resolution literature is relevant to the 2× restoration component**, but results from natural-image datasets such as Flickr1024 or DIV2K should not be presented as SEM evidence.

5. **Unpaired methods such as CycleGAN and Noise2Noise-style approaches are alternatives**, primarily relevant when suitable clean paired observations are unavailable.

6. **Diffusion-based restoration provides another research direction**, but its computational requirements make direct comparison with a single-pass architecture important before drawing project conclusions.

7. **SEM restoration has domain-specific generalization challenges**, particularly when acquisition conditions change between training and evaluation.

---

# Remaining Research Gaps

- How robust are restoration models to changes in SEM acquisition parameters?
- How well do models trained on the provided paired data generalize to the hidden evaluation distribution?
- What combination of pixel, structural, perceptual, and frequency losses is most appropriate for this SEM task?
- How much of the super-resolution literature transfers from natural images to semiconductor/SEM imagery?
- How do single-pass restoration models compare with more computationally expensive diffusion approaches under the project's evaluation constraints?

These questions require project-specific experiments or additional evidence and should not be answered from the literature alone.