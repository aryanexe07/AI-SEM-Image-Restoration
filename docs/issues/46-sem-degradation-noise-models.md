# Issue #46 — SEM Degradation and Noise Models

## Objective

Review credible research literature concerning degradation mechanisms relevant to SEM image restoration.

The project benchmark, based on the KLA webinar findings, uses three confirmed degradation mechanisms:

1. Additive Gaussian noise
2. Multiplicative speckle noise
3. 2× downsampling

KLA also stated that the order of these degradations should not be assumed to be fixed.

---

## 1. Additive Gaussian Noise

### Finding

The project benchmark explicitly identifies Gaussian noise as an additive degradation.

**Evidence classification:** Direct Evidence — KLA/project-confirmed.

### SEM literature

SEM literature indicates that reduced pixel dwell time decreases the signal-to-noise ratio (SNR), producing increased acquisition noise. This provides a physical basis for noise occurring under high-speed or low-dose acquisition conditions.

**Sources:** Park et al.; Shin et al.

**Evidence classification:** Direct Evidence — SEM literature.

### Relevance to the benchmark

Gaussian noise is one of the confirmed degradation components that the restoration model must handle.

---

## 2. Multiplicative Speckle Noise

### Finding

The project benchmark explicitly identifies speckle noise as a multiplicative degradation.

**Evidence classification:** Direct Evidence — KLA/project-confirmed.

### Dataset observation

The project `NoisyLR` dataset contains pixel values extending outside the standard normalized range, approximately from -0.27 to 2.15.

This demonstrates that the dataset contains out-of-range intensity values. However, the pixel range alone does not establish the individual physical contribution of Gaussian versus speckle noise.

**Source:** Dataset Characterization Report.

**Evidence classification:** Project Inference.

### Relevance to the benchmark

The restoration model must operate on the degradation characteristics present in the provided dataset rather than assuming that all inputs follow a normalized [0,1] intensity distribution.

---

## 3. Downsampling

### Finding

The benchmark includes a 2× spatial resolution degradation. The project dataset maps 128×128 low-resolution inputs to 256×256 targets.

**Sources:** KLA Webinar Findings; Dataset Characterization Report.

**Evidence classification:** Direct Evidence — KLA/project-confirmed.

### Synthetic downsampling

Research on image restoration indicates that conventional interpolation-based degradation can alter high-frequency image information. Therefore, synthetic low-resolution generation does not necessarily reproduce the characteristics of images acquired directly from an imaging system.

**Source:** Benjdira et al.

**Evidence classification:** Direct Evidence — general restoration literature.

### Relevance to the benchmark

The project should distinguish between conclusions supported by the benchmark's actual paired data and assumptions about how a generic image-processing pipeline would produce degradation.

---

## 4. Degradation Ordering

### Finding

KLA explicitly states that Gaussian noise, speckle noise, and downsampling may occur in different orders. The specific sequence should therefore not be assumed to be fixed.

**Source:** KLA Webinar Findings.

**Evidence classification:** Direct Evidence — KLA/project-confirmed.

### Relevance to the benchmark

The restoration task should be considered as recovery from combined degradations without assuming a known degradation sequence.

Any claim about the exact physical order used to generate a particular sample remains unknown unless supported by project documentation.

---

## 5. Synthetic Degradation Limitations

The project documentation states that the provided `NoisyLR` inputs already contain native degradation and that additional synthetic noise injection is not part of the current data pipeline.

This indicates that artificially adding generic noise may not necessarily reproduce the native degradation distribution present in the benchmark.

**Source:** Data Pipeline Design Specification.

**Evidence classification:** Project Inference.

General image-restoration literature also indicates that conventional synthetic degradation processes can alter image characteristics, particularly high-frequency information.

**Source:** Benjdira et al.

**Evidence classification:** Direct Evidence — general restoration literature.

### Research implication

The degree to which the provided training data represents the exact degradation distribution of the hidden evaluation data remains an empirical question.

---

## 6. Severity Variation

### Benchmark evidence

The KLA/project documentation indicates that the hidden evaluation data may contain different degradation severity levels from the training data.

**Source:** KLA Webinar Findings.

**Evidence classification:** Direct Evidence — KLA/project-confirmed.

### SEM literature

SEM restoration performance can be affected by changes in acquisition conditions such as focus, beam alignment, and accelerating voltage.

**Source:** Park et al.

**Evidence classification:** Direct Evidence — SEM literature.

### Relevance to the benchmark

These findings indicate that generalization across variations in degradation severity and acquisition conditions is an important consideration for the restoration task.

The exact distribution of the hidden evaluation data is not publicly established.

**Evidence classification:** Unknown / Insufficient Evidence.

---

## 7. Detector and Instrument Effects

### Charging

SEM literature reports that charging on non-conductive specimens can produce image distortion, signal loss, wash-out/saturation, and edge-bloom effects.

**Source:** Park et al.

**Evidence classification:** Direct Evidence — SEM literature.

### Drift and scan-line effects

Uncorrected inter-frame drift and scan-line shifts can introduce geometric errors during image acquisition.

**Source:** Park et al.

**Evidence classification:** Direct Evidence — SEM literature.

### Relevance to the benchmark

These effects demonstrate that real SEM imagery can contain instrument-related artifacts beyond simple statistical noise models.

However, these effects should **not automatically be considered part of the benchmark degradation definition**, because the KLA-confirmed benchmark definition specifically identifies Gaussian noise, speckle noise, and downsampling.

**Evidence classification:** Strongly Supported / Project Boundary.

---

## 8. Research Gaps

The following questions remain unresolved for the specific benchmark:

1. What is the exact degradation distribution of the hidden evaluation set?
2. What is the exact severity distribution of Gaussian and speckle noise in the hidden set?
3. What exact physical or synthetic process produced the paired low-resolution inputs?
4. What is the exact sequence of degradation operations for individual samples?
5. How closely does the degradation distribution in the provided training data match the hidden evaluation data?
6. What exact weighting, if any, is used to combine PSNR, SSIM, and LPIPS in the final benchmark score?

These questions cannot be established from the currently reviewed sources alone.

**Evidence classification:** Unknown / Insufficient Evidence.

---

## 9. Conclusion

The reviewed project documentation and SEM literature establish that the benchmark involves three confirmed degradation mechanisms: additive Gaussian noise, multiplicative speckle noise, and 2× downsampling. KLA also states that their order should not be assumed to be fixed.

SEM literature provides physical evidence that acquisition conditions can influence image quality and that real SEM images may contain instrument-related effects such as charging and drift. These effects demonstrate the limitations of treating SEM degradation as a purely generic image-processing problem.

However, instrument-related effects should not be added to the benchmark degradation definition without explicit project evidence.

The major remaining uncertainty is the relationship between the provided training degradation and the hidden evaluation distribution. This requires experimental generalization analysis rather than assumption.

---

## Evidence Classification

| Classification | Meaning |
|---|---|
| Direct Evidence | Explicitly supported by KLA/project documentation or cited research literature |
| Strongly Supported | Supported by multiple relevant sources but requires some interpretation |
| Project Inference | Interpretation of evidence specifically for this project |
| Unknown / Insufficient Evidence | Not established by the reviewed sources |

## Sources

- KLA Webinar Findings
- Dataset Characterization Report
- Data Pipeline Design Specification
- Park et al.
- Shin et al.
- Benjdira et al.