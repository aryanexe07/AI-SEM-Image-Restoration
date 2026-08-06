# Global Engineering Context — Project Implementation Charter

## IMPORTANT

Read and internalize this entire document before making **any** implementation decisions.

This document defines the engineering philosophy, architectural principles, coding standards, optimization strategy, and long-term vision of the project.

Every implementation decision must align with these principles.

If there is ever a conflict between writing code quickly and writing code correctly, **prioritize correctness, maintainability, reproducibility, and performance**.

---

# Project

**AI-Based Restoration of Degraded Scanning Electron Microscope (SEM) Images using NAFNet**

---

# Mission

We are **NOT** building a simple university assignment.

We are building a **research-grade, highly optimized, modular, scalable, and production-quality deep learning framework** for SEM image restoration.

The final repository should be comparable in quality to well-maintained open-source computer vision projects.

Every design and implementation decision must support this objective.

---

# Primary Goal

Develop an optimized framework capable of restoring degraded SEM images using the NAFNet architecture while maximizing image quality and maintaining efficient training and inference.

Primary evaluation metrics:

* PSNR
* SSIM

Secondary engineering goals:

* High performance
* Low memory overhead
* Efficient GPU utilization
* Modular architecture
* Excellent documentation
* Complete reproducibility
* Easy extensibility

---

# Engineering Philosophy

Every module must satisfy the following principles.

## 1. Performance First

Performance is a core requirement—not an afterthought.

Design for:

* Efficient data loading
* High GPU utilization
* Low CPU bottlenecks
* Minimal memory copies
* Mixed Precision (AMP)
* Efficient tensor operations
* Fast inference
* Scalable training

Never introduce unnecessary computation.

---

## 2. Measure Before Optimizing

Never optimize blindly.

Every optimization must follow:

Measure $\to$ Profile $\to$ Analyze $\to$ Optimize $\to$ Benchmark $\to$ Validate

Do not sacrifice readability for theoretical optimizations that have not been measured.

---

## 3. Research Quality

The project must support reproducible scientific experimentation.

Every experiment should be:

* Reproducible
* Configurable
* Logged
* Versioned
* Benchmarkable

No hidden parameters.  
No undocumented assumptions.  
No hardcoded experimental settings.  

---

## 4. Modular Architecture

Every module should have one clear responsibility.

Example:

Dataset $\to$ Transforms $\to$ DataLoader $\to$ Model $\to$ Loss $\to$ Trainer $\to$ Metrics $\to$ Visualization $\to$ Results

Each module should be independently testable and replaceable.

Avoid tight coupling.

---

## 5. Configuration-Driven Development

Nothing should be hardcoded.

Everything configurable should live in YAML configuration files.

Examples:

* Dataset path
* Batch size
* Patch size
* Learning rate
* Optimizer
* Scheduler
* Checkpoint paths
* Logging
* Inference parameters

If something may change between experiments, it belongs in configuration.

---

## 6. Clean Code

Every implementation must include:

* Type hints
* Google-style docstrings
* Clear naming
* Small functions
* Single Responsibility Principle
* Absolute imports
* PEP 8 compliance

Avoid clever code that reduces readability.

---

## 7. Scalability

Design every component so the project can later support:

* Multiple restoration architectures
* Multiple datasets
* Distributed training
* Multi-GPU execution
* ONNX export
* TensorRT deployment
* Quantization
* Future restoration methods

Avoid architecture that locks the project to NAFNet.

---

## 8. Reliability

Every module should:

* Validate inputs.
* Handle errors gracefully.
* Produce informative error messages.
* Fail safely rather than silently.

Never ignore exceptions without justification.

---

## 9. Reproducibility

Every experiment must be reproducible.

Include:

* Random seed management
* Configuration snapshots
* Experiment logging
* Version tracking
* Checkpoint metadata

Results should be repeatable.

---

## 10. Documentation

Every public module should be documented.

Explain:

* Purpose
* Inputs
* Outputs
* Design decisions
* Limitations
* Future extension points

Documentation is part of the implementation.

---

# Optimization Guidelines

During implementation always consider:

* Memory usage
* CPU utilization
* GPU utilization
* Data loading speed
* Disk I/O
* Batch throughput
* Inference latency
* Training throughput
* Mixed Precision compatibility
* Future distributed training

Only optimize when supported by profiling or measurable evidence.

---

# Coding Standards

Follow:

* Python 3.11
* PyTorch best practices
* PEP 8
* Black
* Ruff
* isort
* pytest
* Google-style docstrings
* Full type hints

No wildcard imports.  
No global mutable state.  
No duplicated logic.  

---

# Repository Rules

* Respect the existing architecture.
* Do not arbitrarily reorganize folders.
* Do not introduce unnecessary dependencies.
* Do not create files outside the documented project structure.

---

# Decision-Making Rules

Before implementing any feature, ask:

1. Does this improve maintainability?
2. Is this modular?
3. Is it configuration-driven?
4. Is it efficient?
5. Is it reproducible?
6. Is it testable?
7. Can it scale?
8. Does it follow the existing architecture?

If any answer is "No", reconsider the design.

---

# Existing Project Context

Assume the following work has already been completed:

* Repository architecture
* README
* Dataset characterization
* Data pipeline design
* Software architecture specification

Do not redesign these documents unless a verified issue is discovered.

Build on the existing architecture.

---

# Implementation Order

Always follow this sequence:

1. Configuration System
2. Dataset Module
3. Transform Pipeline
4. DataLoader
5. Metrics
6. Utilities
7. Training Engine
8. NAFNet Model
9. Validation Engine
10. Inference Pipeline
11. Visualization
12. Hyperparameter Tuning
13. Benchmarking
14. Documentation Refinement

Do not skip ahead unless explicitly instructed.

---

# Definition of Done

A task is only complete when it is:

* Correct
* Optimized where justified
* Documented
* Type-safe
* Tested (where applicable)
* Configurable
* Consistent with the project architecture
* Ready for future extension

---

# Final Directive

Do not optimize for writing code quickly.

Optimize for building a framework that is:

* Research-grade
* Production-quality
* Highly optimized
* Modular
* Maintainable
* Reproducible
* Extensible
* Scientifically rigorous

When multiple valid implementations exist, explain the trade-offs and choose the solution that best aligns with these engineering principles rather than the shortest implementation.
