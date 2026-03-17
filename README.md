# Deepfake Detection – NTIRE 2026 Submission

## Overview

This repository contains multiple iterations of our deepfake detection models developed for the **NTIRE 2026 Robust Deepfake Detection Challenge**.

The project evolved through several stages, with each version improving robustness, training stability, and performance. The **final model used for submission is in `code_preliminary4_deepfake`**.

---

## Repository Structure

```text
.
├── code_preliminary1_deepfake   # Initial model
├── code_preliminary2_deepfake   # Improved augmentations
├── code_preliminary3_deepfake   # Ensemble + TTA experiments
├── code_preliminary4_deepfake   # FINAL MODEL (used for submission)
```

---

## Version to Use

**Use this folder:**

```text
code_preliminary4_deepfake
```

This version includes:

* ConvNeXt-Small backbone
* Frequency-domain branch (FFT)
* Balanced training (WeightedRandomSampler)
* Regularized training (mixup, label smoothing, early stopping)
* Optimized inference with lightweight TTA

---


## Notes

* Earlier versions are kept for reference and experimentation.
* Only `code_preliminary4_deepfake` reflects the final optimized pipeline.
* The model is designed for **robust detection under degraded conditions**.

---

## Authors

* Aishwarya A
* Akshara S
* Ashwathi N

Team: **Acube**
