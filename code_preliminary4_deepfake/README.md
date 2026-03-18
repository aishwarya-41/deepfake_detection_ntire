# Deepfake Detection – NTIRE 2026 Submission

## Overview

This project implements a **robust deepfake detection system** using a ConvNeXt-based architecture with a frequency-aware branch and regularized training strategy.

The pipeline includes:

* Robust data augmentation
* Dual-branch feature extraction (RGB + FFT)
* Balanced sampling for class imbalance
* Regularized training (mixup, label smoothing, early stopping)
* Lightweight test-time augmentation for fast inference

---

## Project Structure

```
project/
│
├── dataset.py
├── model.py
├── train.py
├── inference_solo.py
├── inference.py
│
├── best_model.pth        # generated after training
├── submission.txt        # generated after inference
```

---

## Dataset Setup

You must provide the dataset **outside the code directory**.

Example structure:

```
../training_data_final/
        000_fake.jpg
        0001_real.jpg
        0002_fake.jpg
        0003_real.jpg

../publictest_data_final/
        test1.jpg
        test2.jpg
```

Notes:

* Labels are automatically inferred from file names (`real` / `fake`)
* Test folder should contain **only images (no labels)**

---

## Installation

Install required dependencies:

```
pip install torch torchvision timm albumentations opencv-python scikit-learn
```

---

## Training

Run the following command:

```
python train.py
```

This will:

* Train the model
* Save the best checkpoint as:

  ```
  best_model.pth
  ```
* Also save the latest checkpoint:

  ```
  last_model.pth
  ```

---

## Inference

After training, run:

```
python inference.py
```

This will:

* Load the best model (`best_model.pth`) and (`last_model.pth`)
* Perform inference on the test dataset
* Apply lightweight test-time augmentation (horizontal flip)
* Generate the submission file:

```
submission.txt
```

---

## Output Format

The output file `submission.txt` contains one prediction per line:

```
0.8732
0.1245
0.6521
...
```

Each value represents the probability of the image being **fake**.

---

## Performance

The model achieves an AUC of:

```
AUC = 0.68 (on validation)
AUC = 0.686 (on public test)
```
This shows its robust to different kinda of data and works the same for both

---

## Notes

* Training uses balanced sampling to handle class imbalance.
* Frequency-domain features are incorporated using FFT.
* Backbone is frozen initially and later fine-tuned.
* Early stopping is used to prevent overfitting.
* Inference is optimized for speed and leaderboard constraints.

---

## Authors

* Aishwarya A
* Akshara S
* Ashwathi N

Team: **Acube**

