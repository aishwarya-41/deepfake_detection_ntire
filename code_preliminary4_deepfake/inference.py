"""
inference.py  —  Ensemble + TTA Inference for NTIRE Deepfake Detection

Changes vs baseline:
  1. Loads new DeepfakeDetector (dual-branch) architecture
  2. Checkpoint names match new train.py (best_model.pth / last_model.pth)
  3. TTA done via tensor ops (faster, no albumentations loop)
  4. 5-pass TTA: original + hflip + 3 scales
  5. Ensemble of best + last checkpoint averaged
  6. Handles corrupted/unreadable images gracefully
"""

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TEST_FOLDER    = "../validation_data_final"
OUTPUT_FILE    = "submission_ensemble.txt"
BEST_CKPT      = "best_model.pth"
LAST_CKPT      = "last_model.pth"
IMG_SIZE       = 224
BEST_WEIGHT    = 0.6   # best checkpoint gets slightly more weight
LAST_WEIGHT    = 0.4


# ─────────────────────────────────────────────
# Preprocessing (no augmentation — TTA via tensor ops)
# ─────────────────────────────────────────────
preprocess = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def load_image(path):
    """Returns preprocessed tensor (1, 3, H, W) or None if unreadable."""
    img = cv2.imread(path)
    if img is None:
        print(f"  [WARN] Could not read {path}, skipping.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess(image=img)["image"].unsqueeze(0)   # (1, 3, H, W)


# ─────────────────────────────────────────────
# TTA — 5 passes via tensor ops (fast)
# ─────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, tensor, device):
    """
    Returns averaged probability across 5 augmented views:
      1. Original
      2. Horizontal flip
      3. Scale 0.90
      4. Scale 0.95
      5. Scale 1.05
    """
    x     = tensor.to(device)
    size  = (IMG_SIZE, IMG_SIZE)
    views = [
        x,
        x.flip(-1),
        F.interpolate(F.interpolate(x, scale_factor=0.90, mode='bilinear', align_corners=False), size=size, mode='bilinear', align_corners=False),
        F.interpolate(F.interpolate(x, scale_factor=0.95, mode='bilinear', align_corners=False), size=size, mode='bilinear', align_corners=False),
        F.interpolate(F.interpolate(x, scale_factor=1.05, mode='bilinear', align_corners=False), size=size, mode='bilinear', align_corners=False),
    ]
    probs = [torch.sigmoid(model(v)).item() for v in views]
    return float(np.mean(probs))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))
    print(f"Running Ensemble Inference on: {device}")

    # ── Load both checkpoints ─────────────────────────────────────────
    model_best = get_model().to(device)
    model_last = get_model().to(device)

    model_best.load_state_dict(torch.load(BEST_CKPT, map_location=device))
    model_last.load_state_dict(torch.load(LAST_CKPT, map_location=device))

    model_best.eval()
    model_last.eval()
    print(f"Loaded: {BEST_CKPT}  +  {LAST_CKPT}")

    # ── Get sorted image list ─────────────────────────────────────────
    exts  = ('.png', '.jpg', '.jpeg', '.webp')
    paths = sorted([f for f in os.listdir(TEST_FOLDER)
                    if f.lower().endswith(exts)])
    print(f"Found {len(paths)} images in {TEST_FOLDER}")

    # ── Inference loop ────────────────────────────────────────────────
    results = []
    for i, fname in enumerate(paths):
        img_path = os.path.join(TEST_FOLDER, fname)
        tensor   = load_image(img_path)

        if tensor is None:
            results.append(0.5)    # neutral score for unreadable images
            continue

        prob_best = tta_predict(model_best, tensor, device)
        prob_last = tta_predict(model_last, tensor, device)

        # Weighted ensemble: best checkpoint gets slightly more trust
        final_prob = BEST_WEIGHT * prob_best + LAST_WEIGHT * prob_last
        results.append(final_prob)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(paths)} ...")

    # ── Write submission ──────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        for prob in results:
            f.write(f"{prob}\n")

    print(f"\nDone! {len(results)} predictions written to {OUTPUT_FILE}")
    print("Good luck on the leaderboard! 🚀")


if __name__ == "__main__":
    main()
