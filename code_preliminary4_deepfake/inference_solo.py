import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model


TEST_FOLDER = "../publictest_data_final"
OUTPUT_FILE = "submission.txt"
BEST_CKPT   = "best_model.pth"
IMG_SIZE    = 224


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
        print(f"  [WARN] Could not read {path}, using neutral score.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess(image=img)["image"].unsqueeze(0)



# Light TTA 
@torch.no_grad()
def fast_tta_predict(model, tensor, device):
    x = tensor.to(device)
    probs = [
        torch.sigmoid(model(x)).item(),
        torch.sigmoid(model(x.flip(-1))).item()
    ]
    return float(np.mean(probs))


def main():
    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))
    print(f"Running Fast Inference on: {device}")

    model = get_model().to(device)
    model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
    model.eval()
    print(f"Loaded: {BEST_CKPT}")

    exts  = ('.png', '.jpg', '.jpeg', '.webp')
    paths = sorted([f for f in os.listdir(TEST_FOLDER)
                    if f.lower().endswith(exts)])
    print(f"Found {len(paths)} images in {TEST_FOLDER}")

    results = []
    for i, fname in enumerate(paths):
        img_path = os.path.join(TEST_FOLDER, fname)
        tensor   = load_image(img_path)

        if tensor is None:
            results.append(0.5)   # neutral score
            continue

        prob = fast_tta_predict(model, tensor, device)
        results.append(prob)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(paths)} ...")

    with open(OUTPUT_FILE, "w") as f:
        for prob in results:
            f.write(f"{prob}\n")

    print(f"\n{len(results)} predictions written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()