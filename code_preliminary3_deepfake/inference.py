import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import get_model

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Ensemble Inference on: {device}")

    # 1. Load BOTH models
    model_best = get_model().to(device)
    model_last = get_model().to(device)
    
    # Matching the new file names from your train.py
    model_best.load_state_dict(torch.load("best_robust_model.pth", map_location=device))
    model_last.load_state_dict(torch.load("last_model.pth", map_location=device))
    
    model_best.eval()
    model_last.eval()

    # 2. TTA Transforms
    tta_transforms = [
        A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()]),
        A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()]),
        A.Compose([A.Resize(224, 224), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), A.Normalize(), ToTensorV2()])
    ]

    test_folder = "../validation_data_final"
    paths = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    with open("submission_ensemble.txt", "w") as f:
        for p in paths:
            img_path = os.path.join(test_folder, p)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            all_probs = []
            
            # Loop through TTA and both models
            for transform in tta_transforms:
                input_tensor = transform(image=img)["image"].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Get probability from Best model
                    prob_best = torch.sigmoid(model_best(input_tensor).squeeze()).item()
                    # Get probability from Last model
                    prob_last = torch.sigmoid(model_last(input_tensor).squeeze()).item()
                    
                    # Average them for this specific TTA view
                    all_probs.append((prob_best + prob_last) / 2)
            
            # Final average of all TTA views
            f.write(f"{np.mean(all_probs)}\n")

    print("Generated submission_ensemble.txt. Good luck with the leaderboard!")

if __name__ == "__main__":
    main()