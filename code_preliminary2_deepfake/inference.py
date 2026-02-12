import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import get_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = get_model().to(device)
model.load_state_dict(torch.load("robust_model.pth", map_location=device))
model.eval()

# Inference Transform
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_folder = "../validation_data_final"
paths = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

with open("submission2.txt", "w") as f:
    for p in paths:
        img_path = os.path.join(test_folder, p)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        input_tensor = transform(image=img)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor).item()
            # Convert logit to probability
            prob = 1 / (1 + np.exp(-logits))

        f.write(f"{prob}\n")

print(f"Generated submission.txt with {len(paths)} entries.")