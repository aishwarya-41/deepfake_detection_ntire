# inference.py
import os
import cv2
import torch
from torchvision import transforms

from model import get_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = get_model().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_folder = "../validation_data_final"  # change to test folder when needed
paths = sorted(os.listdir(test_folder))

with open("submission.txt", "w") as f:
    for p in paths:
        img = cv2.imread(os.path.join(test_folder, p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = model(img).item()

        f.write(f"{prob}\n")

print("submission.txt created")
