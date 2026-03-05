import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    def __init__(self, folder_path, train=True):
        self.paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.train = train

        if self.train:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                # --- EXTREME DEGRADATION FOR NTIRE ---
                A.OneOf([
                    A.GaussianBlur(blur_limit=(7, 15), p=0.5),
                    A.MotionBlur(blur_limit=(7, 15), p=0.3),
                    A.Defocus(radius=(3, 7), p=0.2),
                ], p=0.8),
                A.OneOf([
                    A.GaussNoise(var_limit=(20.0, 100.0), p=0.5),
                    A.ImageCompression(quality_lower=20, quality_upper=60, p=0.5),
                ], p=0.7),
                A.Downscale(scale_min=0.2, scale_max=0.5, p=0.4), 
                # ------------------------------------
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        label = 1.0 if "fake" in os.path.basename(path).lower() else 0.0
        return image, torch.tensor(label, dtype=torch.float32)