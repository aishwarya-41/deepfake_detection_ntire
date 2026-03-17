import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepfakeDataset(Dataset):
    def __init__(self, folder_path, train=True, img_size=224):
        self.img_size = img_size
        self.train    = train

        exts = ('.png', '.jpg', '.jpeg', '.webp')
        self.paths = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(exts):
                    self.paths.append(os.path.join(root, f))

        if not self.paths:
            raise RuntimeError(f"No images found in {folder_path}")

        self.labels = [self._get_label(p) for p in self.paths]
        n_fake = sum(self.labels)
        n_real = len(self.labels) - n_fake
        print(f"[Dataset] {'Train' if train else 'Val'}: "
              f"{len(self.paths)} images | real={n_real} fake={n_fake}")

        self.transform = self._build_train_transform() if train \
                         else self._build_val_transform()

    @staticmethod
    def _get_label(path: str) -> float:
        """
        Priority:
          1. Parent folder name contains 'fake' or 'real'
          2. Filename contains 'fake'
          3. Default → real (0)
        """
        parts = path.lower().replace('\\', '/').split('/')
        for part in reversed(parts[:-1]):         
            if 'fake' in part:
                return 1.0
            if 'real' in part:
                return 0.0
            
        fname = os.path.basename(path).lower()
        return 1.0 if 'fake' in fname else 0.0

    def get_sample_weights(self):
        """
        Returns per-sample weights for WeightedRandomSampler
        so each batch is ~50/50 real/fake regardless of dataset balance.
        """
        labels  = np.array(self.labels)
        n_fake  = labels.sum()
        n_real  = len(labels) - n_fake
        w_fake  = 1.0 / n_fake  if n_fake > 0 else 0
        w_real  = 1.0 / n_real  if n_real > 0 else 0
        weights = np.where(labels == 1, w_fake, w_real)
        return torch.tensor(weights, dtype=torch.float32)

    # Augmentation pipelines 
    def _build_train_transform(self):
        s = self.img_size
        return A.Compose([
            A.RandomResizedCrop(s,s, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08,
                               rotate_limit=8, p=0.4),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
                A.Defocus(radius=(2, 4)),
            ], p=0.5),

            A.ImageCompression(quality_lower=40, quality_upper=95, p=0.6),

            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),

            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.1, hue=0.04, p=0.4),
            A.RandomBrightnessContrast(p=0.3),

            A.CoarseDropout(
                max_holes=6,
                max_height=24,
                max_width=24,
                fill_value=0,
                p=0.2
            ),

            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _build_val_transform(self):
        s = self.img_size
        return A.Compose([
            A.Resize(height=s,width=s),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        image = cv2.imread(path)
        if image is None:

            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label
