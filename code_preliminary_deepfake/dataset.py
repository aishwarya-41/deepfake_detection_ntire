# dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):
    def __init__(self, folder_path, train=True):
        self.paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        self.train = train

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

    def add_degradation(self, img):
        if np.random.rand() < 0.5:
            img = cv2.GaussianBlur(img, (5,5), 0)

        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)

        if np.random.rand() < 0.5:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, 1)

        return img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.train:
            img = self.add_degradation(img)

        img = self.transform(img)

        label = 1 if "fake" in path else 0
        return img, torch.tensor(label, dtype=torch.float32)
