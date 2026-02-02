import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset import DeepfakeDataset
from model import get_model


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_folder = "../training_data_final"

    dataset = DeepfakeDataset(train_folder, train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = get_model().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs).squeeze()
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader)}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


if __name__ == "__main__":
    main()
