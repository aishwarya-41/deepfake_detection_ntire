import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import DeepfakeDataset
from model import get_model
from sklearn.metrics import roc_auc_score
import numpy as np

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training for Stability on: {device}")

    full_dataset = DeepfakeDataset("../training_data_final", train=True)
    
    # 900 Train / 100 Val split for much better AUC stability
    train_size = 900
    val_size = 100
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # Using a slightly higher LR for Tiny, but with better weight decay
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_auc = 0.0

    for epoch in range(15):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            smoothed_labels = labels * 0.9 + 0.05
            
            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, smoothed_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs).squeeze()
                prob = torch.sigmoid(logits).item()
                all_preds.append(prob)
                all_labels.append(labels.item())
        
        val_auc = roc_auc_score(all_labels, all_preds)
        scheduler.step()
        
        # Save both Best and Last to allow for Ensemble Inference
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_robust_model.pth")
            status = "--> NEW BEST"
        else:
            status = ""
            
        # Always save the latest version as well
        torch.save(model.state_dict(), "last_model.pth")

        print(f"Epoch {epoch+1}/15 | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} {status}")

if __name__ == "__main__":
    main()