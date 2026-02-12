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
    print(f"Training on: {device}")

    # 1. Load Dataset
    full_dataset = DeepfakeDataset("../training_data_final", train=True)
    
    # 950 Train / 50 Val split
    train_size = 950
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 2. Initialize Model
    model = get_model().to(device)
    
    # 3. Label Smoothing (Crucial for NTIRE Robustness)
    # This prevents the model from being "too certain" about noisy training data
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_auc = 0.0

    print("Starting training...")
    for epoch in range(10): # Increased to 10 as we are saving "Best"
        model.train()
        train_loss = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Apply Label Smoothing manually
            # 0 -> 0.05, 1 -> 0.95
            smoothed_labels = labels * 0.9 + 0.05
            
            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, smoothed_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 4. Validation Loop
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs).squeeze()
                prob = torch.sigmoid(logits).item()
                all_preds.append(prob)
                all_labels.append(labels.item())
        
        val_auc = roc_auc_score(all_labels, all_preds)
        scheduler.step()
        
        # 5. Save the BEST model, not the last one
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "robust_model.pth")
            save_msg = "--> Best Model Saved!"
        else:
            save_msg = ""

        print(f"Epoch {epoch+1}/10 | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} {save_msg}")

    print(f"\nTraining finished. Best Val AUC achieved: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()