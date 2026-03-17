import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

from dataset import DeepfakeDataset
from model   import get_model


DATA_DIR      = "../training_data_final"
IMG_SIZE      = 224
EPOCHS        = 30
BATCH_SIZE    = 16
LR            = 1e-4      
WEIGHT_DECAY  = 5e-2       
VAL_FRAC      = 0.15      
SEED          = 42
USE_MIXUP     = True
MIXUP_ALPHA   = 0.2       
USE_TTA       = True
FREEZE_EPOCHS = 5          
PATIENCE      = 7         
SAVE_BEST     = "best_model.pth"
SAVE_LAST     = "last_model.pth"



class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.s = smoothing

    def forward(self, logits, targets):
        targets = targets * (1 - self.s) + 0.5 * self.s
        return F.binary_cross_entropy_with_logits(logits, targets)


def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)


@torch.no_grad()
def tta_predict(model, imgs, device):
    model.eval()
    x    = imgs.to(device)
    size = (IMG_SIZE, IMG_SIZE)
    views = [
        x,
        x.flip(-1),
        F.interpolate(F.interpolate(x, scale_factor=0.90, mode='bilinear',
                      align_corners=False), size=size, mode='bilinear', align_corners=False),
        F.interpolate(F.interpolate(x, scale_factor=0.95, mode='bilinear',
                      align_corners=False), size=size, mode='bilinear', align_corners=False),
        F.interpolate(F.interpolate(x, scale_factor=1.05, mode='bilinear',
                      align_corners=False), size=size, mode='bilinear', align_corners=False),
    ]
    probs = [torch.sigmoid(model(v)) for v in views]
    return torch.stack(probs).mean(0)


def set_backbone_grad(model, requires_grad: bool):
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad



def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))
    print(f"Device: {device}")

    full_train_ds = DeepfakeDataset(DATA_DIR, train=True,  img_size=IMG_SIZE)
    full_val_ds   = DeepfakeDataset(DATA_DIR, train=False, img_size=IMG_SIZE)

    labels  = full_train_ds.labels
    indices = list(range(len(full_train_ds)))
    tr_idx, val_idx = train_test_split(
        indices, test_size=VAL_FRAC, stratify=labels, random_state=SEED
    )

    train_ds = Subset(full_train_ds, tr_idx)
    val_ds   = Subset(full_val_ds,   val_idx)
    print(f"Split → train: {len(train_ds)}  val: {len(val_ds)}")

    all_weights   = full_train_ds.get_sample_weights()
    train_weights = all_weights[tr_idx]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = get_model().to(device)

    set_backbone_grad(model, False)
    print(f"Backbone FROZEN for first {FREEZE_EPOCHS} epochs")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = LabelSmoothingBCE(smoothing=0.05)
    use_amp   = device.type == 'cuda'
    scaler    = GradScaler(enabled=use_amp)

    best_val_auc  = 0.0
    patience_ctr  = 0

    for epoch in range(EPOCHS):

        if epoch == FREEZE_EPOCHS:
            set_backbone_grad(model, True)
            backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
            head_params     = [p for n, p in model.named_parameters() if 'backbone' not in n]
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': LR * 0.1},
                {'params': head_params,     'lr': LR},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-6
            )
            print(f"Epoch {epoch+1}: Backbone UNFROZEN — fine-tuning with LR/10")
        model.train()
        train_loss = 0.0
        for imgs, labels_batch in train_loader:
            imgs         = imgs.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            if USE_MIXUP:
                imgs, ya, yb, lam = mixup(imgs, labels_batch, MIXUP_ALPHA)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(imgs)
                loss   = (mixup_loss(criterion, logits, ya, yb, lam)
                          if USE_MIXUP else criterion(logits, labels_batch))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                probs = tta_predict(model, imgs, device) if USE_TTA \
                        else torch.sigmoid(model(imgs.to(device)))
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels_batch.numpy())

        val_auc  = roc_auc_score(all_labels, all_preds)
        avg_loss = train_loss / len(train_loader)

        status = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), SAVE_BEST)
            status      = "  ← NEW BEST"
            patience_ctr = 0
        else:
            patience_ctr += 1

        torch.save(model.state_dict(), SAVE_LAST)

        current_lr = optimizer.param_groups[-1]['lr']
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e}{status}")

        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    print(f"\nTraining complete. Best Val AUC: {best_val_auc:.4f}")
    print(f"Best model saved to: {SAVE_BEST}")


if __name__ == "__main__":
    main()
