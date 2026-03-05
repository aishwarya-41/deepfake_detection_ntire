import timm
import torch.nn as nn

def get_model():
    # Back to Tiny for stability on 1k samples
    model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True)
    n_features = model.head.fc.in_features
    
    # Simpler, more stable head
    model.head.fc = nn.Sequential(
        nn.BatchNorm1d(n_features),
        nn.Linear(n_features, 1) # Directly to output to reduce overfitting
    )
    return model