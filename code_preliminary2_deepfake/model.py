import timm
import torch.nn as nn

def get_model():
    # ConvNeXt is superior for handling distorted/noisy images
    model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True)
    
    # Re-build the head for binary classification
    n_features = model.head.fc.in_features
    model.head.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(n_features, 1)
    )
    return model