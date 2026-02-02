# model.py
import timm
import torch.nn as nn

def get_model():
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 1),
        nn.Sigmoid()
    )
    return model
