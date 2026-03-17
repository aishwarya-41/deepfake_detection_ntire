import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(32 * 16, out_dim)

    def forward(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft  = torch.fft.fft2(gray)
        mag  = torch.log1p(torch.abs(torch.fft.fftshift(fft))).unsqueeze(1)
        mag  = (mag - mag.amin(dim=(-2, -1), keepdim=True)) / \
               (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return self.fc(self.conv(mag).flatten(1))


class DeepfakeDetector(nn.Module):
    def __init__(self,
                 backbone_name='convnext_small.fb_in22k_ft_in1k',
                 freq_dim=64,
                 drop=0.5):       
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name, pretrained=True,
            num_classes=0, global_pool='avg'
        )
        with torch.no_grad():
            rgb_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]

        self.freq = FrequencyBranch(out_dim=freq_dim)

        fused = rgb_dim + freq_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Dropout(drop),
            nn.Linear(fused, 256),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        rgb_feat  = self.backbone(x)
        freq_feat = self.freq(x)
        fused     = torch.cat([rgb_feat, freq_feat], dim=1)
        return self.head(fused).squeeze(1)


def get_model(backbone_name='convnext_small.fb_in22k_ft_in1k'):
    return DeepfakeDetector(backbone_name=backbone_name)


if __name__ == '__main__':
    model  = get_model()
    dummy  = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    print(f"Output shape: {logits.shape}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n/1e6:.1f}M")
