"""
AdaptSign — Backbone Model
ResNet18 fine-tuned for traffic sign classification.
Extended to support 51 classes (43 GTSRB + 8 Indian)
"""

import torch
import torch.nn as nn
from torchvision import models


class AdaptSignBackbone(nn.Module):
    """
    CNN backbone based on ResNet18.
    Pretrained on ImageNet, fine-tuned for traffic signs.

    Why ResNet18 (not Dai et al.'s full transformer)?
    - Faster to train on laptop/Colab
    - Still highly accurate (94%+ on GTSRB)
    - EWC works the same regardless of backbone
    - Swap in Dai et al.'s backbone later for Sem 7/8

    Architecture:
        ResNet18 → FC(512→256) → ReLU → Dropout → FC(256→num_classes)
    """

    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Remove original FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone if requested (faster training)
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.features(x)  # (B, 512, 1, 1)
        out = self.classifier(features)
        return out

    def get_features(self, x):
        """
        Return feature embeddings (before classifier).
        Used by DriftDetector to measure domain shift.
        """
        with torch.no_grad():
            features = self.features(x)
            features = features.view(features.size(0), -1)
        return features

    def add_classes(self, n_new_classes):
        """
        Expand classifier for new sign classes (Indian signs).
        IKS: Lok Vigyan — extend knowledge to local Indian context
        """
        old_fc = self.classifier[-1]
        old_out = old_fc.out_features
        new_out = old_out + n_new_classes

        # New FC with old weights preserved
        new_fc = nn.Linear(old_fc.in_features, new_out)
        with torch.no_grad():
            new_fc.weight[:old_out] = old_fc.weight
            new_fc.bias[:old_out] = old_fc.bias

        self.classifier[-1] = new_fc
        self.num_classes = new_out
        print(f"[BACKBONE] Expanded: {old_out} → {new_out} classes")


def get_model(num_classes=43, pretrained=True, device='cpu'):
    """
    Factory function to create and return model on device.
    """
    model = AdaptSignBackbone(
        num_classes=num_classes,
        pretrained=pretrained
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_params:,} | Trainable: {trainable:,}")

    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(num_classes=43, device=device)

    # Test forward pass
    dummy = torch.randn(4, 3, 64, 64).to(device)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (4, 43)

    feats = model.get_features(dummy)
    print(f"Features shape: {feats.shape}")  # (4, 512)

    print("Backbone OK ✓")
